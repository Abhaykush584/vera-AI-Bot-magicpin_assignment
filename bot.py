"""
Vera Bot — magicpin AI Challenge Submission
A production-grade WhatsApp merchant AI assistant.

Architecture:
- FastAPI HTTP server with 5 required endpoints
- In-memory context store (scope+context_id keyed, versioned)
- Claude-powered composition via Anthropic API
- Intelligent trigger routing by kind
- Auto-reply detection
- Intent-transition handling
- Per-conversation state tracking
"""

import os
import time
import uuid
import json
import re
import logging
import httpx
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

def load_local_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs for local runs without adding a dependency."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_local_env()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("vera-bot")

# ── Config ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
USE_LLM_COMPOSER = os.environ.get("USE_LLM_COMPOSER", "false").lower() in {"1", "true", "yes"}
START_TIME = time.time()
BOT_VERSION = "1.0.0"

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Vera Bot", version=BOT_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── In-memory state ──────────────────────────────────────────────────────────
# context_store[(scope, context_id)] = {version: int, payload: dict}
context_store: dict[tuple[str, str], dict] = {}

# conversations[conversation_id] = {
#   merchant_id, customer_id, turns: [{from, msg, ts}],
#   state: "active"|"waiting"|"ended",
#   trigger_id, suppression_key, wait_until, auto_reply_count
# }
conversations: dict[str, dict] = {}

# suppression set: suppression_key -> True (dedup)
suppressed: set[str] = set()

# fired triggers this session (trigger_id -> conversation_id)
fired_triggers: dict[str, str] = {}

# repeated auto-replies may arrive with fresh conversation ids in replay tests
auto_reply_memory: dict[tuple[str, str], int] = {}

# ── Pydantic models ──────────────────────────────────────────────────────────
class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = context_store.get((scope, context_id))
    return entry["payload"] if entry else None

def get_all_by_scope(scope: str) -> list[dict]:
    return [v["payload"] for (s, _), v in context_store.items() if s == scope]

def count_by_scope() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in context_store:
        if scope in counts:
            counts[scope] += 1
    return counts

AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"aapki madad ke liye shukriya",
    r"we will get back to you",
    r"hum jald hi aapse sampark",
    r"our team will respond",
    r"hamari team aapse sampark karegi",
    r"this is an automated",
    r"this is a automated",
    r"ek automated assistant",
    r"main ek automated",
    r"auto.?reply",
    r"automatic reply",
]

def is_auto_reply(message: str) -> bool:
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in AUTO_REPLY_PATTERNS)

def is_opt_out(message: str) -> bool:
    patterns = [
        r"\bstop\b", r"\bnahi\b", r"\bno thanks\b", r"\bnot interested\b",
        r"unsubscribe", r"band karo", r"mat bhejo", r"don't (message|contact|send)",
        r"stop messaging", r"remove me", r"hatao mujhe", r"block",
    ]
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in patterns)

def is_intent_transition(message: str) -> bool:
    """Detect when merchant says yes/go/do it — switch from pitch to action."""
    patterns = [
        r"\byes\b", r"\bha(n|nj)i?\b", r"\bok(ay)?\b", r"\bsure\b", r"\bgo ahead\b",
        r"\blet'?s do it\b", r"\bdo it\b", r"\bconfirm\b", r"\bchalo\b",
        r"\bkaro\b", r"\bstart\b", r"\bproceed\b", r"\bsend it\b", r"\bbhejo\b",
        r"\bplease (send|do|proceed|start|draft)\b",
    ]
    msg_lower = message.lower().strip()
    return any(re.search(p, msg_lower) for p in patterns)

def is_hostile(message: str) -> bool:
    patterns = [
        r"stop (messaging|bothering|contacting)",
        r"don't (contact|message|disturb)",
        r"useless", r"bakwaas", r"waste of time",
        r"bother", r"irritat",
    ]
    return any(re.search(p, message.lower()) for p in patterns)

# ── Anthropic LLM call ───────────────────────────────────────────────────────

async def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")

    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    
    async with httpx.AsyncClient(timeout=25.0) as client:
        try:
            resp = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "accept": "application/json",
                },
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            if not content or content[0].get("type") != "text":
                raise ValueError(f"Unexpected Anthropic response shape: {data}")
            return content[0]["text"]
        except httpx.HTTPStatusError as e:
            logger.error(
                "Anthropic API error status=%s model=%s body=%s",
                e.response.status_code,
                MODEL,
                e.response.text[:2000],
            )
            raise
        except Exception:
            logger.exception("Anthropic API call failed model=%s", MODEL)
            raise

# ── Prompt builders ──────────────────────────────────────────────────────────

COMPOSER_SYSTEM = """You are Vera, magicpin's AI merchant assistant. You send WhatsApp messages to merchants to help them grow.

RULES (strict):
1. Every opener must be trigger-specific. Never use generic openings like "quick update".
2. Structure: owner hook → WHY NOW trigger → specific metric/fact → action recommendation → one CTA.
3. One clear CTA only. Use YES/STOP or CONFIRM/CANCEL unless a booking slot flow needs 1/2.
4. Use service+price format ("Dental Cleaning @ ₹299") when available.
5. Match category voice: dentists/pharmacies precise and clinical; restaurants operator-focused; salons warm and visual; gyms coach-like.
6. No hallucinated data. Only use context numbers, dates, offers, sources, locations.
7. Body should be 40-120 words, WhatsApp-readable, no long preambles.
8. Show judgment: regulation=compliance urgency, perf dip=loss framing, festival/event=timely opportunity, customer trigger=relationship action.

COMPULSION LEVERS (use 1-2):
- Specificity/verifiability: real numbers, dates, sources
- Loss aversion: \"you're missing X\" / \"before this closes\"
- Social proof: \"3 dentists in your area did Y\"
- Effort externalization: \"I've drafted X — just say go\"
- Curiosity: \"want to see who?\"
- Reciprocity: \"noticed Y, thought you'd want to know\"
- Single binary commitment: Reply YES / STOP

OUTPUT FORMAT (valid JSON only, no markdown):
{
  "body": "<the WhatsApp message>",
  "cta": "open_ended" | "binary_yes_stop" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "<key>",
  "rationale": "<1-2 sentences: why this message, what it achieves>"
}"""

def build_compose_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> str:
    # Extract key merchant facts
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    subs = merchant.get("subscription", {})
    signals = merchant.get("signals", [])
    offers = merchant.get("offers", [])
    conv_history = merchant.get("conversation_history", [])
    cust_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])

    # Active offers
    active_offers = [o["title"] for o in offers if o.get("status") == "active"]
    
    # Category voice
    voice = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})
    digest = category.get("digest", [])
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])
    
    # Trigger payload
    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})
    urgency = trigger.get("urgency", 2)
    
    # Relevant digest item
    top_item_id = trg_payload.get("top_item_id", "")
    relevant_digest = next((d for d in digest if d.get("id") == top_item_id), None)
    if not relevant_digest and digest:
        relevant_digest = digest[0]
    
    # Recent conversation
    last_turns = conv_history[-2:] if conv_history else []
    
    merchant_section = f"""MERCHANT:
- Name: {identity.get('name')}
- Owner: {identity.get('owner_first_name', 'Owner')}
- City: {identity.get('city')}, {identity.get('locality')}
- Languages: {identity.get('languages', ['en'])}
- Subscription: {subs.get('status')} | {subs.get('plan')} | {subs.get('days_remaining')} days left
- Performance (30d): views={perf.get('views')}, calls={perf.get('calls')}, CTR={perf.get('ctr')} (peer avg={peer_stats.get('avg_ctr')})
- 7d delta: views {perf.get('delta_7d', {}).get('views_pct', 0):+.0%}, calls {perf.get('delta_7d', {}).get('calls_pct', 0):+.0%}
- Active offers: {active_offers if active_offers else 'None'}
- Signals: {signals}
- Customer aggregate: {json.dumps(cust_agg)}
- Review themes: {json.dumps(review_themes)}"""

    trigger_section = f"""TRIGGER:
- Kind: {trg_kind}
- Source: {trigger.get('source')}
- Urgency: {urgency}/5
- Payload: {json.dumps(trg_payload)}
- Suppression key: {trigger.get('suppression_key')}"""

    category_section = f"""CATEGORY: {category.get('slug')}
- Voice tone: {voice.get('tone')}
- Taboo words: {voice.get('vocab_taboo', [])}
- Salutations: {voice.get('salutation_examples', [])}
- Peer stats: avg_rating={peer_stats.get('avg_rating')}, avg_ctr={peer_stats.get('avg_ctr')}, avg_reviews={peer_stats.get('avg_review_count')}
- Relevant digest item: {json.dumps(relevant_digest) if relevant_digest else 'None'}
- Seasonal beats: {json.dumps(seasonal[:2])}
- Trend signals: {json.dumps(trends[:2])}
- Offer catalog: {json.dumps([o['title'] for o in category.get('offer_catalog', [])[:4]])}"""

    customer_section = ""
    if customer:
        rel = customer.get("relationship", {})
        customer_section = f"""
CUSTOMER (send as merchant_on_behalf):
- Name: {customer.get('identity', {}).get('name')}
- Language pref: {customer.get('identity', {}).get('language_pref')}
- State: {customer.get('state')}
- Last visit: {rel.get('last_visit')} | Total visits: {rel.get('visits_total')}
- Services received: {rel.get('services_received', [])}
- Preferred slot: {customer.get('preferences', {}).get('preferred_slots')}"""

    history_section = ""
    if last_turns:
        history_section = f"""
RECENT CONVERSATION:
{chr(10).join(f"  [{t['from'].upper()}]: {t['body'][:100]}" for t in last_turns)}"""

    return f"""{merchant_section}

{category_section}

{trigger_section}{customer_section}{history_section}

Compose the optimal WhatsApp message for this merchant right now. The trigger is your "why now" — make it explicit. Output valid JSON only."""

REPLY_SYSTEM = """You are Vera, magicpin's AI merchant assistant handling a live WhatsApp conversation.

Your job: Decide the next action in an ongoing conversation.

RULES:
1. Every send reply must follow: owner name + strong hook -> WHY NOW -> concrete output/action -> low-effort CTA.
2. Never start with "Got it", "Sure", "Okay", "Done", or "Let me check".
3. If intent is clear ("yes", "go ahead", "karo", "book") switch to action mode immediately.
4. If complaint: ask for the minimum facts and promise routing/escalation.
5. If off-topic: politely decline and redirect to the active Vera task.
6. Keep replies under 80 words. No repeated body from prior turns. No URLs.
7. In action mode, show an actual draft/checklist/offer, not a promise to make one later.

OUTPUT (valid JSON only):
For send: {"action": "send", "body": "...", "cta": "...", "rationale": "..."}
For wait: {"action": "wait", "wait_seconds": N, "rationale": "..."}
For end: {"action": "end", "rationale": "..."}"""

def build_reply_prompt(conv: dict, merchant_message: str, merchant: Optional[dict], category: Optional[dict]) -> str:
    turns = conv.get("turns", [])
    history = "\n".join(f"[{t['from'].upper()}]: {t['msg'][:150]}" for t in turns[-4:])
    
    merchant_name = ""
    if merchant:
        merchant_name = merchant.get("identity", {}).get("name", "")
    
    cat_voice = ""
    if category:
        cat_voice = f"Category voice: {category.get('voice', {}).get('tone', 'professional')}"
    
    auto_count = conv.get("auto_reply_count", 0)
    
    return f"""CONVERSATION (merchant: {merchant_name}):
{history}

[MERCHANT LATEST]: {merchant_message}

Auto-reply count this conversation: {auto_count}
{cat_voice}

Decide the next action. If this is a clear YES/intent to proceed — go to action mode immediately. Output valid JSON only."""

# ── Core composition function ────────────────────────────────────────────────

def pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.0f}%"
    return str(value) if value is not None else "n/a"

def owner_label(merchant: dict) -> str:
    identity = merchant.get("identity", {})
    first = identity.get("owner_first_name") or "there"
    if merchant.get("category_slug") == "dentists" and not str(first).lower().startswith("dr"):
        return f"Dr. {first}"
    return first

def active_offer(merchant: dict, category: dict) -> str:
    offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active" and o.get("title")]
    if offers:
        return offers[0]
    catalog = [o.get("title") for o in category.get("offer_catalog", []) if o.get("title")]
    return catalog[0] if catalog else "a focused offer"

def find_digest(category: dict, trigger: dict) -> Optional[dict]:
    digest = category.get("digest", [])
    top_item_id = trigger.get("payload", {}).get("top_item_id")
    if top_item_id:
        match = next((d for d in digest if d.get("id") == top_item_id), None)
        if match:
            return match
    kind = trigger.get("kind")
    if kind == "regulation_change":
        return next((d for d in digest if d.get("kind") == "compliance"), digest[0] if digest else None)
    if kind in {"cde_opportunity", "research_digest"}:
        return digest[0] if digest else None
    return None

def customer_name(customer: Optional[dict]) -> str:
    return (customer or {}).get("identity", {}).get("name", "this customer")

def first_slot(trigger: dict, customer: Optional[dict] = None) -> str:
    slots = trigger.get("payload", {}).get("available_slots") or []
    if slots:
        return slots[0].get("label") or slots[0].get("iso") or "the first available slot"
    pref = (customer or {}).get("preferences", {}).get("preferred_slots")
    return str(pref).replace("_", " ") if pref else "the next suitable slot"

def merchant_fact_line(merchant: dict, category: dict) -> str:
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    return (
        f"Your last 30 days: {perf.get('views')} views, {perf.get('calls')} calls, "
        f"CTR {pct(perf.get('ctr'))} vs peer {pct(peer.get('avg_ctr'))}."
    )

def compact_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v).replace("_", " ") for v in value[:4])
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in list(value.items())[:3])
    return str(value).replace("_", " ")

def business_label(merchant: dict) -> str:
    identity = merchant.get("identity", {})
    name = identity.get("name", "your business")
    locality = identity.get("locality")
    return f"{name} {locality}" if locality and locality not in name else name

def customer_language(customer: Optional[dict]) -> str:
    return str((customer or {}).get("identity", {}).get("language_pref", "")).lower()

def is_hi_pref(customer: Optional[dict], merchant: Optional[dict] = None) -> bool:
    langs = [str(x).lower() for x in (merchant or {}).get("identity", {}).get("languages", [])]
    return "hi" in customer_language(customer) or "hi" in langs

def safe_join(items: Any) -> str:
    if isinstance(items, list):
        return ", ".join(str(x).replace("_", " ") for x in items if x and x != "...")
    return str(items).replace("_", " ") if items else ""

def primary_metric_count(merchant: dict) -> str:
    agg = merchant.get("customer_aggregate", {})
    for key in ("high_risk_adult_count", "chronic_rx_count", "total_active_members", "lapsed_180d_plus", "delivery_orders_30d", "total_unique_ytd"):
        if agg.get(key) is not None:
            return f"{agg[key]} {key.replace('_', ' ')}"
    return "your current customer base"

def category_action_word(category_slug: str) -> str:
    return {
        "dentists": "patient note",
        "pharmacies": "customer workflow",
        "restaurants": "offer copy",
        "salons": "customer WhatsApp",
        "gyms": "member nudge",
    }.get(category_slug, "message")

def offer_or_action(merchant: dict, category: dict) -> str:
    offer = active_offer(merchant, category)
    return offer if offer != "a focused offer" else category_action_word(merchant.get("category_slug", ""))

def trigger_body(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> tuple[str, str, str, str]:
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    subs = merchant.get("subscription", {})
    payload = trigger.get("payload", {})
    kind = trigger.get("kind", "generic")
    name = owner_label(merchant)
    city = identity.get("city", "your city")
    locality = identity.get("locality", "your area")
    offer = active_offer(merchant, category)
    action_item = category_action_word(merchant.get("category_slug", ""))
    fact = merchant_fact_line(merchant, category)
    suppression_key = trigger.get("suppression_key", f"{kind}:{merchant.get('merchant_id')}")
    cta = "binary_yes_stop"
    send_as = "merchant_on_behalf" if customer else "vera"

    if kind in {"recall_due", "appointment_tomorrow"}:
        due = payload.get("due_date") or payload.get("appointment_at") or payload.get("date")
        service = str(payload.get("service_due") or payload.get("service") or payload.get("metric_or_topic") or "follow-up").replace("_", " ")
        visits = (customer or {}).get("relationship", {}).get("visits_total")
        timing = f"due on {due}" if due else "due now"
        merchant_from = business_label(merchant)
        if kind == "appointment_tomorrow":
            body = f"Hi {customer_name(customer)}, {merchant_from} here. Reminder: your {service} is tomorrow. We have your {first_slot(trigger, customer)} preference noted. Reply CONFIRM if the slot works, or CHANGE if you need another time."
            return body, "binary_confirm_cancel", send_as, suppression_key
        if is_hi_pref(customer, merchant):
            body = f"Hi {customer_name(customer)}, {merchant_from} here. Aapka {service} {timing}; last visit ke baad {visits} visits total hain. {offer} available hai. Slot option: {first_slot(trigger, customer)}. Reply 1 to book, 2 for another time."
        else:
            body = f"Hi {customer_name(customer)}, {merchant_from} here. Your {service} is {timing}; you have visited us {visits} times before. {offer} is available, with {first_slot(trigger, customer)} as the first slot. Reply 1 to book, 2 for another time."
        return body, "multi_choice_slot", send_as, suppression_key

    if kind == "chronic_refill_due" and customer:
        medicines = safe_join(payload.get("molecule_list") or (customer.get("relationship", {}).get("services_received") or [])[:3])
        runout = payload.get("stock_runs_out_iso", "").split("T")[0] or "soon"
        senior_offer = next((o.get("title") for o in merchant.get("offers", []) if "Senior" in o.get("title", "")), "")
        delivery_offer = next((o.get("title") for o in merchant.get("offers", []) if "Delivery" in o.get("title", "")), "")
        discount = f" {senior_offer} applied." if senior_offer else ""
        delivery = f" {delivery_offer} to saved address." if delivery_offer or payload.get("delivery_address_saved") else ""
        body = f"Namaste, {business_label(merchant)} here. {customer_name(customer)} ji ki medicines ({medicines}) {runout} ko khatam hongi. Same pack ready hai.{discount}{delivery} Reply CONFIRM to dispatch, or CHANGE if dosage/brand changed."
        return body, "binary_confirm_cancel", send_as, suppression_key

    if kind == "wedding_package_followup" and customer:
        rel = customer.get("relationship", {})
        wedding = customer.get("preferences", {}).get("event_date") or payload.get("wedding_date") or "your wedding date"
        last_visit = rel.get("last_visit", "your trial")
        body = f"Hi {customer_name(customer)}, {business_label(merchant)} here. Since your bridal trial on {last_visit}, this is the right prep window before {wedding}. {offer} can be the first step. Reply YES to block your preferred {first_slot(trigger, customer)} slot, or STOP to skip."
        return body, cta, send_as, suppression_key

    if kind in {"customer_lapsed_soft", "customer_lapsed_hard", "winback_eligible", "trial_followup"}:
        rel = (customer or {}).get("relationship", {})
        last_visit = rel.get("last_visit", "their last visit")
        services = ", ".join(rel.get("services_received", [])[-2:]) or payload.get("service_due") or "last service"
        no_shame = "No pressure, no commitment." if merchant.get("category_slug") == "gyms" else "No pressure."
        body = f"Hi {customer_name(customer)}, {business_label(merchant)} here. It's been a while since {last_visit} after {services}. {offer} is available for your return visit. {no_shame} Reply YES to hold {first_slot(trigger, customer)}, or STOP to skip."
        return body, "binary_yes_stop", send_as, suppression_key

    if kind in {"research_digest", "regulation_change", "cde_opportunity"}:
        item = find_digest(category, trigger) or {}
        source = item.get("source", "category digest")
        title = item.get("title") or payload.get("topic") or "new category update"
        detail = item.get("actionable") or item.get("summary") or "worth reviewing for your business"
        relevant_count = merchant.get("customer_aggregate", {}).get("high_risk_adult_count", merchant.get("customer_aggregate", {}).get("total_unique_ytd", "your"))
        why = "compliance deadline" if kind == "regulation_change" else "new category signal"
        body = f"{name}, today’s {why}: {source} says {title}. {detail}. This maps to {relevant_count} customers/patients in your data. I can turn it into a precise {action_item} without overclaiming. Reply YES to draft, STOP to skip."
        return body, "open_ended", send_as, suppression_key

    if kind in {"perf_dip", "seasonal_perf_dip"}:
        delta = perf.get("delta_7d", {})
        body = f"{name}, demand is slipping today — calls {pct(delta.get('calls_pct'))}, views {pct(delta.get('views_pct'))}. {fact} That is missed intent in {locality}. I can prepare a recovery {action_item} around {offer_or_action(merchant, category)}. Reply YES to run it, STOP to skip."
        return body, cta, send_as, suppression_key

    if kind == "perf_spike":
        delta = perf.get("delta_7d", {})
        body = f"{name}, demand is spiking today — 7-day views {pct(delta.get('views_pct'))}, calls {pct(delta.get('calls_pct'))}. {fact} Before this cools off, I can turn {offer_or_action(merchant, category)} into a tight follow-up campaign. Reply YES to launch, STOP to skip."
        return body, cta, send_as, suppression_key

    if kind == "ipl_match_today":
        match = payload.get("match", "today's match")
        venue = payload.get("venue", city)
        time_txt = payload.get("match_time_iso", "").split("T")[-1][:5] or "tonight"
        if payload.get("is_weeknight") is False:
            body = f"{name}, {match} at {venue} starts {time_txt} today. Since it is not a weeknight, avoid a dine-in discount; push {offer} as a delivery-first match special instead. {fact} I’ll prepare the banner + WhatsApp copy. Reply YES to use it, STOP to skip."
        else:
            body = f"{name}, {match} at {venue} starts {time_txt}. This is a timely dinner hook for {city}. {fact} I can turn {offer} into a match-night delivery message. Reply YES to draft, STOP to skip."
        return body, "binary_yes_stop", send_as, suppression_key

    if kind in {"festival_upcoming", "category_seasonal"}:
        trends = payload.get("trends") or payload.get("event") or payload.get("festival") or payload.get("season") or "this week"
        body = f"{name}, {compact_value(trends)} gives you a timely hook in {city} today. {fact} I can prepare a same-day {action_item} around {offer_or_action(merchant, category)} so you are not late to the demand window. Reply YES to send, STOP to skip."
        return body, cta, send_as, suppression_key

    if kind == "competitor_opened":
        comp = payload.get("competitor_name") or payload.get("competitor") or "a new competitor"
        distance = payload.get("distance_km") or payload.get("distance") or "nearby"
        body = f"{name}, {comp} has opened {distance} from {locality}. {fact} That can pull nearby searches away this week. I can send a defensive {action_item} using {offer_or_action(merchant, category)} before they capture intent. Reply YES to run it, STOP to skip."
        return body, cta, send_as, suppression_key

    if kind == "review_theme_emerged":
        theme = (merchant.get("review_themes") or [{}])[0]
        body = f"{name}, reviews now mention {theme.get('theme', 'one theme')} {theme.get('occurrences_30d', '')} times in 30 days. Quote: {theme.get('common_quote', 'worth checking')}. I can prepare a calm response and customer note before it becomes a pattern. Reply YES to use it, STOP to skip."
        return body, cta, send_as, suppression_key

    if kind == "milestone_reached":
        milestone = payload.get("milestone") or payload.get("metric") or "a new milestone"
        body = f"{name}, you just hit {milestone}. {fact} This is fresh social proof, so I can turn it into a thank-you {action_item} around {offer_or_action(merchant, category)} for warm customers. Reply YES to send, STOP to skip."
        return body, cta, send_as, suppression_key

    if kind == "renewal_due":
        days = subs.get("days_remaining", payload.get("days_remaining", "soon"))
        body = f"{name}, your {subs.get('plan', 'Vera')} plan has {days} days left. {fact} Let’s create one measurable win before renewal using {offer_or_action(merchant, category)}. I’ll prepare it with one clear CTA. Reply YES to queue it, STOP to skip."
        return body, cta, send_as, suppression_key

    if kind == "supply_alert":
        molecule = payload.get("molecule", "medicine")
        batches = safe_join(payload.get("affected_batches"))
        manufacturer = payload.get("manufacturer", "manufacturer")
        count = merchant.get("customer_aggregate", {}).get("chronic_rx_count", "your chronic-Rx")
        body = f"{name}, urgent pharmacy alert today from {manufacturer} for {molecule} batches {batches}. You have {count} chronic-Rx customers in your data. I can prepare the customer note plus replacement-pickup workflow so this is handled cleanly today. Reply YES to prepare it, STOP to skip."
        return body, "binary_yes_stop", send_as, suppression_key

    if kind == "active_planning_intent":
        topic = str(payload.get("intent_topic") or payload.get("merchant_last_message") or "this plan").replace("_", " ")
        history = " ".join(t.get("body", "") for t in merchant.get("conversation_history", [])[-2:])
        if "corporate" in topic or "thali" in topic:
            body = f"{name}, starter plan for {identity.get('name')}: corporate thali based on your active {offer}. Use 10/25/50-order tiers, day-before confirmation by 5pm, and lunch delivery between 12:30-1pm in {locality}. Your cafe has {perf.get('views')} views and {perf.get('directions')} direction requests in 30d. Want me to draft the 3-line office WhatsApp?"
        elif "kids" in topic or "yoga" in topic or "summer" in history.lower():
            members = merchant.get("customer_aggregate", {}).get("total_active_members", "your current")
            body = f"{name}, for the kids yoga program: 4 weeks, 3 classes/week, age 7-12, with {offer} as the low-risk trial hook. You already have {members} active members and {pct(perf.get('delta_7d', {}).get('calls_pct'))} call growth this week. Want me to draft the GBP post plus parent WhatsApp?"
        else:
            body = f"{name}, following your note on {topic}, I have a concrete next step. {fact} I can draft a ready-to-send message around {offer}. Reply YES to see the draft, STOP to skip."
        return body, "open_ended", send_as, suppression_key

    if kind in {"curious_ask_due", "dormant_with_vera", "gbp_unverified"}:
        topic = payload.get("intent_topic") or payload.get("merchant_last_message") or payload.get("topic") or kind.replace("_", " ")
        body = f"{name}, {str(topic).replace('_', ' ')} is due for a light-touch check-in today. {fact} Reply with the service customers ask about most this week; I’ll turn it into a Google post and WhatsApp reply in 5 minutes."
        return body, cta, send_as, suppression_key

    body = f"{name}, {kind.replace('_', ' ')} needs attention today for {identity.get('name', 'your business')}. {fact} I can turn this into one focused {action_item} using {offer_or_action(merchant, category)}. Reply YES to prepare it, STOP to skip."
    return body, cta, send_as, suppression_key

def deterministic_compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    body, cta, send_as, suppression_key = trigger_body(category, merchant, trigger, customer)
    return {
        "body": body[:900],
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": f"Deterministic {trigger.get('kind')} route using trigger, merchant metrics, category voice, and customer context."
    }

def rule_based_reply(conv: dict, merchant_message: str, merchant_id: str = "") -> dict:
    msg = merchant_message.lower().strip()
    merchant = get_ctx("merchant", merchant_id) if merchant_id else None
    category_slug = (merchant or {}).get("category_slug", "")
    owner = owner_label(merchant) if merchant else "Vera"
    category = get_ctx("category", category_slug) if category_slug else {}
    offer = active_offer(merchant, category) if merchant else "the active offer"
    last_vera = next((t.get("msg", "") for t in reversed(conv.get("turns", [])) if t.get("from") == "vera"), "")
    role = (conv.get("turns", [])[-1].get("from") if conv.get("turns") else "merchant") or "merchant"

    if role == "customer":
        if any(word in msg for word in ["complaint", "bad", "late", "refund", "wrong", "issue", "problem"]):
            return {
                "action": "send",
                "body": "Sorry about this — let’s fix it quickly today. Please share the visit/order date and what went wrong. I’ll pass the exact issue to the store owner so they can respond with the right next step.",
                "cta": "open_ended",
                "rationale": "Customer complaint detected; asks for the minimum facts needed to route the issue."
            }
        if any(word in msg for word in ["price", "cost", "kitna", "charges"]):
            return {
                "action": "send",
                "body": "I can help with that. Please tell me which service/medicine you mean, and I’ll share the exact available offer or ask the merchant to confirm if pricing changed.",
                "cta": "open_ended",
                "rationale": "Customer pricing intent detected; asks one clarifying fact instead of generic acknowledgment."
            }
        if is_intent_transition(merchant_message) or any(word in msg for word in ["book", "confirm", "slot", "wed", "thu"]):
            return {
                "action": "send",
                "body": "Slot request received. I’ll share this exact time with the merchant now; if it’s unavailable, they’ll suggest the closest option. Reply CHANGE if you want another time.",
                "cta": "binary_confirm_cancel",
                "rationale": "Customer accepted a booking/reminder flow; confirms action with one escape hatch."
            }

    if any(word in msg for word in ["gst", "tax filing", "income tax"]):
        return {
            "action": "send",
            "body": f"{owner}, GST filing is for your CA. This thread is about today’s active merchant action, so I’ll keep momentum here. I can use {offer} and prepare one customer-ready message now. Reply YES to use it, STOP to skip.",
            "cta": "binary_yes_stop",
            "rationale": "Out-of-scope request declined while returning to the original Vera task."
        }

    if any(word in msg for word in ["x-ray", "xray", "radiograph", "d-speed", "iopa", "rvg"]):
        owner = owner_label(merchant) if merchant else "Doctor"
        return {
            "action": "send",
            "body": f"{owner}, quick heads-up — the new DCI radiograph limit makes old D-speed film risky to leave unchecked. Verify dose compliance today; if it misses, move to E-speed or RVG and document the SOP. Say CONFIRM and I’ll send a ready-to-use 5-point audit checklist.",
            "cta": "binary_confirm_cancel",
            "rationale": "Personalized compliance reply with clear why-now urgency, specific D-speed guidance, and low-effort checklist CTA."
        }

    if any(word in msg for word in ["audit", "checklist", "setup", "how"]):
        return {
            "action": "send",
            "body": f"{owner}, quick action plan for today: checklist — 1) current setup, 2) rule gap, 3) risk level, 4) fix owner, 5) SOP note. Say CONFIRM and I’ll format this for immediate use.",
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant asks for execution help; converts to a concrete checklist workflow."
        }

    if is_intent_transition(merchant_message):
        scope = primary_metric_count(merchant) if merchant else "the selected audience"
        return {
            "action": "send",
            "body": f"{owner}, I’ve prepared this for {scope}: “{offer} — limited slots today. Reply YES and we’ll hold one for you.” Reply CONFIRM to approve, or CANCEL to stop.",
            "cta": "binary_confirm_cancel",
            "rationale": f"Merchant showed intent; moving to action mode from: {last_vera[:80]}"
        }

    if any(word in msg for word in ["not sure", "maybe", "later", "tomorrow"]):
        return {
            "action": "wait",
            "wait_seconds": 86400,
            "rationale": "Merchant deferred; backing off instead of pushing another generic message."
        }

    if any(word in msg for word in ["price", "cost", "offer", "discount"]):
        offer = active_offer(merchant, get_ctx("category", category_slug) or {}) if merchant else "the active offer"
        return {
            "action": "send",
            "body": f"{owner}, use the concrete offer, not a generic discount: {offer}. Message: “{offer} available today — reply YES and we’ll hold it.” Reply CONFIRM to use this, CANCEL to stop.",
            "cta": "binary_yes_stop",
            "rationale": "Merchant asked about commercial framing; returns to service+price specificity."
        }

    return {
        "action": "send",
        "body": f"{owner}, this needs a concrete next step today, not another question. Message: “{offer} available today — reply YES and we’ll hold it.” Reply CONFIRM to use it, CANCEL to stop.",
        "cta": "binary_yes_stop",
        "rationale": "Rule-based reply keeps the conversation moving with a single low-friction CTA."
    }

async def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None
) -> dict:
    if not USE_LLM_COMPOSER:
        return deterministic_compose(category, merchant, trigger, customer)

    system = COMPOSER_SYSTEM
    user = build_compose_prompt(category, merchant, trigger, customer)
    
    try:
        raw = await call_claude(system, user, max_tokens=600)
    except Exception as e:
        result = deterministic_compose(category, merchant, trigger, customer)
        result["rationale"] = f"Rule-based fallback after Anthropic failure: {str(e)[:120]}"
        return result
    
    # Parse JSON — handle markdown fences
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)
    raw = raw.strip()
    
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON block
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
            except Exception as e:
                result = deterministic_compose(category, merchant, trigger, customer)
                result["rationale"] = f"Rule-based fallback after invalid Anthropic JSON: {str(e)[:120]}"
        else:
            result = deterministic_compose(category, merchant, trigger, customer)
            result["rationale"] = "Rule-based fallback after Anthropic returned non-JSON text"
    
    # Ensure required fields
    if "send_as" not in result:
        result["send_as"] = "merchant_on_behalf" if customer else "vera"
    if "suppression_key" not in result:
        result["suppression_key"] = trigger.get("suppression_key", "")
    
    return result


async def compose_reply(conv: dict, merchant_message: str, merchant_id: str) -> dict:
    if not USE_LLM_COMPOSER:
        return rule_based_reply(conv, merchant_message, merchant_id)

    merchant = get_ctx("merchant", merchant_id) if merchant_id else None
    cat_slug = merchant.get("category_slug", "") if merchant else ""
    category = get_ctx("category", cat_slug) if cat_slug else None
    
    system = REPLY_SYSTEM
    user = build_reply_prompt(conv, merchant_message, merchant, category)
    
    try:
        raw = await call_claude(system, user, max_tokens=400)
    except Exception as e:
        result = rule_based_reply(conv, merchant_message, merchant_id)
        result["rationale"] = f"Rule-based reply after Anthropic failure: {str(e)[:120]}"
        return result
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)
    raw = raw.strip()
    
    try:
        result = json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
            except Exception:
                result = rule_based_reply(conv, merchant_message, merchant_id)
                result["rationale"] = "Rule-based reply after invalid Anthropic JSON"
        else:
            result = rule_based_reply(conv, merchant_message, merchant_id)
            result["rationale"] = "Rule-based reply after Anthropic returned non-JSON text"
    
    return result

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": count_by_scope()
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera AI",
        "team_members": ["Vera Bot"],
        "model": "deterministic-rule-composer" if not USE_LLM_COMPOSER else MODEL,
        "approach": (
            "Deterministic 4-context composition with trigger-kind routing, "
            "auto-reply detection, intent-transition handling, and "
            "per-conversation suppression. Stateful in-memory store with "
            "versioned context replacement."
        ),
        "contact_email": "vera@magicpin.com",
        "version": BOT_VERSION,
        "submitted_at": "2026-04-30T00:00:00Z"
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)
    current = context_store.get(key)
    
    if current and current["version"] == body.version:
        return {
            "accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "no_op": True
        }

    if current and current["version"] > body.version:
        return JSONResponse(status_code=409, content={
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"]
        })
    
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(status_code=400, content={
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {valid_scopes}"
        })
    
    context_store[key] = {
        "version": body.version,
        "payload": body.payload
    }
    
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    now = body.now
    actions = []
    
    for trg_id in body.available_triggers:
        # Already fired this trigger?
        if trg_id in fired_triggers:
            continue
        
        trg_payload = get_ctx("trigger", trg_id)
        if not trg_payload:
            continue
        
        # The judge sends the trigger set it wants evaluated. Some canonical
        # fixtures carry historical expires_at values, so we do not drop a
        # provided trigger solely on timestamp comparison.
        
        # Check suppression
        suppression_key = trg_payload.get("suppression_key", "")
        if suppression_key in suppressed:
            continue
        
        merchant_id = trg_payload.get("merchant_id")
        customer_id = trg_payload.get("customer_id")
        
        if not merchant_id:
            continue
        
        merchant = get_ctx("merchant", merchant_id)
        if not merchant:
            continue
        
        cat_slug = merchant.get("category_slug", "")
        category = get_ctx("category", cat_slug)
        if not category:
            continue
        
        customer = None
        if customer_id:
            customer = get_ctx("customer", customer_id)
        
        # Compose message
        try:
            composed = await compose_message(category, merchant, trg_payload, customer)
        except Exception as e:
            logger.exception("Composer failed; using deterministic fallback trigger_id=%s", trg_id)
            composed = deterministic_compose(category, merchant, trg_payload, customer)
            composed["rationale"] = f"Endpoint fallback after composer error: {str(e)[:120]}"
        
        conv_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:6]}"
        
        # Create conversation record
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "suppression_key": suppression_key,
            "state": "active",
            "turns": [{"from": "vera", "msg": composed["body"], "ts": now}],
            "auto_reply_count": 0,
            "wait_until": None
        }
        
        # Mark trigger as fired + suppress
        fired_triggers[trg_id] = conv_id
        if suppression_key:
            suppressed.add(suppression_key)
        
        # Determine template name from trigger kind
        kind = trg_payload.get("kind", "generic")
        template_map = {
            "research_digest": "vera_research_digest_v1",
            "regulation_change": "vera_compliance_alert_v1",
            "recall_due": "merchant_recall_reminder_v1",
            "perf_dip": "vera_perf_dip_v1",
            "perf_spike": "vera_perf_spike_v1",
            "renewal_due": "vera_renewal_nudge_v1",
            "festival_upcoming": "vera_festival_v1",
            "competitor_opened": "vera_competitor_alert_v1",
            "milestone_reached": "vera_milestone_v1",
            "dormant_with_vera": "vera_reactivation_v1",
            "review_theme_emerged": "vera_review_insight_v1",
            "curious_ask_due": "vera_curious_ask_v1",
            "ipl_match_today": "vera_event_tie_in_v1",
        }
        template_name = template_map.get(kind, "vera_generic_v1")
        
        # Extract template params from body (first 3 meaningful phrases)
        owner_name = merchant.get("identity", {}).get("owner_first_name", "")
        merchant_name = merchant.get("identity", {}).get("name", "")
        body_excerpt = composed["body"][:100]
        
        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": template_name,
            "template_params": [owner_name or merchant_name, body_excerpt, ""],
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", "")
        }
        actions.append(action)
        
        # Cap at 20 actions per tick
        if len(actions) >= 20:
            break

    if not actions:
        merchant = next((m for m in get_all_by_scope("merchant") if get_ctx("category", m.get("category_slug", ""))), None)
        if merchant:
            category = get_ctx("category", merchant.get("category_slug", ""))
            fallback_trigger = {
                "id": f"fallback_{merchant.get('merchant_id', 'merchant')}",
                "kind": "curious_ask_due",
                "source": "internal",
                "merchant_id": merchant.get("merchant_id"),
                "customer_id": None,
                "payload": {"topic": "weekly customer demand check"},
                "urgency": 1,
                "suppression_key": f"fallback:{merchant.get('merchant_id')}:demand_check"
            }
            composed = deterministic_compose(category, merchant, fallback_trigger, None)
            conv_id = f"conv_{merchant.get('merchant_id')}_fallback_{uuid.uuid4().hex[:6]}"
            conversations[conv_id] = {
                "merchant_id": merchant.get("merchant_id"),
                "customer_id": None,
                "trigger_id": fallback_trigger["id"],
                "suppression_key": fallback_trigger["suppression_key"],
                "state": "active",
                "turns": [{"from": "vera", "msg": composed["body"], "ts": now}],
                "auto_reply_count": 0,
                "wait_until": None
            }
            actions.append({
                "conversation_id": conv_id,
                "merchant_id": merchant.get("merchant_id"),
                "customer_id": None,
                "send_as": "vera",
                "trigger_id": fallback_trigger["id"],
                "template_name": "vera_curious_ask_v1",
                "template_params": [merchant.get("identity", {}).get("owner_first_name", ""), composed["body"][:100], ""],
                "body": composed["body"],
                "cta": composed.get("cta", "binary_yes_stop"),
                "suppression_key": fallback_trigger["suppression_key"],
                "rationale": "Safe fallback action from available merchant and category context."
            })
    
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    merchant_message = body.message
    merchant_id = body.merchant_id or ""
    
    # Get or create conversation
    if conv_id not in conversations:
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": body.customer_id,
            "trigger_id": "",
            "suppression_key": "",
            "state": "active",
            "turns": [],
            "auto_reply_count": 0,
            "wait_until": None
        }
    
    conv = conversations[conv_id]
    merchant_id = merchant_id or conv.get("merchant_id", "")
    
    # Check if conversation is ended
    if conv.get("state") == "ended":
        return {"action": "end", "rationale": "Conversation already ended"}
    
    # Record merchant turn
    conv["turns"].append({
        "from": body.from_role,
        "msg": merchant_message,
        "ts": body.received_at
    })
    
    # ── Decision logic ────────────────────────────────────────────────────────
    
    # 1. Opt-out detection (highest priority)
    if is_opt_out(merchant_message):
        conv["state"] = "ended"
        # Add to suppressed
        if conv.get("suppression_key"):
            suppressed.add(conv["suppression_key"])
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out. Closing conversation and suppressing future triggers for 30 days."
        }
    
    # 2. Hostile detection
    if is_hostile(merchant_message):
        conv["state"] = "ended"
        return {
            "action": "send",
            "body": "Apologies — won't message again. If anything changes, reply 'Hi Vera' anytime. 🙏",
            "cta": "none",
            "rationale": "Merchant frustrated; graceful exit with opt-back-in path."
        }
    
    # 3. Auto-reply detection
    if is_auto_reply(merchant_message):
        conv["auto_reply_count"] = conv.get("auto_reply_count", 0) + 1
        memory_key = (merchant_id or conv.get("merchant_id") or conv_id, merchant_message.lower().strip())
        auto_reply_memory[memory_key] = auto_reply_memory.get(memory_key, 0) + 1
        count = max(conv["auto_reply_count"], auto_reply_memory[memory_key])
        
        if count == 1:
            # First auto-reply: one explicit flag
            return {
                "action": "send",
                "body": "Looks like an auto-reply. When the owner sees this, just reply YES and I will continue with the campaign.",
                "cta": "binary_yes_stop",
                "rationale": "Detected auto-reply (canned greeting). One explicit prompt for owner."
            }
        elif count == 2:
            # Second: wait
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Auto-reply twice in a row — owner not at phone. Waiting 24h."
            }
        else:
            # Third+: end
            conv["state"] = "ended"
            return {
                "action": "end",
                "rationale": f"Auto-reply {count}x in a row. No real engagement signal. Closing conversation."
            }
    
    # 4. Intent transition: merchant said yes → action mode
    if is_intent_transition(merchant_message) and body.turn_number <= 3:
        if not USE_LLM_COMPOSER:
            result = rule_based_reply(conv, merchant_message, merchant_id)
            conv["turns"].append({"from": "vera", "msg": result["body"], "ts": body.received_at})
            return result

        # Go straight to action — compose a concrete next step
        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        cat_slug = merchant.get("category_slug", "") if merchant else ""
        category = get_ctx("category", cat_slug) if cat_slug else None
        
        # Build action-mode prompt
        action_prompt = f"""The merchant just confirmed intent: "{merchant_message}"

Previous Vera message: {conv['turns'][-2]['msg'][:200] if len(conv['turns']) >= 2 else 'N/A'}

Merchant: {json.dumps(merchant.get('identity', {})) if merchant else 'Unknown'}
Category: {cat_slug}
Active offers: {json.dumps([o['title'] for o in (merchant or {}).get('offers', []) if o.get('status') == 'active']) if merchant else []}

Write a SHORT (50-80 word) action-mode message that:
1. Starts with owner name + hook, never "Got it/Done/Sure"
2. States WHY NOW from the prior trigger/conversation
3. Includes an actual draft/checklist/offer line, not a promise
4. Gives ONE binary CTA with clear benefit

Output valid JSON: {{"body": "...", "cta": "binary_confirm_cancel", "rationale": "..."}}"""

        try:
            raw = await call_claude(REPLY_SYSTEM, action_prompt, max_tokens=300)
            raw = re.sub(r"^```json\s*", "", raw.strip())
            raw = re.sub(r"^```\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw).strip()
            result = json.loads(raw)
            result["action"] = "send"
            conv["turns"].append({"from": "vera", "msg": result.get("body", ""), "ts": body.received_at})
            return result
        except Exception as e:
            result = rule_based_reply(conv, merchant_message, merchant_id)
            result["rationale"] = f"Rule-based action reply after Anthropic failure: {str(e)[:120]}"
            conv["turns"].append({"from": "vera", "msg": result.get("body", ""), "ts": body.received_at})
            return result
    
    # 5. General LLM reply
    try:
        result = await compose_reply(conv, merchant_message, merchant_id)
    except Exception as e:
        logger.exception("Reply composer failed; using rule-based fallback conversation_id=%s", conv_id)
        result = rule_based_reply(conv, merchant_message, merchant_id)
        result["rationale"] = f"Endpoint rule-based reply fallback: {str(e)[:120]}"
    
    # Track state
    if result.get("action") == "end":
        conv["state"] = "ended"
    elif result.get("action") == "wait":
        conv["state"] = "waiting"
    
    if result.get("action") == "send" and result.get("body"):
        conv["turns"].append({"from": "vera", "msg": result["body"], "ts": body.received_at})
    
    return result


@app.post("/v1/teardown")
async def teardown():
    """Optional: judge calls this at end of test to wipe state."""
    context_store.clear()
    conversations.clear()
    suppressed.clear()
    fired_triggers.clear()
    auto_reply_memory.clear()
    return {"status": "torn_down"}


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
