"""
generate_submission.py
======================
Generates submission.jsonl — 30 lines, one per canonical test pair.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/generate_submission.py

Output: submission.jsonl in the project root
"""

import os
import re
import sys
import json
import time
import httpx

# Allow running from project root or scripts/ folder
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"

EXPANDED = os.path.join(BASE, "expanded")
DATASET  = os.path.join(BASE, "dataset")

# ─── Data loaders ────────────────────────────────────────────────────────────
def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_category(slug):
    # Try expanded first, fall back to dataset
    for base in [EXPANDED, DATASET]:
        p = os.path.join(base, "categories", f"{slug}.json")
        if os.path.exists(p):
            return load(p)
    raise FileNotFoundError(f"Category not found: {slug}")

def get_merchant(mid):
    p = os.path.join(EXPANDED, "merchants", f"{mid}.json")
    if os.path.exists(p):
        return load(p)
    # Try seeds
    seeds = load(os.path.join(DATASET, "seeds", "merchants_seed.json"))
    for m in seeds.get("merchants", []):
        if m.get("merchant_id") == mid:
            return m
    raise FileNotFoundError(f"Merchant not found: {mid}")

def get_trigger(tid):
    p = os.path.join(EXPANDED, "triggers", f"{tid}.json")
    if os.path.exists(p):
        return load(p)
    seeds = load(os.path.join(DATASET, "seeds", "triggers_seed.json"))
    for t in seeds.get("triggers", []):
        if t.get("id") == tid:
            return t
    raise FileNotFoundError(f"Trigger not found: {tid}")

def get_customer(cid):
    if not cid:
        return None
    p = os.path.join(EXPANDED, "customers", f"{cid}.json")
    if os.path.exists(p):
        return load(p)
    seeds = load(os.path.join(DATASET, "seeds", "customers_seed.json"))
    for c in seeds.get("customers", []):
        if c.get("customer_id") == cid:
            return c
    return None  # ok to be missing

# ─── System prompt (same as bot.py) ──────────────────────────────────────────
SYSTEM = """You are Vera, magicpin's AI merchant assistant on WhatsApp.
You send short, sharp messages to merchants to help them grow their business.

HARD RULES:
1. ONE clear CTA per message. Binary (YES/STOP) for action triggers. Open-ended for info/digest triggers. None for pure info.
2. NO URLs in message body (instant fail).
3. NO fabricated data. Use ONLY numbers/names/facts from the contexts given.
4. NO generic % discounts. Use service+price: "Haircut @ ₹99", "Cleaning @ ₹299".
5. NO preambles. Start with the hook.
6. Body: 40-120 words.

VOICE: dentists/pharmacies = peer-clinical precision; salons = warm-practical; restaurants = operator-to-operator; gyms = coach-to-operator.
LANGUAGE: if merchant.identity.languages includes "hi", use natural Hindi-English code-mix.
OWNER NAME: Always open with owner's first name (identity.owner_first_name).
WHY-NOW: The trigger kind is the explicit reason for messaging today — make it clear.

COMPULSION LEVERS (use 1-2):
specificity/source-citation | loss-aversion | social-proof | effort-externalization | curiosity | binary-commit

OUTPUT: valid JSON only, no fences.
{"body":"...","cta":"open_ended|binary_yes_stop|binary_yes_no|binary_confirm_cancel|multi_choice_slot|none","send_as":"vera|merchant_on_behalf","suppression_key":"...","rationale":"..."}"""

def build_prompt(category, merchant, trigger, customer=None):
    identity  = merchant.get("identity", {})
    perf      = merchant.get("performance", {})
    subs      = merchant.get("subscription", {})
    offers    = merchant.get("offers", [])
    signals   = merchant.get("signals", [])
    cust_agg  = merchant.get("customer_aggregate", {})
    rev       = merchant.get("review_themes", [])

    active_offers = [o["title"] for o in offers if o.get("status") == "active"]

    voice      = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})
    digest     = category.get("digest", [])
    seasonal   = category.get("seasonal_beats", [])
    trends     = category.get("trend_signals", [])
    offer_cat  = [o.get("title") for o in category.get("offer_catalog", [])[:5]]

    trg_payload = trigger.get("payload", {})
    top_item_id = trg_payload.get("top_item_id", "")
    digest_item = next((d for d in digest if d.get("id") == top_item_id), digest[0] if digest else None)

    m = f"""MERCHANT:
merchant_id={merchant.get("merchant_id")} | name={identity.get("name")} | owner={identity.get("owner_first_name","Owner")}
locality={identity.get("locality")}, {identity.get("city")} | languages={identity.get("languages",["en"])}
sub={subs.get("plan")}/{subs.get("status")}/{subs.get("days_remaining")}d
perf 30d: views={perf.get("views")}, calls={perf.get("calls")}, ctr={perf.get("ctr")} vs peer_avg={peer_stats.get("avg_ctr")}
7d delta: views {perf.get("delta_7d",{}).get("views_pct",0):+.0%}, calls {perf.get("delta_7d",{}).get("calls_pct",0):+.0%}
active_offers={active_offers or "none"}
signals={signals}
cust_agg=total_ytd={cust_agg.get("total_unique_ytd")}, lapsed_180d={cust_agg.get("lapsed_180d_plus")}, retention={cust_agg.get("retention_6mo_pct")}, high_risk_adults={cust_agg.get("high_risk_adult_count","n/a")}
review_themes={json.dumps(rev[:2])}"""

    c = f"""CATEGORY: {category.get("slug")}
voice={voice.get("tone")} | taboos={voice.get("vocab_taboo",[])}
peer: avg_ctr={peer_stats.get("avg_ctr")}, avg_rating={peer_stats.get("avg_rating")}, avg_reviews={peer_stats.get("avg_review_count")}
offer_catalog={offer_cat}
top_digest_item={json.dumps(digest_item) if digest_item else "none"}
seasonal={json.dumps(seasonal[:2])}
trends={json.dumps(trends[:2])}"""

    t = f"""TRIGGER:
id={trigger.get("id")} | kind={trigger.get("kind")} | source={trigger.get("source")} | urgency={trigger.get("urgency")}/5
payload={json.dumps(trg_payload)}
suppression_key={trigger.get("suppression_key")}
expires={trigger.get("expires_at")}"""

    cx = ""
    if customer:
        rel  = customer.get("relationship", {})
        pref = customer.get("preferences", {})
        cx = f"""
CUSTOMER (send_as=merchant_on_behalf):
id={customer.get("customer_id")} | name={customer.get("identity",{}).get("name")} | lang={customer.get("identity",{}).get("language_pref")}
state={customer.get("state")} | last_visit={rel.get("last_visit")} | visits={rel.get("visits_total")} | services={rel.get("services_received",[])}
preferred_slot={pref.get("preferred_slots")} | consent={customer.get("consent",{}).get("scope",[])}"""

    return f"""{m}

{c}

{t}{cx}

Compose optimal WhatsApp message. The trigger kind "{trigger.get("kind")}" is why you're messaging today — say it explicitly. Output JSON only."""

def call_claude(prompt):
    if not ANTHROPIC_API_KEY:
        return '{"body":"Vera stub","cta":"open_ended","send_as":"vera","suppression_key":"stub","rationale":"no key"}'
    with httpx.Client(timeout=28.0) as client:
        r = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 700,
                "temperature": 0,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]

def parse_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise

def main():
    pairs_path = os.path.join(EXPANDED, "test_pairs.json")
    pairs = load(pairs_path)["pairs"]
    print(f"Generating {len(pairs)} submissions...\n")

    results = []
    errors  = 0

    for i, pair in enumerate(pairs):
        test_id     = pair["test_id"]
        merchant_id = pair["merchant_id"]
        trigger_id  = pair["trigger_id"]
        customer_id = pair.get("customer_id")

        print(f"  [{i+1:02d}/{len(pairs)}] {test_id} | {merchant_id} | {trigger_id}", end="", flush=True)

        try:
            merchant = get_merchant(merchant_id)
            trigger  = get_trigger(trigger_id)
            customer = get_customer(customer_id)
            cat_slug = merchant.get("category_slug", "restaurants")
            category = get_category(cat_slug)

            prompt   = build_prompt(category, merchant, trigger, customer)
            raw      = call_claude(prompt)
            composed = parse_json(raw)

            if "send_as" not in composed:
                composed["send_as"] = "merchant_on_behalf" if customer else "vera"
            if "suppression_key" not in composed:
                composed["suppression_key"] = trigger.get("suppression_key", f"pair:{test_id}")

            result = {
                "test_id":        test_id,
                "body":           composed.get("body", ""),
                "cta":            composed.get("cta", "open_ended"),
                "send_as":        composed.get("send_as", "vera"),
                "suppression_key":composed.get("suppression_key", ""),
                "rationale":      composed.get("rationale", "")
            }
            chars = len(result["body"])
            print(f" ✓ ({chars} chars)")

        except Exception as ex:
            errors += 1
            print(f" ✗ ERROR: {ex}")
            merchant_name = ""
            try:
                merchant_name = get_merchant(merchant_id).get("identity", {}).get("name", "")
            except Exception:
                pass
            result = {
                "test_id":        test_id,
                "body":           f"Hi, quick update from Vera for {merchant_name or merchant_id}.",
                "cta":            "open_ended",
                "send_as":        "vera",
                "suppression_key": f"fallback:{test_id}",
                "rationale":      f"Error fallback: {str(ex)[:80]}"
            }

        results.append(result)
        time.sleep(0.4)  # gentle rate limiting

    out_path = os.path.join(BASE, "submission.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! {len(results)} lines written → submission.jsonl")
    print(f"Errors: {errors}/{len(results)}")
    print(f"\nSample outputs:")
    for r in results[:3]:
        print(f"\n{r['test_id']}: {r['body'][:100]}...")

if __name__ == "__main__":
    main()
