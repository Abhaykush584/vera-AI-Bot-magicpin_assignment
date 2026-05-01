# Vera Bot — magicpin AI Challenge

## Approach

### Core architecture: 4-context LLM composition

Every message is composed from all four context layers simultaneously:

```
compose(category, merchant, trigger, customer?) → {body, cta, send_as, suppression_key, rationale}
```

The system prompt encodes all hard rules (no URLs, no fabrication, service+price format, voice matching, owner first name), all compulsion levers, and the "why-now" rule that forces every message to make the trigger kind explicit.

**Routing by trigger kind** — rather than a single generic prompt, the trigger kind informs the framing. A `research_digest` trigger gets a clinical-peer framing with source citation. A `perf_dip` trigger gets a reframe strategy (seasonal? actionable?). A `curious_ask_due` trigger gets the asking-the-merchant pattern. This is implemented as structured context blocks in the user prompt, not separate prompt files — the LLM infers the right voice.

### Conversation state machine

Every conversation tracks state through 6 tiers (evaluated in priority order per turn):

1. **Opt-out detection** (regex over 12 patterns) → end + suppress suppression_key permanently
2. **Hostile detection** → one-line graceful exit with re-opt-in path ("reply 'Hi Vera' anytime")
3. **Auto-reply detection** (regex + repeat counting) → staged: flag once → wait 24h → end
4. **Intent-transition detection** (yes/haan/karo/go ahead, max turn 4) → immediately switch to action mode; write the artifact, give binary CTA, never ask another qualifying question
5. **Off-topic redirect** (GST, taxes, unrelated) → one-line decline + thread redirect
6. **General LLM reply** with conversation history injected

### Specificity wins: no hallucination

The compose prompt explicitly labels every number as "use ONLY facts from these contexts" and includes the most relevant digest item, peer stats, and offer catalog. Numbers without provenance in the context are blocked by the system prompt instruction.

### Anti-repetition

Every conversation record stores `last_vera_body`. The compose prompt includes this as a "DO NOT REPEAT" block. The judge penalizes -2 per repeated body.

---

## Tradeoffs

| Choice | Why | Cost |
|---|---|---|
| In-memory state only | Zero setup, survives the 45-60min test window | Data lost on restart — add Redis for production |
| Single-prompt composition | Simpler, no retrieval overhead, fits all 4 contexts in ~800 tokens | Slightly higher per-call latency vs cached responses |
| temperature=0 | Deterministic for same input — judge can replay | No variation; if the first composition is wrong, retries won't help |
| Synchronous tick | Simpler code, avoids async complexity | With 20 triggers in one tick, could approach 30s limit; mitigated by returning early |
| Regex for auto-reply / intent | Fast, zero API cost, covers the common cases | Will miss novel phrasing; a classifier would be better for production |

## What additional context would help most

1. **Real auto-reply corpus** — a labelled set of real WA Business auto-reply messages to train a better detector (regex misses regional variants like Marathi/Tamil)
2. **Offer acceptance rate by trigger kind** — which compulsion lever works best for which category × trigger combination
3. **Peer comparison at locality level** — the data has city-level peer stats; locality-level (Lajpat Nagar vs Hauz Khas) would make specificity much sharper
4. **Conversation cadence history** — knowing how many times a merchant has been nudged in the last 7 days to prevent over-messaging before the judge even checks
