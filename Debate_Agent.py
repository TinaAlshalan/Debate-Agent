"""
Debate Agent
by: Tina Alshalan
"""

import os
import json
import re
import requests  # Tavily
from datetime import datetime
from openai import OpenAI

#  CONFIG (OpenAI) 
OPENAI_API_KEY = "" 
MODEL_PRO = "gpt-4o"
MODEL_CON = "gpt-4o"
MODEL_JUDGE = "gpt-4o"

client = OpenAI(api_key=OPENAI_API_KEY)

SEPARATOR = "\n" + ("-" * 65) + "\n"

#  CONFIG + HELPERS (TAVILY)
TAVILY_API_KEY = "" 

SOCIAL_HOSTS = (
    "twitter.com","x.com","tiktok.com","reddit.com","facebook.com",
    "instagram.com","threads.net","youtube.com","youtu.be","bsky.app","bluesky.social","quora.com"
)

def split_evidence_vs_public(results):
    ev, po = [], []
    for r in results or []:
        u = (r.get("url") or "").lower()
        (po if any(host in u for host in SOCIAL_HOSTS) else ev).append(r)
    return ev, po

def fmt_fact(tag, item):
    t = (item.get("title") or "").strip() or (item.get("snippet") or "").strip()[:60]
    d = item.get("date") or ""
    return f"[{tag}] {t} — {d} — {item.get('url','')}"

def tavily_search(query, max_results=25, depth="advanced", time_range="month",
                  include_domains=None, exclude_domains=None):
    """Web search; time_range: 'day' | 'week' | 'month' | 'year'."""
    if not TAVILY_API_KEY or not (query or "").strip():
        return []

    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "query": query.strip(),
        "search_depth": depth,
        "include_answer": False,
        "include_raw_content": False,
        "max_results": int(max_results),
        "time_range": time_range,
    }
    if include_domains:
        payload["include_domains"] = list(include_domains)
    if exclude_domains:
        payload["exclude_domains"] = list(exclude_domains)

    try:
        r = requests.post("https://api.tavily.com/search", headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"WARNING: Tavily search failed: {e}")
        return []

    results = []
    for it in data.get("results", []):
        results.append({
            "title": (it.get("title") or "").strip(),
            "url": (it.get("url") or "").strip(),
            "date": it.get("published_date") or it.get("date") or "",
            "snippet": ((it.get("content") or "")[:240]).strip(),
        })
    return results


#  NOVELTY HELPERS 
def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  
    return s.strip()

def _split_sentences(s: str):
    parts = re.split(r"(?<=[.!?])\s+|\n+", s.strip())
    return [p.strip() for p in parts if p.strip()]

def _jaccard(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    a, b = set(a_tokens), set(b_tokens)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _too_similar(sent: str, memory_norm_sentences, thresh=0.82):
    s_norm = _normalize_text(sent)
    s_tokens = s_norm.split()
    for prev in memory_norm_sentences:
        sim = _jaccard(s_tokens, prev.split())
        if sim >= thresh:
            return True
    return False

def _dedupe_and_enforce_novelty(text: str, memory_norm_sentences, min_keep=3):
    sents = _split_sentences(text)
    kept = []
    added_norms = []
    seen_local = []  

    for s in sents:
        s_norm = _normalize_text(s)
        if not s_norm or s_norm in seen_local:
            continue
        if _too_similar(s, memory_norm_sentences) or _too_similar(s, seen_local):
            continue
        kept.append(s)
        seen_local.append(s_norm)
        added_norms.append(s_norm)

    if len(kept) < min_keep:
        for s in sents:
            if len(kept) >= min_keep:
                break
            s_norm = _normalize_text(s)
            if s_norm not in seen_local:
                kept.append(s)
                seen_local.append(s_norm)
                added_norms.append(s_norm)

    final = " ".join(kept).strip()
    return final, added_norms

def _banlist_from_memory(memory_norm_sentences, max_items=12, min_len=6):
    items = []
    for s in memory_norm_sentences[-(max_items*2):]:
        tokens = s.split()
        if len(tokens) >= min_len:
            items.append(" ".join(tokens[:10]))
            if len(items) >= max_items:
                break
    return items

# CRITERIA HELPERS 
def parse_criteria(s: str):
    if not s.strip():
        return {}
    crit = {}
    warnings = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            warnings.append(f" Ignoring '{part}' (missing '=')")
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            crit[k] = float(v.strip())
        except Exception:
            warnings.append(f" Ignoring '{k}' (invalid number: {v})")
    if warnings:
        print("\n".join(warnings))
    total = sum(crit.values()) or 1.0
    return {k: v / total for k, v in crit.items()}

def format_criteria(d: dict) -> str:
    return "(none)" if not d else ", ".join(f"{k}={v:.2f}" for k, v in d.items())

def collect_facts():
    print("Enter background facts & references (press Enter to finish):")
    facts = []
    while True:
        try:
            line = input("- ")
        except EOFError:
            break
        if line is None:
            break
        line = line.strip()
        if line == "":
            break
        facts.append(line)
    return facts

def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def recompute_totals(criteria: dict, cb: list) -> tuple:
    total_a = 0.0
    total_b = 0.0
    for item in cb or []:
        name = str(item.get("name", "")).strip()
        if not name or name not in criteria:
            continue
        w = float(criteria[name])
        a = clamp01(item.get("agent_a", item.get("pro", 0.0)))
        b = clamp01(item.get("agent_b", item.get("con", 0.0)))
        total_a += w * a
        total_b += w * b
    return total_a, total_b

def _print_bullets(label, val):
    print(label + ":")
    if isinstance(val, list):
        for x in val:
            print(" - " + str(x))
    elif isinstance(val, str) and val.strip():
        txt = val.replace("\n", " ")
        parts = []
        for sep in [". ", "; ", ", "]:
            if sep in txt:
                parts = [p.strip() for p in txt.split(sep) if p.strip()]
                break
        if not parts:
            parts = [txt.strip()]
        for p in parts:
            print(" - " + p)
    else:
        print(" -")

def yes(inp: str) -> bool:
    return inp.strip().lower() in {"y","yes","1","true","t"}

#  LLM CALLS (With Banlist & Retry) 
def _agent_call(model, system_prompt, user_prompt, banlist=None, temperature=0.7, max_retries=2, memory_norm=None, min_keep=3):
    banlist = banlist or []
    extra_rules = ""
    if banlist:
        bullets = "\n".join(f"- {b}" for b in banlist)
        extra_rules = f"""
STRICT BANLIST (do NOT reuse the following phrases/ideas/wording; rephrase or introduce new points):
{bullets}
- Do not paraphrase them; provide fresh, distinct substance.
"""

    composed_user = f"{user_prompt}\n{extra_rules}".strip()

    attempt = 0
    last_text = ""
    while attempt <= max_retries:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":composed_user}
            ],
            temperature=temperature,
        )
        out = resp.choices[0].message.content.strip()

        filtered, added_norms = _dedupe_and_enforce_novelty(out, memory_norm or [], min_keep=min_keep)

        if len(filtered) < max(80, int(0.5*len(out))) and attempt < max_retries:
            attempt += 1
            composed_user += "\n\nHARD RULE: Your prior output repeated existing content. Produce a substantially NEW version with different claims/evidence and distinct wording."
            last_text = filtered or out
            continue

        return filtered, added_norms

    return last_text or out, []

def _stance_for(side_name: str) -> str:
    return "support" if side_name.strip().lower().startswith(("pro", "support")) else "oppose"

#  Role Prompters 
def call_opening(side_name, model, statement, facts, criteria, role_label, banlist, memory_norm):
    stance = _stance_for(side_name)
    system = f"""You are '{side_name}' ({role_label}), a senior military strategist and defense policy expert.
You {stance} the statement and must keep this position consistent throughout all rounds.
Be concise, neutral in tone, and strictly avoid arguments or evidence in the opening."""
    user = f"""Main Idea: {statement}
{chr(10).join(f"- {f}" for f in facts) if facts else "(none)"}

Evaluation Criteria (weighted): {criteria if criteria else "(none)"}

TASK: Opening statement (120–180 words). **NO ARGUMENTS OR EVIDENCE**.
Rules:
- No addressing the opponent yet.
- Start with one clear line stating your stance (support/oppose) without "because".
- Briefly describe the main idea and assumptions of your position (natural language).
- Keep a neutral tone.
- Do NOT include claims, justifications, examples, data, or rebuttals.
- Conclude with a single-sentence thesis that is NEW and not reused."""
    return _agent_call(model, system, user, banlist=banlist, temperature=0.6, memory_norm=memory_norm, min_keep=3)

def call_rebuttal(side_name, model, statement, facts, criteria, opponent_opening, role_label, banlist, memory_norm):
    stance = _stance_for(side_name)
    system = f"""You are '{side_name}' ({role_label}), a military and defense affairs expert trained in strategic analysis and national security evaluation.
You {stance} the statement and must keep this position consistent throughout all rounds.
Be direct, evidence-based, and professional."""
    user = f"""Statement: "{statement}"
Your task: Rebut the opponent's opening below (introduce arguments here, not in the opening).

Opponent Opening:
{opponent_opening}

Facts/Context:
{chr(10).join(f"- {f}" for f in facts) if facts else "(none)"}
Criteria (weighted): {criteria if criteria else "(none)"}

Rebuttal (110–160 words):
- Target 2–3 specific propositions implicit in the opponent's stance and refute them with NEW logic/evidence.
- **No repetition:** do NOT restate any sentence or claim already said in earlier turns by either side.
- Expose assumptions, risks, or feasibility gaps.
- Add 2–3 new counter-points with short supporting references or rationale.
- When citing, append the tag like [E1]/[P1] at the end of the sentence if relevant."""
    return _agent_call(model, system, user, banlist=banlist, temperature=0.65, memory_norm=memory_norm, min_keep=4)

def call_counter_rebuttal(side_name, model, statement, facts, criteria, opponent_rebuttal, role_label, banlist, memory_norm):
    stance = _stance_for(side_name)
    system = f"""You are '{side_name}' ({role_label}), a senior military strategist skilled in operational reasoning and defense assessment.
You {stance} this position and must keep it consistent throughout all rounds.
Respond precisely and factually."""
    user = f"""Statement: "{statement}"
Respond briefly to the opponent's rebuttal:

Opponent Rebuttal:
{opponent_rebuttal}

Facts:
{chr(10).join(f"- {f}" for f in facts) if facts else "(none)"}
Criteria: {criteria if criteria else "(none)"}

Counter-Rebuttal (90–130 words):
- **No repetition:** avoid repeating earlier points or sentences (yours or theirs); advance the discussion with NEW substance.
- Correct 1–2 misinterpretations precisely.
- Defend 1 key claim with fresh evidence or operational reasoning.
- Close with a one-line takeaway that is novel.
- When citing, append the tag like [E1]/[P1] if relevant."""
    return _agent_call(model, system, user, banlist=banlist, temperature=0.6, memory_norm=memory_norm, min_keep=3)

def call_free_debate(side_name, model, statement, facts, criteria, opponent_last, role_label, banlist, memory_norm, max_words=140):
    stance = _stance_for(side_name)
    system = f"""You are '{side_name}' ({role_label}), a defense strategy expert and military analyst engaging in a professional debate.
You {stance} the statement and must keep this position consistent throughout all rounds.
Keep it sharp, realistic, and grounded in defense principles."""
    user = f"""Statement: "{statement}"
Reply to the opponent's latest message:

Opponent Last:
{opponent_last or "(none)"}

Facts:
{chr(10).join(f"- {f}" for f in facts) if facts else "(none)"}
Criteria: {criteria if criteria else "(none)"}

Free-Debate Turn (~{max_words} words):
- **No repetition:** do not reuse prior sentences, claims, examples, or sources from any earlier turn.
- Address 1–2 key points from opponent's latest with fresh analysis or data.
- Introduce 1 genuinely NEW operational insight.
- End with one sharp, NEW takeaway line.
- When citing, append the tag like [E1]/[P1] if relevant."""
    return _agent_call(model, system, user, banlist=banlist, temperature=0.65, memory_norm=memory_norm, min_keep=3)

def call_closing(side_name, model, statement, facts, criteria, transcript_snippet, role_label, banlist, memory_norm):
    stance = _stance_for(side_name)
    system = f"""You are '{side_name}' ({role_label}), a senior defense strategist and military policy expert.
You {stance} the statement and must keep this position consistent throughout all rounds.
Deliver a concise, high-level military-informed closing."""
    user = f"""Statement: "{statement}"
Short transcript snippet for context:
{transcript_snippet}

Facts:
{chr(10).join(f"- {f}" for f in facts) if facts else "(none)"}
Criteria: {criteria if criteria else "(none)"}

Closing (80–120 words):
- Synthesize your strongest NEW point vis-à-vis criteria (avoid repetition).
- Acknowledge 1 trade-off honestly (new wording).
- Conclude with a clear, NEW action recommendation.
- When citing, append the tag like [E1]/[P1] if relevant."""
    return _agent_call(model, system, user, banlist=banlist, temperature=0.6, memory_norm=memory_norm, min_keep=3)

#  JUDGE 
def call_llm_judge(statement, criteria, transcript, agent_a, agent_b):
    have_criteria = bool(criteria)
    system = "You are a neutral Judge Agent specialized in defense analysis and military decision evaluation. Return a RAW JSON object only. No prose or markdown."

    if have_criteria:
        user = f"""
Statement: "{statement}"
Provided Evaluation Criteria (weighted): {criteria}

Debate Transcript (ordered turns):
{transcript}

INSTRUCTIONS:
- ALWAYS include two additional criteria besides the provided ones:
  1) "source_credibility"
  2) "public_opinion"
- Return a weight for EVERY criterion. Weights must sum to 1.0.
- Score both sides in [0,1] per criterion and include a short reason.

Return JSON with:
- criteria_breakdown: list of objects {{ "name","weight","agent_a","agent_b","reason" }}
- recommendation: "{agent_a}" | "{agent_b}" | "Defer"
- confidence: float [0,1]
- rationale: short explanation
- conditions, next_steps, tradeoffs, agreements, contradictions: lists of strings
"""
    else:
        user = f"""
Statement: "{statement}"
Evaluation Criteria: (none)

Debate Transcript (ordered turns):
{transcript}

INSTRUCTIONS:
- Infer 3–4 suitable criteria (feasibility, risk, cost, effectiveness, time-to-field, ethics, etc.)
- ALSO include:
  1) "source_credibility"
  2) "public_opinion"
- Normalize weights to sum to 1.0.
- Score both sides per criterion [0,1] with a short reason.

Return JSON with:
- criteria_breakdown: list of objects {{ "name","weight","agent_a","agent_b","reason" }}
- recommendation: "{agent_a}" | "{agent_b}" | "Defer"
- confidence: float [0,1]
- rationale: short explanation
- conditions, next_steps, tradeoffs, agreements, contradictions: lists of strings
"""
    resp = client.chat.completions.create(
        model=MODEL_JUDGE,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else ""

    try:
        data = json.loads(raw)
    except Exception:
        data = {
            "criteria_breakdown": [],
            "recommendation": "Defer",
            "confidence": 0.5,
            "rationale": raw[:500],
            "conditions": [],
            "next_steps": [],
            "tradeoffs": [],
            "agreements": [],
            "contradictions": []
        }

    for k, default in [
        ("criteria_breakdown", []), ("tradeoffs", []),
        ("agreements", []), ("contradictions", []), ("conditions", []),
        ("next_steps", []), ("recommendation", "Defer"), ("confidence", 0.5),
        ("rationale", "")
    ]:
        data.setdefault(k, default)

    for k in ["conditions", "next_steps"]:
        if isinstance(data[k], str):
            txt = data[k].replace("\n", " ")
            items = []
            for sep in [". ", "; ", ", "]:
                if sep in txt:
                    items = [p.strip() for p in txt.split(sep) if p.strip()]
                    break
            if not items and txt.strip():
                items = [txt.strip()]
            data[k] = items

    return data

# MAIN 
def main():
    print(" Debate Agent (Up to 10 Rounds) ")

    statement = input("Enter the debate statement or question: ").strip()
    if not statement:
        print("Error: You must enter a statement.")
        return

    agent_a = "Proponent"
    agent_b = "Opponent"


    # Number of rounds (6..10)
    while True:
        rounds_inp = input("How many rounds? [6..10] (>=4 means last round is Closings): ").strip()
        if not rounds_inp:
            rounds = 6
            break
        try:
            rounds = int(rounds_inp)
            if 6 <= rounds <= 10:
                break
        except Exception:
            pass
        print("Please enter an integer between 6 and 10.")

    print(SEPARATOR)

    # Criteria
    use_criteria = yes(input("Use evaluation criteria? [y/N]: "))
    criteria = {}
    if use_criteria:
        crit_str = input("Enter criteria (e.g., cost=0.35,feasibility=0.25,risk=0.2,impact=0.2): ").strip()
        criteria = parse_criteria(crit_str)
    print("Using evaluation criteria (normalized):", format_criteria(criteria))
    print(SEPARATOR)

    # Facts
    use_facts = yes(input("Add background facts? [y/N]: "))
    facts = collect_facts() if use_facts else []
    if not facts:
        print("Facts/Context: (none)")

    #  Tavily 
    use_web = yes(input("Fetch web sources (Tavily)? [y/N]: "))

    if use_web and not TAVILY_API_KEY:
        print("WARNING: TAVILY_API_KEY is empty; skipping web fetch.")
    elif use_web:
        print("Using Tavily (last 30 days)...")
        base_q = f'{statement} analysis OR report OR policy OR public opinion OR debate OR reactions'

        hits_e = tavily_search(
            base_q,
            max_results=12,
            time_range="month",
            exclude_domains=SOCIAL_HOSTS
        )

        hits_p = tavily_search(
            f'{statement} public opinion OR survey OR poll OR reactions OR debate',
            max_results=12,
            time_range="month",
            include_domains=SOCIAL_HOSTS
        )

        def _dedup(urls):
            seen, out = set(), []
            for x in urls:
                u = (x.get("url") or "").strip()
                if u and u not in seen:
                    seen.add(u); out.append(x)
            return out

        evidence = _dedup(hits_e)
        public   = _dedup(hits_p)

        if not public:
            q_social = (
                f'{statement} "public opinion" OR survey OR poll OR reactions '
                'site:twitter.com OR site:x.com OR site:reddit.com OR site:youtube.com '
                'OR site:bsky.app OR site:bluesky.social OR site:quora.com OR site:tiktok.com'
            )
            public = _dedup(tavily_search(q_social, max_results=12, time_range="month"))

        for i, it in enumerate(evidence[:5], start=1):
            facts.append(fmt_fact(f"E{i}", it))
        for i, it in enumerate(public[:5], start=1):
            facts.append(fmt_fact(f"P{i}", it))

        if evidence or public:
            print("Injected web facts:")
            for f in facts[-(len(evidence[:5]) + len(public[:5])):]:
                print("-", f)


    transcript = []

    #  Anti-repetition memory per agent (normalized sentences) 
    memory = {
        agent_a: [], 
        agent_b: []
    }

    #  Round 1: Openings 
    print(SEPARATOR)
    print("Round 1: Openings")

    ban_a = _banlist_from_memory(memory[agent_a])
    opening_a, added_a = call_opening(agent_a, MODEL_PRO, statement, facts, criteria, "Agent A (supports)", ban_a, memory[agent_a])
    memory[agent_a].extend(added_a)
    transcript.append({"round":1,"side":agent_a,"type":"opening","text":opening_a})
    print(f"\n[{agent_a} Opening]\n{opening_a}\n")
    

    ban_b = _banlist_from_memory(memory[agent_b])
    opening_b, added_b = call_opening(agent_b, MODEL_CON, statement, facts, criteria, "Agent B (opposes)", ban_b, memory[agent_b])
    memory[agent_b].extend(added_b)
    transcript.append({"round":1,"side":agent_b,"type":"opening","text":opening_b})
    print(f"\n[{agent_b} Opening]\n{opening_b}\n")

    #  Round 2: Rebuttals 
    if rounds >= 2:
        print(SEPARATOR)
        print("Round 2: Rebuttals")

        ban_a = _banlist_from_memory(memory[agent_a])
        rebuttal_a, added_a = call_rebuttal(agent_a, MODEL_PRO, statement, facts, criteria, opponent_opening=opening_b, role_label="Agent A (supports)", banlist=ban_a, memory_norm=memory[agent_a])
        memory[agent_a].extend(added_a)
        transcript.append({"round":2,"side":agent_a,"type":"rebuttal","text":rebuttal_a})
        print(f"\n[{agent_a} Rebuttal]\n{rebuttal_a}\n")

        ban_b = _banlist_from_memory(memory[agent_b])
        rebuttal_b, added_b = call_rebuttal(agent_b, MODEL_CON, statement, facts, criteria, opponent_opening=opening_a, role_label="Agent B (opposes)", banlist=ban_b, memory_norm=memory[agent_b])
        memory[agent_b].extend(added_b)
        transcript.append({"round":2,"side":agent_b,"type":"rebuttal","text":rebuttal_b})
        print(f"\n[{agent_b} Rebuttal]\n{rebuttal_b}\n")
    else:
        rebuttal_a = rebuttal_b = ""

    #  Round 3: Counter-Rebuttals 
    if rounds >= 3:
        print(SEPARATOR)
        print("Round 3: Counter-Rebuttals")

        ban_a = _banlist_from_memory(memory[agent_a])
        counter_a, added_a = call_counter_rebuttal(agent_a, MODEL_PRO, statement, facts, criteria, opponent_rebuttal=rebuttal_b or "(none)", role_label="Agent A (supports)", banlist=ban_a, memory_norm=memory[agent_a])
        memory[agent_a].extend(added_a)
        transcript.append({"round":3,"side":agent_a,"type":"counter-rebuttal","text":counter_a})
        print(f"\n[{agent_a} Counter-Rebuttal]\n{counter_a}\n")

        ban_b = _banlist_from_memory(memory[agent_b])
        counter_b, added_b = call_counter_rebuttal(agent_b, MODEL_CON, statement, facts, criteria, opponent_rebuttal=rebuttal_a or "(none)", role_label="Agent B (opposes)", banlist=ban_b, memory_norm=memory[agent_b])
        memory[agent_b].extend(added_b)
        transcript.append({"round":3,"side":agent_b,"type":"counter-rebuttal","text":counter_b})
        print(f"\n[{agent_b} Counter-Rebuttal]\n{counter_b}\n")

    #  Rounds 4..(N-1): Free Debate 
    last_a = counter_a if rounds >= 3 else (opening_a if rounds == 1 else rebuttal_a)
    last_b = counter_b if rounds >= 3 else (opening_b if rounds == 1 else rebuttal_b)

    if rounds >= 4:
        for r in range(4, rounds):  # up to N-1
            print(SEPARATOR)
            print(f"Round {r}: Free Debate")

            ban_a = _banlist_from_memory(memory[agent_a])
            free_a, added_a = call_free_debate(agent_a, MODEL_PRO, statement, facts, criteria, opponent_last=last_b, role_label="Agent A (supports)", banlist=ban_a, memory_norm=memory[agent_a], max_words=140)
            memory[agent_a].extend(added_a)
            transcript.append({"round":r,"side":agent_a,"type":"free-debate","text":free_a})
            print(f"\n[{agent_a} Free-Debate]\n{free_a}\n")
            last_a = free_a

            ban_b = _banlist_from_memory(memory[agent_b])
            free_b, added_b = call_free_debate(agent_b, MODEL_CON, statement, facts, criteria, opponent_last=last_a, role_label="Agent B (opposes)", banlist=ban_b, memory_norm=memory[agent_b], max_words=140)
            memory[agent_b].extend(added_b)
            transcript.append({"round":r,"side":agent_b,"type":"free-debate","text":free_b})
            print(f"\n[{agent_b} Free-Debate]\n{free_b}\n")
            last_b = free_b

    #  Final Round (N): Closings if N>=4 
    if rounds >= 4:
        print(SEPARATOR)
        print(f"Round {rounds}: Closings")
        snippet = (last_a or "")[:300] + (" ..." if last_a else "")
        snippet += "\n" + ((last_b or "")[:300] + (" ..." if last_b else ""))

        ban_a = _banlist_from_memory(memory[agent_a])
        closing_a, added_a = call_closing(agent_a, MODEL_PRO, statement, facts, criteria, snippet or "(no snippet)", "Agent A (supports)", banlist=ban_a, memory_norm=memory[agent_a])
        memory[agent_a].extend(added_a)
        transcript.append({"round":rounds,"side":agent_a,"type":"closing","text":closing_a})
        print(f"\n[{agent_a} Closing]\n{closing_a}\n")

        ban_b = _banlist_from_memory(memory[agent_b])
        closing_b, added_b = call_closing(agent_b, MODEL_CON, statement, facts, criteria, snippet or "(no snippet)", "Agent B (opposes)", banlist=ban_b, memory_norm=memory[agent_b])
        memory[agent_b].extend(added_b)
        transcript.append({"round":rounds,"side":agent_b,"type":"closing","text":closing_b})
        print(f"\n[{agent_b} Closing]\n{closing_b}\n")

    #  Judge 
    print(SEPARATOR)
    print("Judge Agent analyzing full transcript and issuing verdict...")
    transcript_str = json.dumps(transcript, ensure_ascii=False, indent=2)
    verdict = call_llm_judge(statement, criteria, transcript_str, agent_a, agent_b)

    cb = verdict.get("criteria_breakdown", [])

    criteria_used = {}
    for item in cb:
        name = str(item.get("name", "")).strip()
        w = item.get("weight", None)
        if name and isinstance(w, (int, float)) and w >= 0:
            criteria_used[name] = float(w)
    if not criteria_used:
        if criteria:
            criteria_used = criteria.copy()
        elif cb:
            eq = 1.0 / len(cb)
            criteria_used = {str(it.get("name", f"crit_{i+1}")): eq for i, it in enumerate(cb)}
        else:
            criteria_used = {}
    s = sum(criteria_used.values()) or 1.0
    criteria_used = {k: v / s for k, v in criteria_used.items()}

    local_a, local_b = recompute_totals(criteria_used, cb)
    local_a, local_b = round(local_a, 4), round(local_b, 4)

    delta = local_a - local_b
    if abs(delta) < 0.05:
        local_rec = "Defer"
    elif delta > 0:
        local_rec = agent_a
    else:
        local_rec = agent_b

    local_conf = round(0.55 + min(0.45, abs(delta)), 2)
    verdict.setdefault("recommendation", local_rec)
    verdict.setdefault("confidence", local_conf)
    verdict["total_" + agent_a] = local_a
    verdict["total_" + agent_b] = local_b

    # Persist log
    with open("debate_log.jsonl","a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat()+"Z",
            "statement":statement,"agent_a":agent_a,"agent_b":agent_b,
            "criteria":criteria_used,
            "facts":facts,"transcript":transcript,
            "criteria_breakdown":cb,"total_a":local_a,"total_b":local_b,
            "recommendation":verdict.get("recommendation"),
            "confidence":verdict.get("confidence"),
            "rationale":verdict.get("rationale",""),
            "conditions":verdict.get("conditions",[]),
            "next_steps":verdict.get("next_steps",[]),
            "tradeoffs":verdict.get("tradeoffs",[]),
            "agreements":verdict.get("agreements",[]),
            "contradictions":verdict.get("contradictions",[])
        }, ensure_ascii=False) + "\n")

    #  Console Output
    print(SEPARATOR)
    print("Debate Transcript (condensed)")
    for t in transcript:
        print(f"[Round {t['round']}] {t['side']} - {t['type']}")
        print(t["text"] + "\n")

    print(SEPARATOR)
    print("Criteria (used)")
    if criteria_used and not criteria:
        print("Inferred by Judge (normalized):", format_criteria(criteria_used))
    elif criteria_used and criteria:
        print("Provided + required (normalized):", format_criteria(criteria_used))
    else:
        print("(none)")

    print(SEPARATOR)
    print(" Criteria weighted comparison ")
    if cb and criteria_used:
        for item in cb:
            name = item.get('name', 'criterion')
            a_score = item.get('agent_a', item.get('pro'))
            b_score = item.get('agent_b', item.get('con'))
            w = float(criteria_used.get(name, 0))
            print(f"- {name}: {agent_a}={a_score}, {agent_b}={b_score}, weight={w}")
            reason = item.get("reason", "")
            if reason:
                print(f"  reason: {reason}")
    else:
        print("(no per criterion breakdown or no criteria available)")
    print(f"Total {agent_a} (recomputed): {local_a}")
    print(f"Total {agent_b} (recomputed): {local_b}")

    print(SEPARATOR)
    print(" Verdict ")
    print(f"Recommendation: {verdict.get('recommendation', 'Defer')}")
    print(f"Confidence: {verdict.get('confidence', 0.5)}")
    print(f"Rationale: {verdict.get('rationale', '')}")

    _print_bullets("Conditions", verdict.get("conditions"))
    _print_bullets("Next Steps", verdict.get("next_steps"))

    tradeoffs = verdict.get("tradeoffs", [])
    agreements = verdict.get("agreements", [])
    contradictions = verdict.get("contradictions", [])
    if tradeoffs or agreements or contradictions:
        print(SEPARATOR)
        print("Debate Insights ")
        print("Trade-offs:", ", ".join(tradeoffs) if tradeoffs else "-")
        print("Agreements:", ", ".join(agreements) if agreements else "-")
        print("Contradictions:", ", ".join(contradictions) if contradictions else "-")

    print(SEPARATOR)
    print("Debate Completed.")

if __name__ == "__main__":
    main()

