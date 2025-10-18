# ui_streamlit.py
import os
import json
import pandas as pd
import streamlit as st

import Debate_Agent as DA

st.set_page_config(page_title="Debate Agent", layout="wide")
st.title("Debate Agent")

with st.sidebar:
    st.header("Settings")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    tavily_key = st.text_input("Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY", ""))

    fetch_web = st.checkbox("Fetch web sources (Tavily)", value=True)

    rounds = st.slider("Rounds", min_value=6, max_value=10, value=6, step=1)

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    try:
        DA.client.api_key = openai_key
    except Exception:
        pass

if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key
    DA.TAVILY_API_KEY = tavily_key

st.subheader("Setup")
statement = st.text_area("Debate statement / question", height=120, placeholder="Write the debate statement here…")

agent_a = "Proponent"
agent_b = "Opponent"
cols = st.columns(2)
cols[0].text_input("Agent A", value=agent_a, disabled=True)
cols[1].text_input("Agent B", value=agent_b, disabled=True)

run_btn = st.button("Run Debate")

def _get_text(res):
    if isinstance(res, (tuple, list)) and len(res) >= 1:
        return res[0]
    return res

def _dedup(urls):
    seen, out = set(), []
    for x in urls or []:
        u = (x.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u); out.append(x)
    return out

def _recompute_totals(cb):
    try:
        wa = sum(float(x.get("weight", 0)) * float(x.get("agent_a", x.get("pro", 0))) for x in cb)
        wb = sum(float(x.get("weight", 0)) * float(x.get("agent_b", x.get("con", 0))) for x in cb)
        return wa, wb
    except Exception:
        return None, None

if run_btn:
    if not openai_key:
        st.error("Please provide an OpenAI API key.")
        st.stop()

    facts = []
    if fetch_web:
        if not tavily_key:
            st.warning("No Tavily key set — skipping web sources.")
        else:
            with st.status("Collecting online references...", expanded=False) as s:
                base_q = f'{statement} analysis OR report OR policy OR public opinion OR debate OR reactions'

                hits_e = DA.tavily_search(
                    base_q, max_results=12, time_range="month",
                    exclude_domains=DA.SOCIAL_HOSTS
                )
                hits_p = DA.tavily_search(
                    f'{statement} public opinion OR survey OR poll OR reactions OR debate',
                    max_results=12, time_range="month",
                    include_domains=DA.SOCIAL_HOSTS
                )

                evidence = _dedup(hits_e)
                public = _dedup(hits_p)

                if not public:
                    q_social = (
                        f'{statement} "public opinion" OR survey OR poll OR reactions '
                        'site:twitter.com OR site:x.com OR site:reddit.com OR site:youtube.com '
                        'OR site:bsky.app OR site:bluesky.social OR site:quora.com OR site:tiktok.com'
                    )
                    public = _dedup(DA.tavily_search(q_social, max_results=12, time_range="month"))

                for i, it in enumerate(evidence[:5], start=1):
                    facts.append(DA.fmt_fact(f"E{i}", it))
                for i, it in enumerate(public[:5], start=1):
                    facts.append(DA.fmt_fact(f"P{i}", it))

                s.update(label="Web references ready", state="complete")

            with st.expander("Extracted Evidence & Public Opinion Sources", expanded=True):
                for f in facts:
                    st.write("- " + f)

    memory = {agent_a: [], agent_b: []}
    transcript = []

    st.markdown("### Round 1: Openings")

    ban_a = DA._banlist_from_memory(memory[agent_a])
    box_a = st.empty()
    with st.spinner("Proponent is writing opening…"):
        res_a = DA.call_opening(agent_a, DA.MODEL_PRO, statement, facts, {}, f"{agent_a} (supports)", ban_a, memory[agent_a])
    opening_a = _get_text(res_a)
    if isinstance(res_a, (tuple, list)) and len(res_a) > 1:
        memory[agent_a].extend(res_a[1])
    transcript.append({"round": 1, "side": agent_a, "type": "opening", "text": opening_a})
    box_a.markdown(f"**{agent_a} Opening**\n\n{opening_a}")

    ban_b = DA._banlist_from_memory(memory[agent_b])
    box_b = st.empty()
    with st.spinner("Opponent is writing opening…"):
        res_b = DA.call_opening(agent_b, DA.MODEL_CON, statement, facts, {}, f"{agent_b} (opposes)", ban_b, memory[agent_b])
    opening_b = _get_text(res_b)
    if isinstance(res_b, (tuple, list)) and len(res_b) > 1:
        memory[agent_b].extend(res_b[1])
    transcript.append({"round": 1, "side": agent_b, "type": "opening", "text": opening_b})
    box_b.markdown(f"**{agent_b} Opening**\n\n{opening_b}")

    rebuttal_a = rebuttal_b = ""
    if rounds >= 2:
        st.markdown("### Round 2: Rebuttals")

        ban_a = DA._banlist_from_memory(memory[agent_a])
        box_ra = st.empty()
        with st.spinner("Proponent rebuttal…"):
            res = DA.call_rebuttal(agent_a, DA.MODEL_PRO, statement, facts, {}, opponent_opening=opening_b, role_label=f"{agent_a} (supports)", banlist=ban_a, memory_norm=memory[agent_a])
        rebuttal_a = _get_text(res)
        if isinstance(res, (tuple, list)) and len(res) > 1:
            memory[agent_a].extend(res[1])
        transcript.append({"round": 2, "side": agent_a, "type": "rebuttal", "text": rebuttal_a})
        box_ra.markdown(f"**{agent_a} Rebuttal**\n\n{rebuttal_a}")

        ban_b = DA._banlist_from_memory(memory[agent_b])
        box_rb = st.empty()
        with st.spinner("Opponent rebuttal…"):
            res = DA.call_rebuttal(agent_b, DA.MODEL_CON, statement, facts, {}, opponent_opening=opening_a, role_label=f"{agent_b} (opposes)", banlist=ban_b, memory_norm=memory[agent_b])
        rebuttal_b = _get_text(res)
        if isinstance(res, (tuple, list)) and len(res) > 1:
            memory[agent_b].extend(res[1])
        transcript.append({"round": 2, "side": agent_b, "type": "rebuttal", "text": rebuttal_b})
        box_rb.markdown(f"**{agent_b} Rebuttal**\n\n{rebuttal_b}")

    counter_a = counter_b = ""
    if rounds >= 3:
        st.markdown("### Round 3: Counter-Rebuttals")

        ban_a = DA._banlist_from_memory(memory[agent_a])
        box_ca = st.empty()
        with st.spinner("Proponent counter-rebuttal…"):
            res = DA.call_counter_rebuttal(agent_a, DA.MODEL_PRO, statement, facts, {}, opponent_rebuttal=rebuttal_b or "(none)", role_label=f"{agent_a} (supports)", banlist=ban_a, memory_norm=memory[agent_a])
        counter_a = _get_text(res)
        if isinstance(res, (tuple, list)) and len(res) > 1:
            memory[agent_a].extend(res[1])
        transcript.append({"round": 3, "side": agent_a, "type": "counter-rebuttal", "text": counter_a})
        box_ca.markdown(f"**{agent_a} Counter-Rebuttal**\n\n{counter_a}")

        ban_b = DA._banlist_from_memory(memory[agent_b])
        box_cb = st.empty()
        with st.spinner("Opponent counter-rebuttal…"):
            res = DA.call_counter_rebuttal(agent_b, DA.MODEL_CON, statement, facts, {}, opponent_rebuttal=rebuttal_a or "(none)", role_label=f"{agent_b} (opposes)", banlist=ban_b, memory_norm=memory[agent_b])
        counter_b = _get_text(res)
        if isinstance(res, (tuple, list)) and len(res) > 1:
            memory[agent_b].extend(res[1])
        transcript.append({"round": 3, "side": agent_b, "type": "counter-rebuttal", "text": counter_b})
        box_cb.markdown(f"**{agent_b} Counter-Rebuttal**\n\n{counter_b}")

    last_a = counter_a if rounds >= 3 else (rebuttal_a if rounds >= 2 else opening_a)
    last_b = counter_b if rounds >= 3 else (rebuttal_b if rounds >= 2 else opening_b)

    if rounds >= 4:
        for r in range(4, rounds):
            st.markdown(f"### Round {r}: Free Debate")

            ban_a = DA._banlist_from_memory(memory[agent_a])
            box_fa = st.empty()
            with st.spinner("Proponent free-debate…"):
                res = DA.call_free_debate(agent_a, DA.MODEL_PRO, statement, facts, {}, opponent_last=last_b, role_label=f"{agent_a} (supports)", banlist=ban_a, memory_norm=memory[agent_a], max_words=140)
            free_a = _get_text(res)
            if isinstance(res, (tuple, list)) and len(res) > 1:
                memory[agent_a].extend(res[1])
            transcript.append({"round": r, "side": agent_a, "type": "free-debate", "text": free_a})
            box_fa.markdown(f"**{agent_a} Free-Debate**\n\n{free_a}")
            last_a = free_a

            ban_b = DA._banlist_from_memory(memory[agent_b])
            box_fb = st.empty()
            with st.spinner("Opponent free-debate…"):
                res = DA.call_free_debate(agent_b, DA.MODEL_CON, statement, facts, {}, opponent_last=last_a, role_label=f"{agent_b} (opposes)", banlist=ban_b, memory_norm=memory[agent_b], max_words=140)
            free_b = _get_text(res)
            if isinstance(res, (tuple, list)) and len(res) > 1:
                memory[agent_b].extend(res[1])
            transcript.append({"round": r, "side": agent_b, "type": "free-debate", "text": free_b})
            box_fb.markdown(f"**{agent_b} Free-Debate**\n\n{free_b}")
            last_b = free_b

        st.markdown(f"### Round {rounds}: Closings")
        snippet = ""
        if last_a:
            snippet += last_a[:300] + (" ..." if len(last_a) > 300 else "")
        if last_b:
            snippet += "\n" + last_b[:300] + (" ..." if len(last_b) > 300 else "")

        box_cl_a = st.empty()
        with st.spinner("Proponent closing…"):
            res = DA.call_closing(agent_a, DA.MODEL_PRO, statement, facts, {}, snippet or "(no snippet)", f"{agent_a} (supports)", banlist=DA._banlist_from_memory(memory[agent_a]), memory_norm=memory[agent_a])
        closing_a = _get_text(res)
        transcript.append({"round": rounds, "side": agent_a, "type": "closing", "text": closing_a})
        box_cl_a.markdown(f"**{agent_a} Closing**\n\n{closing_a}")

        box_cl_b = st.empty()
        with st.spinner("Opponent closing…"):
            res = DA.call_closing(agent_b, DA.MODEL_CON, statement, facts, {}, snippet or "(no snippet)", f"{agent_b} (opposes)", banlist=DA._banlist_from_memory(memory[agent_b]), memory_norm=memory[agent_b])
        closing_b = _get_text(res)
        transcript.append({"round": rounds, "side": agent_b, "type": "closing", "text": closing_b})
        box_cl_b.markdown(f"**{agent_b} Closing**\n\n{closing_b}")

    st.markdown("---")
    st.markdown("## Judge Verdict")
    transcript_str = json.dumps(transcript, ensure_ascii=False, indent=2)
    verdict = DA.call_llm_judge(statement, {}, transcript_str, agent_a, agent_b)

    with st.expander("Summary", expanded=True):
        c1, c2 = st.columns(2)
        c1.metric("Recommendation", verdict.get("recommendation", "Defer"))
        conf = verdict.get("confidence", 0.0)
        try:
            c2.metric("Confidence", f"{float(conf)*100:.1f}%")
        except Exception:
            c2.metric("Confidence", f"{conf}")
        if verdict.get("rationale"):
            st.markdown("**Rationale**")
            st.write(verdict["rationale"])

    cb = verdict.get("criteria_breakdown", [])
    tot_a, tot_b = _recompute_totals(cb)

    with st.expander("Criteria Breakdown", expanded=bool(cb)):
        if cb:
            df = pd.DataFrame(cb).rename(columns={
                "name": "criterion",
                "agent_a": agent_a,
                "agent_b": agent_b
            })
            if tot_a is not None:
                total_row = pd.DataFrame([{
                    "criterion": "TOTAL",
                    "weight": "", 
                    agent_a: round(tot_a, 3),
                    agent_b: round(tot_b, 3),
                    "reason": ""
                }])
                df = pd.concat([df, total_row], ignore_index=True)


            st.dataframe(df, use_container_width=True)

            m1, m2 = st.columns(2)
            if tot_a is not None:
                m1.metric(f"Total — {agent_a}", f"{tot_a:.3f}")
                m2.metric(f"Total — {agent_b}", f"{tot_b:.3f}")
        else:
            st.caption("No per-criterion scores were returned by the Judge.")


    def _list_exp(title: str, items):
        with st.expander(title, expanded=bool(items)):
            if items:
                for it in items:
                    st.write(f"- {it}")
            else:
                st.caption("—")

    _list_exp("Conditions", verdict.get("conditions", []))
    _list_exp("Next Steps", verdict.get("next_steps", []))
    _list_exp("Trade-offs", verdict.get("tradeoffs", []))
    _list_exp("Agreements", verdict.get("agreements", []))
    _list_exp("Contradictions", verdict.get("contradictions", []))

    st.markdown("---")
    st.markdown("### Downloads")
    st.download_button("Download Transcript (JSON)", transcript_str.encode("utf-8"), "debate_transcript.json", "application/json")
    st.download_button("Download Judge Verdict (JSON)", json.dumps(verdict, ensure_ascii=False, indent=2).encode("utf-8"), "debate_verdict.json", "application/json")
