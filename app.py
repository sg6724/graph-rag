"""MedGraph-RAG demo UI: uv run streamlit run app.py"""
import json

import streamlit as st
import streamlit.components.v1 as components

from medgraph import config
from medgraph.llm_router import LLMError
from medgraph.metrics import Metrics
from medgraph.pipelines import load_engine
from medgraph.viz import subgraph_html

st.set_page_config(page_title="MedGraph-RAG", page_icon="💊", layout="wide")

MODES = {"cached": "GraphRAG + semantic cache", "graphrag": "GraphRAG", "vanilla": "Vanilla RAG",
         "compare": "Compare: Vanilla vs GraphRAG"}
DATASETS = {"medline": "MedlinePlus health topics (plain English)", "fda": "FDA drug labels (drug interactions)"}


@st.cache_resource(show_spinner="Loading graph, index and embedding model…")
def get_engine(dataset: str):
    """Supabase (Postgres + pgvector) when reachable; otherwise the committed local files."""
    try:
        return load_engine(dataset=dataset), "Supabase Postgres + pgvector"
    except Exception as e:  # no DATABASE_URL, no network, …
        return load_engine(dataset=dataset, backend="files"), f"local files (Supabase unavailable: {type(e).__name__})"


ss = st.session_state
with st.sidebar:
    dataset = st.selectbox("Dataset", list(DATASETS), format_func=DATASETS.get, key="dataset")
engine, storage = get_engine(dataset)
EXAMPLES = engine.profile.examples
if ss.get("active_dataset") != dataset:  # switching datasets starts a clean session
    for k in ("last", "compare", "update_log"):
        ss[k] = None
    ss["metrics"] = Metrics()
    ss["question"] = EXAMPLES[0]
    ss["active_dataset"] = dataset

st.title("💊 MedGraph-RAG")
st.caption(f"Graph-based semantic search over **{DATASETS[dataset]}** + a graph-aware semantic cache · "
           f"storage: {storage} · **Educational demo — not medical advice.**")

# ---------------- sidebar ----------------
with st.sidebar:
    mode = st.radio("Pipeline", list(MODES), format_func=MODES.get)
    st.divider()
    st.subheader("Live metrics")
    metrics_box = st.container()  # filled at the end of the script, after this run's question is recorded
    st.divider()
    st.subheader("Simulate a source update")
    updates = [u for u in json.loads(config.DEMO_UPDATES_PATH.read_text(encoding="utf-8"))
               if u.get("dataset", "fda") == dataset]
    pick = st.selectbox("Update", range(len(updates)), format_func=lambda i: updates[i]["label"], key="update_pick")
    if st.button("Apply update", width="stretch"):
        u = updates[pick]
        with st.spinner("Re-extracting the changed passage and invalidating affected answers…"):
            try:
                ss.update_log = engine.update_label(u["drug"], u["section"], u["text"])
            except LLMError as e:
                st.error(f"All LLM providers are busy right now — try again in a moment. ({e})")
    if st.button("Reset demo", width="stretch"):
        warm = (config.MEDLINE_DIR if dataset == "medline" else config.DATA) / "cache.json"
        if hasattr(engine.cache, "conn"):  # Supabase cache persists: restore the pre-warmed answers
            engine.cache.load(warm) if warm.exists() else engine.cache.clear()
        st.cache_resource.clear()
        ss.clear()
        st.rerun()

# ---------------- ask ----------------
st.write("**Try:** " + " · ".join(f"`{q}`" for q in EXAMPLES))
with st.form("ask"):
    q = st.text_input("Ask about a drug combination", key="question")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and q.strip():
    try:
        with st.spinner("Thinking…"):
            if mode == "compare":
                ss.compare = (engine.vanilla(q), engine.graphrag(q))
                ss.last = None
                for a in ss.compare:
                    ss.metrics.record(a)
            else:
                ss.last = getattr(engine, mode)(q)
                ss.compare = None
                ss.metrics.record(ss.last)
    except LLMError as e:
        st.error(f"All LLM providers are rate-limited right now — cached answers still work. ({e})")


def show_answer(a):
    if a.cache_hit:
        st.success(f"⚡ Cache HIT · similarity {a.cache_score:.3f} · 0 LLM calls · {a.timings['total_ms']:.0f} ms\n\n"
                   f"Reused the answer to: “{a.served_from}”")
    else:
        st.info(f"{MODES.get(a.mode, a.mode)} · {a.provider} · {a.timings.get('total_ms', 0) / 1000:.1f} s")
    if a.blocked_by_key:
        st.warning(f"🛡️ A similar cached question was **blocked** because it's about different drugs: "
                   f"“{a.blocked_by_key}”")
    st.markdown(a.text)
    if a.citations:
        with st.expander(f"Sources ({len(a.citations)})"):
            for cid in a.citations:
                c = engine.chunks.get(cid)
                if c:
                    st.markdown(f"**{cid}** — {c['text'][:600]}…")


if ss.compare:
    left, right = st.columns(2)
    with left:
        st.subheader("Vanilla RAG")
        show_answer(ss.compare[0])
    with right:
        st.subheader("GraphRAG")
        show_answer(ss.compare[1])
        components.html(subgraph_html(engine.graph.node_types(), ss.compare[1].path_nodes,
                                      ss.compare[1].path_edges, ss.compare[1].entities), height=540)
elif ss.last:
    a = ss.last
    left, right = st.columns([1, 1])
    with left:
        show_answer(a)
    with right:
        if a.path_nodes:
            st.caption("Knowledge-graph path used for this answer (large nodes = entities in your question)")
            components.html(subgraph_html(engine.graph.node_types(), a.path_nodes, a.path_edges, a.entities),
                            height=540)
        else:
            st.caption("Vanilla RAG uses text chunks only — no graph.")

if ss.update_log:
    u = ss.update_log
    st.divider()
    st.subheader(f"Label update applied: {u['drug']} ({u['chunk_id']})")
    st.write("Graph nodes changed: " + ", ".join(f"`{n}`" for n in u["changed_nodes"]))
    c1, c2 = st.columns(2)
    with c1:
        st.error(f"Evicted ({len(u['evicted'])}) — depended on changed facts")
        for qq in u["evicted"]:
            st.write("• " + qq)
    with c2:
        st.success(f"Kept ({len(u['retained'])}) — still valid, still served from cache")
        for qq in u["retained"]:
            st.write("• " + qq)


def render_metrics():
    s = ss.metrics.summary()
    with metrics_box:
        c1, c2 = st.columns(2)
        c1.metric("Cache hit rate", f"{s['hit_rate']:.0%}", f"{s['hits']}/{s['cache_queries']}")
        c2.metric("LLM calls saved", s["llm_calls_saved"])
        c1.metric("Avg hit latency", f"{s['avg_hit_ms']:.0f} ms")
        c2.metric("Avg miss latency", f"{s['avg_miss_ms'] / 1000:.1f} s")
        st.metric("Est. $ saved", f"${s['usd_saved']:.2f}", help=f"${config.REF_USD_PER_CALL}/call reference price")
        st.caption(f"Cache entries: {len(engine.cache.entries)} · threshold {engine.cache.threshold:.2f}")
        st.caption(f"LLM providers used: {engine.router.by_provider or 'none yet'}")


render_metrics()
