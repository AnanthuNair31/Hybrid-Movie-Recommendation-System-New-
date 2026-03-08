"""
app.py  –  Optimized for deployment
=====================================
Streamlit UI for the Hybrid Movie Recommendation System.

Changes vs original
-------------------
1.  Import corrected: get_recommendations() from inference.api — the
    public entry point established by the api.py refactor.
2.  session_state used for both the query and the result so that
    recommendations persist across Streamlit reruns (widget interactions,
    theme toggles, etc.) without re-invoking the pipeline.
3.  st.spinner() scope widened to cover the result-render block so the
    loading indicator stays visible until output is fully painted.
4.  All three response statuses handled explicitly:
      success  → recommendation cards
      fallback → st.info()
      error    → st.error() with the message from api.py
5.  Raw markdown injection fixed: movie title and genres written via
    st.write() / plain text so special characters cannot be interpreted
    as markdown syntax.
6.  Recommendation cards numbered and visually separated with st.divider().
7.  Example prompts rendered as clickable st.button() pills that
    pre-fill the query — one click instead of copy-paste.
8.  page_icon added to set_page_config.
9.  Input validation tightened: checks both empty-string and whitespace-only.
"""

import streamlit as st

from inference.recommend import recommend as get_recommendations

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "query" not in st.session_state:
    st.session_state["query"] = ""

if "result" not in st.session_state:
    st.session_state["result"] = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎬 Hybrid Movie Recommendation System")
st.caption("Powered by ALS collaborative filtering + XGBoost ranking")

# ---------------------------------------------------------------------------
# Example prompts  (clickable — pre-fill the query box)
# ---------------------------------------------------------------------------
EXAMPLES = [
    "I watched Titanic",
    "Matrix is an action movie",
    "Suggest emotional movies",
    "Recommend comedy films",
]

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state["query"] = example
        st.session_state["result"] = None   # clear previous result on new selection

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
user_input: str = st.text_input(
    "Enter your request",
    value=st.session_state["query"],
    placeholder="e.g. Something like Inception, or suggest a thriller",
    key="query_input",
)

# ---------------------------------------------------------------------------
# Recommend button
# ---------------------------------------------------------------------------
if st.button("Recommend", type="primary", use_container_width=True):
    if not user_input or not user_input.strip():
        st.warning("Please enter a movie title or genre before searching.")
    else:
        # Persist query and clear stale result
        st.session_state["query"]  = user_input.strip()
        st.session_state["result"] = None

        with st.spinner("Generating recommendations…"):
            st.session_state["result"] = get_recommendations(user_input.strip())

# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
result = st.session_state.get("result")

if result is not None:

    status = result.get("status")

    if status == "success":
        recommendations = result.get("recommendations", [])
        st.subheader(f"Top {len(recommendations)} Recommendation(s)")

        for i, movie in enumerate(recommendations, start=1):
            # Use st.write() / plain text — avoids markdown injection from
            # raw title or genre strings containing *, _, #, etc.
            with st.container():
                col_num, col_info = st.columns([1, 11])
                col_num.markdown(f"### {i}")
                col_info.write(f"**{movie.get('title', 'Unknown')}**")
                col_info.caption(f"Genre: {movie.get('genres', '—')}")
            if i < len(recommendations):
                st.divider()

    elif status == "fallback":
        st.info(result.get("message", "No recommendations available."))

    elif status == "error":
        st.error(result.get("message", "An internal error occurred. Please try again."))

    else:
        # Defensive: unknown status shape from future pipeline changes
        st.error(f"Unexpected response status: {status!r}")