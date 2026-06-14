"""Interactive first-booking destination ranker.

Ranks the destinations a new Airbnb user is most likely to book first, to
personalize onboarding. Trained on a schema-faithful synthetic stand-in because the
competition data is consent-gated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="First Booking Destination", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.grid(True, axis="x", color=D["current_line"], linestyle="--", alpha=0.4)


def ranking_chart(ranked):
    dests = [d for d, _ in ranked][::-1]
    probs = [p * 100 for _, p in ranked][::-1]
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=D["background"])
    ax.barh(dests, probs, color=D["purple"], edgecolor=D["current_line"])
    for i, p in enumerate(probs):
        ax.text(p, i, f"  {p:.1f}%", va="center", color=D["foreground"], fontsize=9)
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, max(probs) * 1.2)
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    st.title("Airbnb First Booking — Destination Ranker")
    st.markdown(
        "Ranks where a new user is most likely to book first, to personalize onboarding "
        "(content, recommendations, language). NDF means no booking in the observation window."
    )
    if predictor.synthetic:
        st.caption(
            "Note: the Airbnb competition data is consent-gated, so this model is trained on a "
            "schema-faithful synthetic stand-in. Drop the real train_users_2.csv into data/raw "
            "and re-run the pipeline for genuine results.")

    with st.sidebar:
        st.header("New user")
        gender = st.selectbox("Gender", ["FEMALE", "MALE", "OTHER", "-unknown-"])
        age = st.slider("Age", 18, 90, 34)
        signup_method = st.selectbox("Signup method", ["basic", "facebook", "google"])
        signup_app = st.selectbox("Signup app", ["Web", "iOS", "Moweb", "Android"])
        language = st.selectbox("Language", ["en", "zh", "fr", "es", "de", "it", "pt", "nl"])
        device = st.selectbox("First device", ["Mac Desktop", "Windows Desktop", "iPhone", "iPad",
                                               "Android Phone", "Other/Unknown"])
        channel = st.selectbox("Affiliate channel", ["direct", "sem-brand", "sem-non-brand", "seo", "other"])
        run = st.button("Rank destinations", type="primary")

    user = {"gender": gender, "age": age, "signup_method": signup_method, "signup_flow": 0,
            "language": language, "affiliate_channel": channel, "affiliate_provider": "direct",
            "first_affiliate_tracked": "untracked", "signup_app": signup_app,
            "first_device_type": device, "first_browser": "Chrome",
            "date_account_created": "2014-06-01", "timestamp_first_active": "20140531120000"}

    if run:
        ranked = predictor.rank(user, k=5)
        st.subheader("Top destinations")
        c = st.columns(2)
        c[0].metric("Most likely", ranked[0][0], f"{ranked[0][1]*100:.0f}%")
        booked = [(d, p) for d, p in ranked if d != "NDF"]
        if booked:
            c[1].metric("Most likely booking", booked[0][0], f"{booked[0][1]*100:.0f}%")
        left, right = st.columns(2)
        with left:
            st.pyplot(ranking_chart(ranked))
        with right:
            st.markdown("**Most influential features (model-wide)**")
            imp = pd.DataFrame(predictor.top_features(6)).rename(
                columns={"feature": "Feature", "importance": "Permutation importance (log-loss)"})
            st.dataframe(imp, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
