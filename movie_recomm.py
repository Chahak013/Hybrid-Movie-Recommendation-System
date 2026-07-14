import random
import textwrap
import urllib.parse

import pandas as pd
import streamlit as st

import hybrid_recommender as hr

# Page setup
st.set_page_config(page_title="Movie Mood Matcher", layout="wide")
if "cache_cleared" not in st.session_state:
    st.cache_data.clear()
    st.session_state["cache_cleared"] = True

# Styles (modern, poster-less cards)
st.markdown("""
<style>
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 2.2rem; }
h1, h2, h3 { letter-spacing:.2px; }

/* Controls */
.stButton > button {
  border-radius: 12px; padding:.65rem 1rem; font-weight:700;
  background: linear-gradient(180deg,#ff7a84,#ff4e5e); border:none; color:#fff;
}
.stButton > button:hover { filter: brightness(1.06); }
.btn-ghost > button {
  background: rgba(255,255,255,0.06) !important; border: 1px solid #2a323d !important;
  color: #eef2f8 !important; font-weight:700 !important; border-radius:12px !important;
}

/* Card grid */
.row { display:grid; grid-template-columns: repeat(4, 1fr); gap: 26px; }
@media (max-width: 1160px){ .row{ grid-template-columns: repeat(3, 1fr);} }
@media (max-width: 860px){ .row{ grid-template-columns: repeat(2, 1fr);} }
@media (max-width: 580px){ .row{ grid-template-columns: repeat(1, 1fr);} }

.card {
  background:#0f1318; border:1px solid #232b35; border-radius:16px; overflow:hidden;
  box-shadow: 0 10px 24px rgba(0,0,0,.35); transition: transform .18s ease, box-shadow .18s ease;
}
.card:hover{ transform: translateY(-6px); box-shadow: 0 16px 38px rgba(0,0,0,.45); }

/* Gradient header inside cards */
.hero {
  height: 220px; padding: 14px; display:flex; flex-direction:column; justify-content:flex-end;
  position:relative; border-bottom:1px solid #232b35;
}
.badge {
  position:absolute; left:14px; top:12px; font-size:.78rem; letter-spacing:.2px;
  background:#0d1117cc; color:#e9eef6; padding:.22rem .6rem; border-radius:10px;
  border:1px solid #2a3140;
}
.title {
  font-size: 1.12rem; font-weight:800; margin:0 0 .55rem 0; line-height:1.25; color:#f2f6fd;
  text-shadow: 0 1px 0 rgba(0,0,0,.4);
}
/* Chips inside hero bottom edge */
.hero-chips { display:flex; gap:.45rem; flex-wrap:wrap; }
.chip {
  display:inline-block; padding:.18rem .55rem; border-radius:999px; font-size:.74rem;
  background:#1b2330; border:1px solid #2b3442; color:#dbe4f1;
}

/* Body & footer */
.body { padding: 12px 14px 14px 14px; }
.foot { display:flex; align-items:center; justify-content:space-between; margin-top:.55rem; }
.score {
  font-size:1.05rem; font-weight:900; letter-spacing:.3px; color:#eaf2ff;
  background: linear-gradient(180deg,#1d2632,#111820); border:1px solid #2b3644;
  padding:.35rem .6rem; border-radius:10px;
}
.link-btn {
  text-decoration:none; font-weight:800; padding:.48rem .72rem; border-radius:10px;
  border:1px solid #2a3340; background:#16202c; color:#e9eef6;
}
.link-btn:hover { filter: brightness(1.12); }

.hr { height:1px; background:linear-gradient(90deg,#0000,#2a313a,#0000); margin:1.4rem 0; }
</style>
""", unsafe_allow_html=True)

# Helpers 
def youtube_trailer_url(title: str) -> str:
    q = urllib.parse.quote_plus(f"{title} official trailer")
    return f"https://www.youtube.com/results?search_query={q}"

def gradient_for_seed(seed: int) -> str:
    """
    Deterministic pleasant gradient based on a seed (movieId).
    Returns inline CSS style for background.
    """
    rnd = random.Random(seed)
    h1 = rnd.randint(200, 320)  # cool range
    h2 = (h1 + rnd.randint(12, 42)) % 360
    return (
        f"background: radial-gradient(120% 120% at 0% 0%, hsla({h1},70%,22%,1) 0%, "
        f"hsla({h2},65%,28%,1) 60%, #0e141b 110%);"
    )

st.title("🎬 Movie Recommendation System")
st.caption("Tell me your mood and the kind of film you’re after — I’ll curate a list you’ll love.")
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

left, right = st.columns([1.1, 1.3])
with left:
    st.subheader("How are you feeling today?")
    moods = list(hr._MOOD_MAP.keys())
    mood = st.radio("Select mood", moods, index=0)

with right:
    st.subheader("What kind of movie do you want?")
    selected_genres = st.multiselect("Pick one or more genres (optional)", hr.ALL_GENRES, default=[])

st.write("")
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    k = st.slider("How many recommendations?", 6, 24, 12, step=2)
with c2:
    go = st.button("Find movies 🍿", type="primary", width="stretch")
with c3:
    surprise = st.button("Surprise me 🎲", width="stretch", help="Pick a random mood & genres")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Results
if go or surprise:
    if surprise:
        mood = random.choice(moods)
        g = hr.ALL_GENRES.copy(); random.shuffle(g)
        selected_genres = g[:random.choice([0, 1, 2])]
        st.info(f"Surprise pick → Mood: **{mood}**  |  Genres: **{', '.join(selected_genres) or 'Any'}**")

    results = hr.recommend_by_mood_and_genres(mood, selected_genres, k=k)

    if results.empty:
        st.warning("No matches found. Try fewer filters.")
    else:
        st.subheader("Your curated picks")
        st.caption("Ranked by a mix of mood fit, genre match, and community popularity.")
        st.write("")

        # HTML for a responsive grid 
        cards = []
        for _, row in results.iterrows():
            title = row["title"]
            genres = [g for g in row["genres"].split("|") if g and g != "(no genres listed)"][:4]
            chips = "".join(f"<span class='chip'>{g}</span>" for g in genres)
            grad = gradient_for_seed(int(row["movieId"]))
            yt = youtube_trailer_url(title)

            card_html = textwrap.dedent(f"""
            <div class="card">
              <div class="hero" style="{grad}">
                <div class="badge">🎬 Movie</div>
                <div class="title">{title}</div>
                <div class="hero-chips">{chips}</div>
              </div>
              <div class="body">
                <div class="foot">
                  <div class="score">{row['score']:.3f}</div>
                  <a class="link-btn" href="{yt}" target="_blank" rel="noopener">▶ Watch trailer</a>
                </div>
              </div>
            </div>
            """)
            cards.append(card_html)

        html = "<div class='row'>" + "".join(cards) + "</div>"
        st.markdown(html, unsafe_allow_html=True)
