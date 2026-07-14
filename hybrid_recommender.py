from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix


movies = pd.read_csv("movies.csv")      # columns: movieId,title,genres
ratings = pd.read_csv("ratings.csv")    # columns: userId,movieId,rating,timestamp

# Keep only movies that actually appear in ratings (improves CF quality)
movies_in_ratings = movies[movies["movieId"].isin(ratings["movieId"].unique())].copy()
movies_in_ratings.reset_index(drop=True, inplace=True)

# Index maps
mid2idx = {mid: i for i, mid in enumerate(movies_in_ratings["movieId"].tolist())}
idx2mid = {i: mid for mid, i in mid2idx.items()}

# Popularity prior (Bayesian average)
rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).rename(
    columns={"mean": "avg", "count": "n"}
)
global_mean = ratings["rating"].mean()
m = 50  # prior strength (20..100 reasonable)

rating_stats["bayes"] = (
    (rating_stats["n"] / (rating_stats["n"] + m)) * rating_stats["avg"]
    + (m / (rating_stats["n"] + m)) * global_mean
)

movies_in_ratings = movies_in_ratings.merge(
    rating_stats[["bayes", "n"]], left_on="movieId", right_index=True, how="left"
).fillna({"bayes": global_mean, "n": 0})

# CONTENT VECTORS (TF-IDF over title + genres)
def _clean_title(t: str) -> str:
    return (
        t.lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace(":", " ")
        .replace("-", " ")
    )

movies_in_ratings["text"] = (
    movies_in_ratings["title"].fillna("").map(_clean_title)
    + " "
    + movies_in_ratings["genres"].fillna("").str.replace("|", " ", regex=False).str.lower()
)

tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words="english")
X_text = tfidf.fit_transform(movies_in_ratings["text"])                  # (n_movies, vocab)
content_sim = cosine_similarity(X_text, dense_output=False)              # (n_movies, n_movies), sparse

# COLLABORATIVE VECTORS (movie × user mean-centered ratings)
# Map users to 0..U-1
unique_users = np.sort(ratings["userId"].unique())
uid2idx = {u: i for i, u in enumerate(unique_users)}

# Filter to movies we kept
r2 = ratings[ratings["movieId"].isin(mid2idx)].copy()

# Mean-center ratings per user
user_mean = r2.groupby("userId")["rating"].transform("mean")
r2["adj_rating"] = r2["rating"] - user_mean

rows = r2["movieId"].map(mid2idx).values  # movie index
cols = r2["userId"].map(uid2idx).values   # user index
data = r2["adj_rating"].values

n_movies = len(movies_in_ratings)
n_users = len(unique_users)
item_user = coo_matrix((data, (rows, cols)), shape=(n_movies, n_users)).tocsr()

# KNN over item-user vectors (cosine)
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(item_user)

# How many ratings each movie received (for reliability weighting)
rating_count = r2.groupby("movieId")["rating"].count().reindex(
    movies_in_ratings["movieId"]
).fillna(0).astype(int).to_numpy()

if rating_count.max() > 0:
    reliability = np.log1p(rating_count) / np.log1p(rating_count.max())
else:
    reliability = np.ones_like(rating_count, dtype=np.float32)

# Utilities
def _find_movie_index_by_title(query: str) -> int:
    """Find a movie row using contains-match on title; fallback to text similarity."""
    q = _clean_title(query)
    exact = movies_in_ratings[movies_in_ratings["title"].str.lower().str.contains(q, na=False)]
    if not exact.empty:
        return int(exact.index[0])
    # fallback: text similarity vs query string
    q_vec = tfidf.transform([q])
    sims = (q_vec @ X_text.T).toarray().ravel()
    return int(np.argmax(sims))

def _top_collab_neighbors(idx: int, n_neighbors: int = 100):
    """Return (neighbor_indices, similarities) using item-based CF."""
    # kneighbors returns cosine distances; similarity = 1 - distance
    distances, indices = knn.kneighbors(item_user[idx], n_neighbors=min(n_neighbors, n_movies))
    indices = indices.ravel()
    dists = distances.ravel()
    sims = 1.0 - dists
    # drop self if present
    mask = indices != idx
    return indices[mask], sims[mask]

# Public API 1: movie-to-movie hybrid
def recommend_hybrid_for_movie(
    title: str,
    k: int = 10,
    alpha: float = 0.5,           # blend: alpha*content + (1-alpha)*collab
    collab_neighbors: int = 150,  # how many CF neighbors to consider
    min_collab_weight: float = 0.0
) -> pd.DataFrame:
    """
    Recommend movies similar to a given title using a hybrid score.
    alpha in [0,1]: 1.0 => content-only, 0.0 => collaborative-only.
    """
    idx = _find_movie_index_by_title(title)

    # --- content component
    cvec = content_sim[idx].toarray().ravel()
    cvec[idx] = 0.0

    # --- collaborative component
    neigh_idx, neigh_sim = _top_collab_neighbors(idx, n_neighbors=collab_neighbors)
    collab = np.zeros(n_movies, dtype=np.float32)
    collab[neigh_idx] = neigh_sim * reliability[neigh_idx]
    collab[idx] = 0.0
    if min_collab_weight > 0:
        collab = np.maximum(collab, min_collab_weight)

    # --- blend
    score = alpha * cvec + (1.0 - alpha) * collab

    top = np.argsort(-score)[:k]
    out = movies_in_ratings.loc[top, ["movieId", "title", "genres"]].copy()
    out.insert(0, "score", np.round(score[top], 6))
    return out.reset_index(drop=True)

# Public API 2: mood + genres (no seed movie)

# Moods → keywords + hidden weights
_MOOD_MAP = {
    "Happy 😊":       {"keywords": "fun feelgood comedy light heartwarming",             "w_text": 0.65, "w_pop": 0.35},
    "Sad 😢":         {"keywords": "emotional tragic heartbreaking drama tearjerker",    "w_text": 0.70, "w_pop": 0.30},
    "Romantic 💕":    {"keywords": "romance love relationship heartwarming",             "w_text": 0.70, "w_pop": 0.30},
    "Adventurous 🧭": {"keywords": "adventure journey epic quest action",                "w_text": 0.60, "w_pop": 0.40},
    "Chill 😌":       {"keywords": "calm cozy relaxing slice of life gentle",            "w_text": 0.75, "w_pop": 0.25},
    "Thrilling 😱":   {"keywords": "thriller suspense tense mystery crime",              "w_text": 0.65, "w_pop": 0.35},
    "Family 👨‍👩‍👧‍👦": {"keywords": "family friendly children animation wholesome",       "w_text": 0.65, "w_pop": 0.35},
}

# All MovieLens genres
ALL_GENRES = [
    "Action","Adventure","Animation","Children","Comedy","Crime","Documentary",
    "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance",
    "Sci-Fi","Thriller","War","Western","IMAX"
]

def recommend_by_mood_and_genres(
    mood: str,
    selected_genres: list[str] | None = None,
    k: int = 10
) -> pd.DataFrame:
    """
    Rank movies using a text prompt derived from mood + genres plus a popularity prior.
    No seed movie needed; ideal for a friendly UI.
    """
    if mood not in _MOOD_MAP:
        mood = "Happy 😊"
    selected_genres = selected_genres or []

    meta = _MOOD_MAP[mood]
    prompt = meta["keywords"] + " " + " ".join(g.lower() for g in selected_genres)

    # Text similarity against movie profiles
    q_vec = tfidf.transform([prompt])
    text_sim = (q_vec @ X_text.T).toarray().ravel()
    if text_sim.max() > 0:
        text_sim = text_sim / text_sim.max()

    # Popularity (Bayesian) normalized 0..1
    pop = movies_in_ratings["bayes"].to_numpy()
    pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)

    # Genre boost (fraction of selected genres present)
    if selected_genres:
        gset = [g.lower() for g in selected_genres]
        has = movies_in_ratings["genres"].str.lower()
        match_count = has.apply(lambda s: sum(g in s for g in gset)).to_numpy()
        genre_boost = match_count / len(gset)
    else:
        genre_boost = np.ones(len(movies_in_ratings))

    # Final hidden-blend score (users never see weights in UI)
    score = (meta["w_text"] * text_sim + meta["w_pop"] * pop) * (0.6 + 0.4 * genre_boost)

    # Top-k
    idx = np.argsort(-score)[:k]
    out = movies_in_ratings.loc[idx, ["movieId", "title", "genres", "n", "bayes"]].copy()
    out.insert(0, "score", np.round(score[idx], 6))
    out.rename(columns={"n": "rating_count", "bayes": "popularity"}, inplace=True)
    return out.reset_index(drop=True)

# Quick CLI demos
if __name__ == "__main__":
    print("\n--- Hybrid similar to 'Toy Story (1995)' ---")
    print(recommend_hybrid_for_movie("Toy Story (1995)", k=10, alpha=0.5))

    print("\n--- Mood-based: Happy + Family ---")
    print(recommend_by_mood_and_genres("Happy 😊", ["Family", "Animation"], k=10))
