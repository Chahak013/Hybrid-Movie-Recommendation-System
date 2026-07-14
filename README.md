# 🎬 Hybrid Movie Recommendation System

A **Hybrid Movie Recommendation System** built using **Python**, **Scikit-Learn**, and **Streamlit** that recommends movies based on a user's **mood** and **genre preferences**. The system combines **Content-Based Filtering** and **Collaborative Filtering** to generate accurate and personalized recommendations.

---

## 📖 Overview

This project utilizes a hybrid recommendation approach by combining:

* **Content-Based Filtering** using TF-IDF Vectorization and Cosine Similarity
* **Collaborative Filtering** using Item-Based K-Nearest Neighbors (KNN)
* **Bayesian Rating Smoothing** to reduce popularity bias

The application provides an interactive Streamlit interface where users can select their mood and preferred genres to receive personalized movie recommendations along with direct trailer links.

---

## ✨ Features

* 🎭 Mood-Based Recommendations
* 🎬 Genre-Based Filtering
* 🤖 Hybrid Recommendation Engine
* 📊 Content-Based Filtering (TF-IDF + Cosine Similarity)
* 👥 Collaborative Filtering (KNN)
* ⭐ Bayesian Popularity Scoring
* 🎨 Interactive Streamlit UI
* ▶️ One-click YouTube Trailer Search
* 📱 Responsive Card-Based Interface

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Scikit-Learn
* Streamlit

---

## 📂 Dataset

This project uses the **MovieLens Dataset**.

Required files:

* `movies.csv`
* `ratings.csv`

---

## 📁 Project Structure

```text
Hybrid-Movie-Recommendation-System/
│
├── hybrid_recommender.py      # Recommendation Engine
├── movie_recomm.py            # Streamlit User Interface
├── movies.csv                 # Movie Dataset
├── ratings.csv                # User Ratings Dataset
├── requirements.txt           # Project Dependencies
├── README.md                  # Project Documentation
└── .gitignore
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Hybrid-Movie-Recommendation-System.git
cd Hybrid-Movie-Recommendation-System
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Application

```bash
streamlit run movie_recomm.py
```

---

## 5. Open the Application

If the browser does not open automatically, visit:

```
http://localhost:8501
```

---

# 🧠 How It Works

### 1. Content-Based Filtering

Movie titles and genres are converted into TF-IDF vectors.

Cosine Similarity is then used to identify movies with similar content.

---

### 2. Collaborative Filtering

The system analyzes user rating patterns using Item-Based K-Nearest Neighbors (KNN) to identify movies liked by users with similar preferences.

---

### 3. Hybrid Recommendation

Both recommendation methods are combined using a weighted scoring formula to improve recommendation quality.

```
Final Score = α × Content Similarity + (1 − α) × Collaborative Similarity
```

---

### 4. Mood-Based Recommendation

The selected mood is mapped to descriptive keywords.

Example:

* Happy → fun, comedy, feel-good
* Romantic → romance, love, relationship
* Thrilling → suspense, mystery, crime

These keywords are compared with movie content using TF-IDF similarity.

The similarity score is combined with a Bayesian popularity score to generate the final recommendations.

---

## 📷 Application Preview

Run the application to access:

* Mood Selection
* Genre Selection
* Personalized Movie Recommendations
* Recommendation Scores
* YouTube Trailer Links

---

## 📌 Future Enhancements

* TMDB API Integration for Posters
* User Login & Profiles
* Watchlist Support
* Recommendation Evaluation Metrics
* Deep Learning-based Recommendation Models
* Movie Search Autocomplete

---

## 👨‍💻 Author

**Chahak Jain**

---

## 📄 License

This project is intended for educational and learning purposes.