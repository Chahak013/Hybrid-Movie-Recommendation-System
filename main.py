import pandas as pd

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("Movies Data:")
print(movies.head())

print("\nRatings Data:")
print(ratings.head())
