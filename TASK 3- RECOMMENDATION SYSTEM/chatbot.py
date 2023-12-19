import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pd.read_csv('movies.csv')

print(movies.head())

# Get user preferences input
user_preferences = input("Enter your movie preferences (e.g., Action Romance): ")

movies['features'] = movies['genres']

vectorizer = TfidfVectorizer(stop_words='english')
features_matrix = vectorizer.fit_transform(movies['features'])

user_preferences_matrix = vectorizer.transform([user_preferences])

cosine_similarities = linear_kernel(user_preferences_matrix, features_matrix).flatten()

movies['similarity'] = cosine_similarities

recommended_movies = movies.sort_values(by='similarity', ascending=False)

print("\nTop Movie Recommendations:")
for index, row in recommended_movies.head(3).iterrows():
    print(f"{row['title']} - Similarity Score: {row['similarity']:.2f}")
