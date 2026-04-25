import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("movies.csv")

# Combine features
data["combined"] = data["genre"] + " " + data["description"]

# Convert text to vectors
cv = CountVectorizer()
matrix = cv.fit_transform(data["combined"])

# Compute similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()
    
    if movie_name not in data["title"].str.lower().values:
        print("Movie not found!")
        return
    
    index = data[data["title"].str.lower() == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    
    # Sort by similarity
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    print(f"\nRecommended movies for '{movie_name}':\n")
    
    for i in sorted_scores[1:4]:
        print(data.iloc[i[0]]["title"])

# User input
movie = input("Enter a movie name: ")
recommend(movie)