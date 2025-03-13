from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import random
import re

app = Flask(__name__)

# Global variables to store our model data
indices = None
cosine_similarities = None
movie_title = None
title_lookup = None  # New dictionary for fuzzy matching

# Load the precomputed model
def load_model():
    global indices, cosine_similarities, movie_title, title_lookup
    
    print("Loading precomputed model...")
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract model components from the loaded pickle file
    indices = model_data['indices']
    cosine_similarities = model_data['cosine_similarities']
    movie_title = model_data['movie_title']
    
    # Create a lookup dictionary for fuzzy matching
    # This maps simplified titles (lowercase, no punctuation) to original titles
    title_lookup = {}
    for title in movie_title:
        # Create a simplified version of each title for easier matching
        simplified = simplify_title(title)
        title_lookup[simplified] = title
    
    print("Model loaded successfully!")

# Function to simplify titles for matching
def simplify_title(title):
    """
    Transforms a movie title into a simplified form for easier matching.
    
    Transformations:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Remove extra spaces
    4. Trim leading/trailing whitespace
    
    Args:
        title (str): The original movie title
        
    Returns:
        str: Simplified version of the title
    """
    # Convert to lowercase
    simplified = title.lower()
    # Remove punctuation and special characters
    simplified = re.sub(r'[^\w\s]', '', simplified)
    # Replace multiple spaces with a single space
    simplified = re.sub(r'\s+', ' ', simplified)
    # Remove leading/trailing whitespace
    simplified = simplified.strip()
    return simplified

# Function to find the best match for a title
def find_best_match(user_input):
    """
    Attempts to find the best matching movie title for a user input.
    
    Matching algorithm:
    1. First tries exact match on simplified titles
    2. If no exact match, tries partial matching (input in title or title in input)
    3. Sorts potential matches by title length (shorter is usually better)
    
    Args:
        user_input (str): The user's search query
        
    Returns:
        str: The matched original movie title, or None if no match found
    """
    # Simplify the user input for matching
    simplified_input = simplify_title(user_input)
    
    # Try exact match on simplified titles first
    if simplified_input in title_lookup:
        return title_lookup[simplified_input]
    
    # If no exact match, try partial matching
    potential_matches = []
    for simple_title, original_title in title_lookup.items():
        # Check if input is contained in a title (e.g., "star wars" in "star wars: episode iv")
        if simplified_input in simple_title:
            potential_matches.append((original_title, len(simple_title)))
        # Check if title is contained in input (e.g., "alien" in "alien movies")
        elif simple_title in simplified_input:
            potential_matches.append((original_title, len(simple_title)))
    
    # Sort by length of title (shorter matches are likely more relevant)
    # This helps prioritize "Star Wars" over "Star Wars: Episode IV - A New Hope"
    # when both match the input
    if potential_matches:
        potential_matches.sort(key=lambda x: x[1])
        return potential_matches[0][0]
    
    # Return None if no match found
    return None

# Content-based recommender function
def content_recommender(title, num_recommendations=10):
    """
    Generates movie recommendations based on content similarity.
    
    Args:
        title (str): The exact movie title to base recommendations on
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        list: List of recommended movie titles
    """
    # Get the index of the movie
    idx = indices[title]
    # Get the similarity scores for all movies
    sim_scores = list(enumerate(cosine_similarities[idx]))
    # Sort movies by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the input movie itself and get the top N
    sim_scores = sim_scores[1:num_recommendations+1]
    # Extract just the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the titles of the recommended movies
    return movie_title.iloc[movie_indices].tolist()

@app.route('/')
def index():
    """
    Root endpoint that displays basic API documentation.
    """
    return """
    <h1>Movie Recommendation API</h1>
    <p>Use the following endpoints:</p>
    <ul>
        <li><code>/recommend?title=MOVIE_TITLE</code> - Get recommendations based on a movie title</li>
        <li><code>/random</code> - Get a random movie recommendation</li>
        <li><code>/movies</code> - List all available movie titles</li>
    </ul>
    """

@app.route('/recommend')
def recommend():
    """
    Endpoint to get movie recommendations based on a title.
    
    Query parameters:
        title: The movie title to get recommendations for (can be approximate)
        
    Returns:
        JSON: Contains the input query, matched movie, and recommendations
    """
    # Get the title from the query parameters
    title = request.args.get('title', '')
    
    # Try to find a matching title using our fuzzy matching function
    matched_title = find_best_match(title)
    
    # Check if a matching movie was found
    if not matched_title:
        # Return helpful error with suggestions if no match found
        return jsonify({
            'error': 'Movie not found',
            'input': title,
            # Provide some potential matches that contain parts of the input
            'suggested_matches': [t for t in movie_title.tolist()[:5] if title.lower() in t.lower()],
            # Show a sample of available movies
            'available_movies': movie_title.tolist()[:10] + ['...']
        }), 404
    
    # Get recommendations based on the matched title
    recommendations = content_recommender(matched_title)
    
    # Return the results
    return jsonify({
        'input': title,  # Original user input
        'matched_movie': matched_title,  # What we found in the database
        'recommendations': recommendations  # Recommended movies
    })

@app.route('/random')
def random_movie():
    """
    Endpoint to get recommendations based on a random movie.
    
    Returns:
        JSON: Contains a random movie and recommendations for it
    """
    # Pick a random movie from the dataset
    random_index = random.randint(0, len(movie_title) - 1)
    random_title = movie_title.iloc[random_index]
    
    # Get recommendations based on the random movie
    recommendations = content_recommender(random_title)
    
    # Return the results
    return jsonify({
        'movie': random_title,
        'recommendations': recommendations
    })

@app.route('/movies')
def list_movies():
    """
    Endpoint to list all available movies in the dataset.
    
    Returns:
        JSON: Contains a list of all movie titles
    """
    # Return all movie titles
    return jsonify({
        'movies': movie_title.tolist()
    })

if __name__ == '__main__':
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False)
    
# Load model at startup
# This needs to be after the app definition but before the app runs
load_model()


