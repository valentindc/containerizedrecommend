# Movie Recommender System

A Docker-containerized Flask API that provides movie recommendations based on content similarity.

## Features

- Content-based movie recommendations
- API endpoints for specific movie recommendations
- Random movie recommendations
- List of all available movies

## Prerequisites

- Docker installed on your system
- The movie dataset file (`final_dataset.csv`)

## Setup Instructions

1. Make sure your dataset file `final_dataset.csv` is in the project directory
2. Build the Docker image:
   ```
   docker build -t movie-recommender .
   ```
3. Run the container:
   ```
   docker run -p 5000:5000 movie-recommender
   ```
4. Access the API at `http://localhost:5000`

## API Endpoints

- `GET /` - Home page with usage instructions
- `GET /recommend?title=MOVIE_TITLE` - Get recommendations for a specific movie
- `GET /random` - Get recommendations based on a random movie
- `GET /movies` - List all available movies

## Example Usage

```bash
# Get recommendations for a specific movie
curl "http://localhost:5000/recommend?title=Psycho"

# Get recommendations based on a random movie
curl "http://localhost:5000/random"
```

## Built using:
- Numpy 2.0.2 Sensitive because of the .pkl format of the model save!!
- scikit-learn             1.6.1
- Werkzeug                 3.1.3
- flask                    3.1.0
- torch                    2.6.0+cu118
- torchaudio               2.6.0+cu118
- torchvision              0.21.0+cu118
- tornado                  6.4.2
- tqdm                     4.67.1


