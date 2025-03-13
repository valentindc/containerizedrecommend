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


# Any user should first have at it's disposal a dataset like those available at kaggle
# With a dataset for all movies available there I downloaded it, prepared it for some of the analysis I had in mind
# All of it can be seen in data_retriever.ipynb, or simply run and get the same dataset I used for training the recommender model.
# The model is very simple as it's sole purpose for the moment was to try and get it working from dataset to the recommender app.
# It was trained on an NVIDIA GPU, if CUDA doesn't find any it should defaul to the CPU and do it anyways (albeit it'll take some time)


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
2. Train the model in first instance and get `model.pkl`, it'll be a BIG file so, building it inside the docker app was brutally slow
3. Build the Docker image:
   ```
   docker build -t movie-recommender .
   ```
4. Run the container:
   ```
   docker run -p 5000:5000 movie-recommender
   ```
5. Access the API at `http://localhost:5000`

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

There's some issues with inputs, they must be written just as found on the dataset.
Momentarily, using /movies and ctrl+f [movie] is the best way I've found
