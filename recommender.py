# precompute_model.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
from tqdm import tqdm  # Optional: for progress tracking



print("Loading data...")
data = pd.read_csv('final_dataset.csv')

print("Preprocessing data...")
data.dropna(subset=['genres', 'stars', 'writers', 'Languages', 
                    'production_companies', 'countries_origin'], 
            inplace=True, axis=0)
data = data.reset_index(drop=True)
data['combined'] = (data['genres'] + '  ' + data['stars'] + ' ' + 
                    data['writers'] + ' ' + data['Languages'] + ' ' + 
                    data['countries_origin'] + ' ' + data['Title'])

print("Creating TF-IDF matrix...")
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(data["combined"])
matrix_dense = matrix.toarray()  # Convert to numpy array

print("Computing cosine similarities with GPU acceleration...")
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert to PyTorch tensors and move to GPU
matrix_tensor = torch.tensor(matrix_dense, dtype=torch.float32).to(device)

# Calculate similarities in batches to avoid VRAM limitations
batch_size = 512  # Adjust based on your GPU memory
num_items = matrix_tensor.shape[0]
cosine_similarities = np.zeros((num_items, num_items))

for i in tqdm(range(0, num_items, batch_size)):
    end_idx = min(i + batch_size, num_items)
    batch = matrix_tensor[i:end_idx]
    
    # Compute dot product for this batch with all items
    dot_product = torch.mm(batch, matrix_tensor.t())
    
    # Compute norms
    batch_norm = torch.norm(batch, dim=1, keepdim=True)
    all_norm = torch.norm(matrix_tensor, dim=1, keepdim=True)
    
    # Compute cosine similarity
    similarities = dot_product / (batch_norm * all_norm.t())
    
    # Move result back to CPU and store
    cosine_similarities[i:end_idx] = similarities.cpu().numpy()

movie_title = data['Title']
indices = pd.Series(data.index, index=data['Title'])

# Save the model components
print("Saving model...")
model_data = {
    'indices': indices,
    'cosine_similarities': cosine_similarities,
    'movie_title': movie_title,
    'vectorizer': vectorizer
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully!")