# Suppress warnings and logging from Hugging Face and SentenceTransformers for cleaner output

import os
import warnings
import logging

# Hugging Face + Transformers verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# SentenceTransformers uses logging, not warnings
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Python warnings (e.g. fast/slow processor notice)
warnings.filterwarnings("ignore")

# Actual imports for the semantic search functionality
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import os
import numpy as np

# Load the SentenceTransformer model
model = SentenceTransformer('clip-ViT-B-32')

image_files = [i for i in os.listdir('images')]

# The actual search function that takes a query and returns the top_k most similar images based on "distance"
def search(query, top_k=3):
    index = load_index()
    query_emb = model.encode([query])

    # For a query term, find images "closest" to it using distance between query and each image
    query_emb = np.asarray(query_emb, dtype='float32')
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, top_k)

    # Return the top_k image file names corresponding to the closest matches
    return [ image_files[i] for i in I[0] ]


# Helper function to load or create the index
def load_index(index_file='faiss_index.bin', embedding_dim=512):
    # Check if the FAISS index file exists
    if os.path.exists(index_file):
        # Load the FAISS index from the file
        index = faiss.read_index(index_file)
    else:
        # Create RGB for each image in image_files
        imgs = [Image.open(f'images/{f}').convert('RGB') for f in image_files]

        # Encode the verses
        print("Creating embeddings...")
        embeddings = model.encode(imgs, show_progress_bar=True)

        # Ensure embeddings are float32 and normalized for cosine similarity
        embeddings = np.asarray(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings)

        # Determine embedding_dim from model output
        embedding_dim = embeddings.shape[1]

        # Create FAISS index using inner product (cosine via normalized vectors)
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)

        # Save the FAISS index to a file
        faiss.write_index(index, index_file)

    return index

# Very simple command-line interface to test the search functionality
if __name__ == "__main__":

    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = search(query, top_k=3)
        print("Top 3 similar images:")
        for res in results:
            print(res)