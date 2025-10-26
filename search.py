from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import os

# Load the SentenceTransformer model
model = SentenceTransformer('clip-ViT-B-32')

image_files = [i for i in os.listdir('images')]

def search(query, top_k=3):
    index = load_index()
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)
    return [ image_files[i] for i in I[0] ]


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

        # Create FAISS index
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)

        # Save the FAISS index to a file
        faiss.write_index(index, index_file)

    return index

if __name__ == "__main__":

    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = search(query, top_k=3)
        print("Top 3 similar images:")
        for res in results:
            print(res)