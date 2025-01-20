import faiss
import pickle
import numpy as np

# Step 1: Load embeddings and original data
with open('question_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('data_questions.pkl', 'rb') as f:
    data = pickle.load(f)

# Step 2: Convert embeddings to a numpy array
embeddings_np = np.array(embeddings).astype(np.float32)

# Step 3: Create a FAISS index
dim = embeddings_np.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(dim)  # Use L2 distance for similarity search
index.add(embeddings_np)  # Add the embeddings to the index

# Step 4: Save the FAISS index for later use
faiss.write_index(index, 'faiss_index.index')

print("FAISS index created and saved.")
