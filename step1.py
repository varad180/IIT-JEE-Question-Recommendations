import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import faiss

# Step 1: Load the Dataset
file_path = 'questions_dataset.csv'  # Update with actual dataset path
data = pd.read_csv(file_path, names=['Question', 'Options', 'Answer', 'Subject'], header=None)

# Step 2: Preprocess the Data
def preprocess(data):
    # Lowercase all questions for uniformity
    data['Question'] = data['Question'].str.lower()
    return data

data = preprocess(data)

# Step 3: Generate Sentence Embeddings using Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['Question'].tolist(), show_progress_bar=True)

# Step 4: Save the embeddings and model for later use
with open('question_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

with open('data_questions.pkl', 'wb') as f:
    pickle.dump(data[['Question', 'Options', 'Subject']], f)

print("Embeddings and questions saved.")
