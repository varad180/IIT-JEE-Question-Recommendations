import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Step 1: Load pre-trained model, FAISS index, and original data
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('data_questions.pkl', 'rb') as f:
    data = pickle.load(f)

index = faiss.read_index('faiss_index.index')

# Step 2: Define recommendation function
def recommend_questions(query, top_n=10):
    query = query.lower()  # Ensure uniformity
    query_embedding = model.encode([query])[0]  # Get the embedding for the query
    query_embedding = np.array(query_embedding).astype(np.float32)

    # Search for the top N most similar questions using FAISS
    distances, indices = index.search(query_embedding.reshape(1, -1), top_n)
    
    recommendations = []
    for i in range(top_n):
        idx = indices[0][i]
        recommendations.append(data.iloc[idx])  # Get the corresponding question details

    return recommendations

# Step 3: Streamlit UI
st.title("Question Recommendation System")
st.write("Enter your query and get recommendations for important questions!")

# User input
query = st.text_input("Enter your query:", "")
top_n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)  # Increased the max value to 20

if st.button("Get Recommendations"):
    if query:
        recommendations = recommend_questions(query, top_n=top_n)
        st.write("## Recommended Questions:")
        
        # Display recommendations with numbering
        for idx, row in enumerate(recommendations, start=1):  # `start=1` for numbering starting from 1
            st.write(f"**{idx}. Question**: {row['Question']}")
            
            st.write("\n")
    else:
        st.warning("Please enter a query to get recommendations.")
