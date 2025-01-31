import streamlit as st
import pdfplumber
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import cohere  # Import Cohere library

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Cohere client
co = cohere.Client("oGY911s4SFviCFwKqUpKBI7IEEc1ZVQ41SzNiwWH")  # Replace with your Cohere API key

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text

# Function to chunk text
def chunk_text(text, chunk_size=200):  # Smaller chunk size
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=3):  # Fewer chunks
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Function to truncate context
def truncate_context(context, max_tokens=3000):
    words = context.split()
    truncated_context = " ".join(words[:max_tokens])
    return truncated_context

# Function to generate answers using Cohere
def generate_answer(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    truncated_context = truncate_context(context)  # Truncate the context
    prompt = f"Query: {query}\nContext: {truncated_context}\nAnswer:"
    
    # Generate response using Cohere
    response = co.generate(
        model="command",  # Use Cohere's "command" model
        prompt=prompt,
        max_tokens=200,  # Adjust as needed
        temperature=0.7,  # Adjust for creativity
        stop_sequences=["\n"],  # Stop generation at newlines
        truncate="END"  # Enable prompt truncation
    )
    
    return response.generations[0].text

# Streamlit app
st.title("RAG Application for Research Papers")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
if uploaded_file:
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)
    
    # Clean and chunk the text
    chunks = chunk_text(clean_text(extracted_text))
    
    # Generate embeddings for the chunks
    embeddings = model.encode(chunks)
    
    # Create a FAISS index and add embeddings
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Accept user query
    query = st.text_input("Enter your query:")
    if query:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(query)
        
        # Generate an answer using Cohere
        answer = generate_answer(query, relevant_chunks)
        
        # Display the answer and relevant sections
        st.write("Answer:", answer)
        st.write("Relevant Sections:")
        for chunk in relevant_chunks:
            st.write(chunk)