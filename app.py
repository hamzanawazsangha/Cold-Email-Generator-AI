import streamlit as st
import sys
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Override SQLite3 for ChromaDB compatibility on Streamlit Cloud
os.environ["PYTHON_SQLITE_LIBRARY"] = "pysqlite3"

try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    st.warning("pysqlite3 is not installed. ChromaDB may not work.")

import chromadb

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Using persistent storage
    collection = chroma_client.get_or_create_collection(name="portfolio")
except Exception as e:
    st.warning(f"ChromaDB is not available: {e}")
    collection = None

# Load MiniLM model and tokenizer only when needed (Lazy Loading)
@st.cache_resource
def load_minilm_model():
    model_path = "./all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return model, tokenizer

# Function to get embeddings using MiniLM
def get_embedding(text):
    model, tokenizer = load_minilm_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling
    return embeddings

# Function to query ChromaDB
def query_chromadb(query_text):
    if not collection:
        return "ChromaDB is not available."
    
    try:
        query_embedding = get_embedding(query_text)
        results = collection.query(query_embeddings=[query_embedding], n_results=1)
        if results.get("documents"):
            return results["documents"][0]
        return "No relevant data found."
    except Exception as e:
        return f"Error querying ChromaDB: {e}"

# Function to generate email using LangChain
def generate_email(job_desc, candidate_details):
    template = PromptTemplate.from_template(
        """
        Write a professional and personalized cold email for a job application.
        Use the following candidate details:
        Name: {name}
        Email: {email}
        Phone: {phone}
        Address: {address}
        LinkedIn: {linkedin}
        GitHub: {github}
        Education: {education}
        Experience: {experience}
        Skills: {skills}
        
        Job Description: {job_desc}
        The email should be well-structured, engaging, and should include candidate details at the bottom of the email in a professional manner.
        """
    )

    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        return "Hugging Face API token is missing. Set HUGGINGFACEHUB_API_TOKEN in your environment variables."
    
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-small",  # Switched to a smaller model
            model_kwargs={"temperature": 0.3, "max_length": 200},  # Further reduced temperature and max length
            huggingfacehub_api_token=api_token
        )
        chain = LLMChain(llm=llm, prompt=template)
        response = chain.run(job_desc=job_desc, **candidate_details)
        return response
    except Exception as e:
        return f"Error generating email: {e}"

# Streamlit UI
def main():
    st.title("AI-Powered Cold Email Generator")
    
    job_description = st.text_area("Enter the Job Description:")
    
    st.subheader("Candidate Details")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    address = st.text_input("Address")
    linkedin = st.text_input("LinkedIn (Optional)")
    github = st.text_input("GitHub (Optional)")
    education = st.text_area("Education Details")
    experience = st.text_area("Work Experience")
    skills = st.text_area("Key Skills")
    
    candidate_details = {
        "name": name,
        "email": email,
        "phone": phone,
        "address": address,
        "linkedin": linkedin,
        "github": github,
        "education": education,
        "experience": experience,
        "skills": skills,
    }
    
    if st.button("Generate Email"):
        if job_description and name and email and phone and education and experience and skills:
            email_content = generate_email(job_description, candidate_details)
            st.subheader("Generated Email:")
            st.write(email_content)
        else:
            st.warning("Please fill in all required fields (Job Description, Name, Email, Phone, Education, Experience, and Skills).")

if __name__ == "__main__":
    main()
