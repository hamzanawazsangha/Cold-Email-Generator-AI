import streamlit as st
import sys
import os
import torch
import asyncio
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import chromadb

# Ensure asyncio event loop compatibility
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Override SQLite3 for ChromaDB compatibility on Streamlit Cloud
os.environ["PYTHON_SQLITE_LIBRARY"] = "pysqlite3"

try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    st.warning("pysqlite3 is not installed. ChromaDB may not work.")

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
@st.cache_resource
def get_chromadb_client():
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        return client.get_or_create_collection(name="portfolio")
    except Exception as e:
        st.warning(f"ChromaDB is not available: {e}")
        return None

collection = get_chromadb_client()

# Lazy loading of MiniLM model and tokenizer
@st.cache_resource
def load_minilm_model():
    model_path = "./all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return model, tokenizer

# Function to get embeddings
def get_embedding(text):
    model, tokenizer = load_minilm_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Query ChromaDB
def query_chromadb(query_text):
    if not collection:
        return "ChromaDB is not available."
    try:
        query_embedding = get_embedding(query_text)
        results = collection.query(query_embeddings=[query_embedding], n_results=1)
        return results.get("documents", ["No relevant data found."])[0]
    except Exception as e:
        return f"Error querying ChromaDB: {e}"

# Generate email using LangChain
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
        """
    )

    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        return "Hugging Face API token is missing. Set HUGGINGFACEHUB_API_TOKEN in your environment variables."
    
    try:
        llm = HuggingFaceHub(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_kwargs={"temperature": 0.3, "max_length": 200},
            huggingfacehub_api_token=api_token
        )
        chain = LLMChain(llm=llm, prompt=template)
        return chain.run(job_desc=job_desc, **candidate_details)
    except Exception as e:
        return f"Error generating email: {e}"

# Streamlit UI
def main():
    st.title("AI-Powered Cold Email Generator")
    
    job_description = st.text_area("Enter the Job Description:")
    
    st.subheader("Candidate Details")
    candidate_details = {
        "name": st.text_input("Full Name"),
        "email": st.text_input("Email"),
        "phone": st.text_input("Phone Number"),
        "address": st.text_input("Address"),
        "linkedin": st.text_input("LinkedIn (Optional)"),
        "github": st.text_input("GitHub (Optional)"),
        "education": st.text_area("Education Details"),
        "experience": st.text_area("Work Experience"),
        "skills": st.text_area("Key Skills"),
    }
    
    if st.button("Generate Email"):
        required_fields = ["name", "email", "phone", "education", "experience", "skills"]
        if job_description and all(candidate_details[field] for field in required_fields):
            email_content = generate_email(job_description, candidate_details)
            st.subheader("Generated Email:")
            st.write(email_content)
        else:
            st.warning("Please fill in all required fields (Job Description, Name, Email, Phone, Education, Experience, and Skills).")

if __name__ == "__main__":
    main()
