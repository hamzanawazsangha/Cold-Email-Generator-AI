import streamlit as st
import chromadb
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import torch
import pyperclip  # For clipboard copy
from transformers import AutoModel, AutoTokenizer

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load MiniLM model and tokenizer from local folder
@st.cache_resource
def load_model():
    model_path = os.path.abspath("./all-MiniLM-L6-v2")  # Convert to absolute path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize ChromaDB client
collection = None
try:
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        os.makedirs(db_path)  # Ensure folder exists for persistence

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="portfolio")

except Exception as e:
    st.warning(f"Could not connect to ChromaDB: {e}")

# Function to get embeddings using MiniLM
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()  # Mean pooling, move to CPU
    return embeddings

# Function to query ChromaDB
def query_chromadb(query_text):
    if collection is None:
        return "ChromaDB is unavailable."

    try:
        query_embedding = get_embedding(query_text)
        results = collection.query(query_embeddings=[query_embedding], n_results=1)
        if results.get("documents") and results["documents"][0]:
            return results["documents"][0][0]  # Extract first document
        return "No relevant data found."
    except Exception as e:
        return f"Error querying ChromaDB: {e}"

# Function to generate email using LangChain
def generate_email(job_desc, candidate_details):
    if not HUGGINGFACE_API_TOKEN:
        return "Missing Hugging Face API token. Please set it in your environment variables."

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
        The email should be well-structured, engaging, and include candidate details at the bottom.
        """
    )

    try:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={"temperature": 0.7},
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
        )
        chain = LLMChain(llm=llm, prompt=template)
        response = chain.run(job_desc=job_desc, **candidate_details)
        return response
    except Exception as e:
        return f"Error generating email: {e}"

# Streamlit UI
def main():
    st.title("AI-Powered Cold Email Generator ✉️")

    job_description = st.text_area("Enter the Job Description:", key="job_desc")

    st.subheader("Candidate Details")
    name = st.text_input("Full Name", key="name")
    email = st.text_input("Email", key="email")
    phone = st.text_input("Phone Number", key="phone")
    address = st.text_input("Address", key="address")
    linkedin = st.text_input("LinkedIn (Optional)", key="linkedin")
    github = st.text_input("GitHub (Optional)", key="github")
    education = st.text_area("Education Details", key="education")
    experience = st.text_area("Work Experience", key="experience")
    skills = st.text_area("Key Skills", key="skills")

    candidate_details = {
        "name": name, "email": email, "phone": phone, "address": address,
        "linkedin": linkedin, "github": github, "education": education,
        "experience": experience, "skills": skills,
    }

    email_content = ""

    if st.button("Generate Email", key="generate_email_btn"):
        if all([job_description, name, email, phone, education, experience, skills]):
            with st.spinner("Generating email... ⏳"):
                email_content = generate_email(job_description, candidate_details)
            st.subheader("Generated Email:")
            st.text_area("", email_content, height=200, key="email_output")
        else:
            st.warning("⚠️ Please fill in all required fields.")

    if email_content:
        if st.button("Copy to Clipboard", key="copy_btn"):
            pyperclip.copy(email_content)
            st.success("Copied to clipboard! ✅")

if __name__ == "__main__":
    main()
