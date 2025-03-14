import streamlit as st
import sys
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import LLMChain
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

# Load Mistral-7B model and tokenizer from local directory
@st.cache_resource
def load_mistral_model():
    model_path = "./models\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"  # Change to your local model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

# Function to generate email using locally stored model
def generate_email(job_desc, candidate_details):
    model, tokenizer = load_mistral_model()
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

    prompt = template.format(job_desc=job_desc, **candidate_details)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300, temperature=0.3)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

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
