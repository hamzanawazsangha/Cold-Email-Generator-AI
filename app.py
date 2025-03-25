import streamlit as st
import chromadb
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModel, AutoTokenizer

# Load environment variables
load_dotenv()

# Improved model loading with error handling
@st.cache_resource
def load_models():
    try:
        model_path = "./all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

tokenizer, model = load_models()

# Initialize ChromaDB client with persistent storage
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="portfolio",
        metadata={"hnsw:space": "cosmo"}  # Optimized for cosine similarity
    )
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

# Improved embedding function with caching
@st.cache(show_spinner=False)
def get_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")
        return None

# Enhanced query function with error handling
def query_chromadb(query_text):
    try:
        query_embedding = get_embedding(query_text)
        if not query_embedding:
            return "Error generating embeddings"
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "distances"]
        )
        return results["documents"][0][0] if results["documents"] else "No relevant data found."
    except Exception as e:
        return f"Query error: {e}"

# Enhanced email generation with safety checks
def generate_email(job_desc, candidate_details):
    required_fields = ['name', 'email', 'phone', 'education', 'experience', 'skills']
    if any(not candidate_details.get(field) for field in required_fields):
        return "Missing required candidate details"

    template = PromptTemplate.from_template("""
        Generate a professional cold email for a job application using these details:
        Applicant: {name}
        Contact: {email} | {phone} | {address}
        Links: {linkedin} | {github}
        Education: {education}
        Experience: {experience}
        Skills: {skills}

        Job Description: {job_desc}

        Structure:
        - Personalized greeting
        - Brief introduction
        - Key qualifications matching job requirements
        - Specific examples of relevant achievements
        - Polite call to action
        - Professional signature with contact info
        Keep it under 300 words.
    """)

    try:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 500,
                "repetition_penalty": 1.2
            },
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        chain = LLMChain(llm=llm, prompt=template)
        return chain.run(job_desc=job_desc, **candidate_details)
    except Exception as e:
        return f"Generation error: {e}"

# Streamlit UI with improved layout and validation
def main():
    st.set_page_config(page_title="AI Email Generator", layout="wide")
    
    with st.sidebar:
        st.header("Configuration")
        st.info("Ensure your Hugging Face API token is set in environment variables")
    
    st.title("📧 AI-Powered Job Application Email Generator")
    st.markdown("---")

    with st.expander("🔍 Job Description", expanded=True):
        job_description = st.text_area("Paste the job description here:", height=150)

    with st.expander("👤 Candidate Details"):
        cols = st.columns(2)
        with cols[0]:
            name = st.text_input("Full Name*")
            email = st.text_input("Email*")
            phone = st.text_input("Phone*")
            address = st.text_input("Address")
        with cols[1]:
            linkedin = st.text_input("LinkedIn URL")
            github = st.text_input("GitHub URL")
            education = st.text_area("Education*")
            experience = st.text_area("Experience*")
            skills = st.text_area("Skills*")

    candidate_details = {
        "name": name, "email": email, "phone": phone, "address": address,
        "linkedin": linkedin, "github": github, "education": education,
        "experience": experience, "skills": skills
    }

    if st.button("✨ Generate Email", type="primary"):
        if not job_description:
            st.warning("Please enter a job description")
        elif any(not candidate_details[field] for field in ['name', 'email', 'phone', 'education', 'experience', 'skills']):
            st.warning("Please fill all required fields (*)")
        else:
            with st.spinner("Generating email..."):
                email_content = generate_email(job_description, candidate_details)
                
            st.markdown("---")
            st.subheader("Generated Email")
            with st.container(border=True):
                st.markdown(email_content)
            
            st.download_button(
                label="📥 Download Email",
                data=email_content,
                file_name="generated_email.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
