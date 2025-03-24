import streamlit as st
import chromadb
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from typing import Dict, Any

# Constants
CHROMA_DB_PATH = "/tmp/chroma_db"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource(show_spinner="Loading GPT-2 model...")
def load_llm_model():
    """Load and cache the GPT-2 model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,  # Keeps emails concise
            temperature=0.7,
            top_p=0.9,
            device="cpu"  # Render-compatible
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        st.session_state.model_loaded = True
        return llm
    except Exception as e:
        st.error(f"❌ Error loading GPT-2 model: {e}")
        st.stop()

@st.cache_resource(show_spinner="Initializing ChromaDB...")
def initialize_chromadb():
    """Initialize ChromaDB with default embeddings"""
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name="portfolio")
        return collection
    except Exception as e:
        st.warning(f"⚠️ ChromaDB initialized with default embeddings: {e}")
        return None

# Load models
llm = load_llm_model()
collection = initialize_chromadb()

def query_chromadb(query_text: str) -> str:
    """Query ChromaDB (using default embeddings)"""
    if collection is None:
        return "⚠️ ChromaDB is not available."
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=1
        )
        if results and results.get("documents"):
            return results["documents"][0][0]
        return "❌ No matching data found."
    except Exception as e:
        return f"⚠️ Query error: {e}"

def generate_email(job_desc: str, candidate_details: Dict[str, Any]) -> str:
    """Generate email using GPT-2"""
    template = """
    Write a professional cold email for this job application:
    
    **Candidate:**
    - Name: {name}
    - Education: {education}
    - Experience: {experience}
    - Key Skills: {skills}
    
    **Job Description:**
    {job_desc}
    
    **Requirements:**
    1. Address the hiring manager professionally
    2. Highlight 2-3 most relevant skills
    3. Show enthusiasm for the role
    4. Keep it under 200 words
    5. End with: "Best regards, {name}"
    """
    
    prompt = PromptTemplate.from_template(template)
    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(job_desc=job_desc, **candidate_details)
        return response.strip()
    except Exception as e:
        return f"⚠️ Generation failed: {e}"

def validate_inputs(job_desc: str, details: Dict[str, Any]) -> bool:
    """Check required fields"""
    required = [
        job_desc,
        details.get("name"),
        details.get("education"),
        details.get("experience"),
        details.get("skills")
    ]
    return all(required)

def main():
    st.set_page_config(
        page_title="📧 Cold Email Generator",
        page_icon="✉️",
        layout="centered"
    )
    
    st.title("📧 AI Cold Email Generator")
    st.caption("Powered by GPT-2 (Local) + ChromaDB")
    
    # Job Description
    job_desc = st.text_area(
        "Paste the Job Description:",
        height=150,
        placeholder="Example: 'We seek a Python developer with 2+ years of experience...'"
    )
    
    # Candidate Details
    with st.expander("🧑‍💼 Your Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*")
            email = st.text_input("Email*")
            phone = st.text_input("Phone")
        with col2:
            linkedin = st.text_input("LinkedIn")
            github = st.text_input("GitHub")
        
        education = st.text_area("Education*", placeholder="Degree, University, Year")
        experience = st.text_area("Experience*", placeholder="Previous roles and achievements")
        skills = st.text_area("Skills*", placeholder="Python, SQL, Project Management...")
    
    candidate_details = {
        "name": name,
        "email": email,
        "phone": phone or "Not provided",
        "linkedin": linkedin or "Not provided",
        "github": github or "Not provided",
        "education": education,
        "experience": experience,
        "skills": skills,
    }
    
    # Generation Button
    if st.button("✨ Generate Email", type="primary"):
        if validate_inputs(job_desc, candidate_details):
            with st.spinner("Crafting your email..."):
                email_content = generate_email(job_desc, candidate_details)
            
            st.subheader("📩 Your Email Draft")
            st.markdown("---")
            st.write(email_content)
            st.markdown("---")
            
            st.download_button(
                "💾 Download Email",
                data=email_content,
                file_name=f"Job_Application_{name.replace(' ', '_')}.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please fill all required fields (*)")

if __name__ == "__main__":
    main()
