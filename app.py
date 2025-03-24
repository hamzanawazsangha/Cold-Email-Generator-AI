import streamlit as st
import chromadb
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from typing import Dict, Any

# Constants (Updated for TinyLlama)
CHROMA_DB_PATH = "/tmp/chroma_db"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource(show_spinner="🚀 Loading TinyLlama (this takes ~2 minutes)...")
def load_llm_model():
    """Load and cache the TinyLlama model with Render-optimized settings"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True  # Critical for Render's memory limits
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            device_map="auto"
        )
        st.session_state.model_loaded = True
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)[:200]}...")
        st.stop()

@st.cache_resource(show_spinner="🔧 Setting up ChromaDB...")
def initialize_chromadb():
    """Initialize ChromaDB with default settings"""
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=chromadb.Settings(anonymized_telemetry=False)
        return client.get_or_create_collection("portfolio")
    except Exception as e:
        st.warning(f"⚠️ ChromaDB initialization warning: {str(e)[:200]}...")
        return None

def generate_email(job_desc: str, candidate_details: Dict[str, Any]) -> str:
    """Generate email using TinyLlama with enhanced prompt engineering"""
    template = """[INST] Write a professional job application email:
    **Candidate Details:**
    - Name: {name}
    - Education: {education}
    - Experience: {experience}
    - Skills: {skills}
    
    **Job Description:**
    {job_desc}
    
    **Requirements:**
    1. Address hiring manager properly
    2. Highlight 2-3 relevant skills
    3. Keep under 250 words
    4. Professional closing [/INST]"""
    
    try:
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(
            llm=load_llm_model(),
            prompt=prompt,
            verbose=False
        )
        return chain.run(job_desc=job_desc, **candidate_details)[:1500]  # Output cap
    except Exception as e:
        return f"⚠️ Generation error: {str(e)[:200]}..."

def main():
    st.set_page_config(
        page_title="📧 Professional Email Generator",
        page_icon="✉️",
        layout="centered"
    )
    
    st.title("📧 AI-Powered Job Application Assistant")
    st.caption("Powered by TinyLlama 1.1B + ChromaDB")
    
    # Job Description Input
    job_desc = st.text_area(
        "Paste the Job Description:",
        height=150,
        placeholder="Example: 'Seeking a Python developer with web development experience...'"
    )
    
    # Candidate Details
    with st.expander("🧑💼 Applicant Details", expanded=True):
        name = st.text_input("Full Name*")
        email = st.text_input("Email*")
        education = st.text_area("Education*", placeholder="Degree, Institution, Year")
        experience = st.text_area("Experience*", placeholder="Previous roles and achievements")
        skills = st.text_area("Key Skills*", placeholder="Python, SQL, Communication...")
    
    if st.button("✨ Generate Email", type="primary"):
        if not all([job_desc, name, email, education, experience, skills]):
            st.warning("Please fill all required fields (*)")
        else:
            with st.spinner("Crafting your application (may take 30-60 seconds)..."):
                candidate_details = {
                    "name": name,
                    "education": education,
                    "experience": experience,
                    "skills": skills
                }
                email_content = generate_email(job_desc, candidate_details)
                
                st.subheader("📩 Your Custom Email Draft")
                st.markdown("---")
                st.write(email_content)
                st.markdown("---")
                
                st.download_button(
                    "💾 Download Email",
                    data=email_content,
                    file_name=f"Job_Application_{name.replace(' ', '_')}.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
