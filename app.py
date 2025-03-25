import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="AI Job Application Assistant",
    page_icon="✉️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .optional-field {
        color: #666666;
        font-style: italic;
    }
    .stTextArea [data-baseweb=base-input] {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

def validate_inputs(job_desc, candidate_details):
    """Validate all required fields are filled"""
    required_fields = ['name', 'email', 'phone', 'education', 'experience', 'skills']
    if not job_desc.strip():
        return "Job description cannot be empty"
    for field in required_fields:
        if not candidate_details.get(field, '').strip():
            return f"Please fill in the {field.replace('_', ' ')} field"
    return None

def generate_email(job_desc, candidate_details):
    """Generate professional email using LLM"""
    # Build optional fields string
    optional_fields = []
    if candidate_details.get('address'):
        optional_fields.append(f"Address: {candidate_details['address']}")
    if candidate_details.get('linkedin'):
        optional_fields.append(f"LinkedIn: {candidate_details['linkedin']}")
    if candidate_details.get('github'):
        optional_fields.append(f"GitHub: {candidate_details['github']}")
    
    optional_fields_str = "\n    ".join(optional_fields) if optional_fields else "None provided"

    template = """You are a professional career coach. Write a compelling cold email for a job application using these details:

    Candidate Information:
    - Name: {name}
    - Contact: {email} | {phone}
    - Education: {education}
    - Experience: {experience}
    - Key Skills: {skills}
    
    Additional Details:
    {optional_fields}

    Job Description:
    {job_desc}

    Email Requirements:
    1. Professional but approachable tone
    2. 3-4 concise paragraphs (under 300 words total)
    3. First paragraph: Introduction and interest in position
    4. Second paragraph: Most relevant qualifications (match 2-3 skills from JD)
    5. Third paragraph: Unique value proposition
    6. Closing: Call to action and appreciation
    7. Include links to LinkedIn/GitHub if provided
    8. Professional signature with contact info

    Important:
    - Never make up information not provided by the candidate
    - For links, use markdown formatting: [LinkedIn](url)
    - If GitHub is provided, mention relevant projects
    """

    try:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 600,
                "repetition_penalty": 1.2,
                "top_p": 0.9
            },
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["job_desc", "name", "email", "phone", "education", 
                           "experience", "skills", "optional_fields"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        with st.spinner("Crafting your perfect application email..."):
            response = chain.run(
                job_desc=job_desc,
                optional_fields=optional_fields_str,
                **{k: v for k, v in candidate_details.items() if k not in ['address', 'linkedin', 'github']}
            )
        
        # Post-process the response
        return response.strip()
    
    except Exception as e:
        st.error("Failed to generate email. Please try again.")
        st.exception(e)
        return None

def main():
    st.title("✉️ AI-Powered Job Application Assistant")
    st.markdown("Create professional, tailored application emails in seconds")
    
    with st.expander("ℹ️ How to use this tool", expanded=True):
        st.write("""
        1. Paste the job description
        2. Fill in your details (required fields marked with *)
        3. Add optional info like LinkedIn/GitHub if available
        4. Click 'Generate Email'
        5. Review and download the result
        """)

    # Main form
    with st.form("application_form"):
        st.subheader("Job Details")
        job_desc = st.text_area(
            "Paste the full job description here*",
            height=200,
            placeholder="Copy and paste the complete job posting...",
            help="The more details you provide, the better we can tailor your email"
        )

        st.subheader("Your Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*")
            email = st.text_input("Email Address*")
            phone = st.text_input("Phone Number*")
            address = st.text_input("Address", help="Optional mailing address")
        
        with col2:
            education = st.text_area("Education Background*", height=100)
            experience = st.text_area("Professional Experience*", height=100)
            skills = st.text_area("Key Skills*", height=100, 
                                help="List your most relevant skills for this position")
        
        # Optional fields section
        st.subheader("Optional Information", help="These will make your application more compelling")
        linkedin = st.text_input("LinkedIn Profile URL", 
                                placeholder="https://linkedin.com/in/yourprofile",
                                help="Will be included in your signature")
        github = st.text_input("GitHub Profile URL", 
                             placeholder="https://github.com/yourusername",
                             help="Mentioned if relevant to the position")

        submitted = st.form_submit_button("✨ Generate Email", type="primary")

    # Handle form submission
    if submitted:
        candidate_details = {
            "name": name,
            "email": email,
            "phone": phone,
            "address": address,
            "education": education,
            "experience": experience,
            "skills": skills,
            "linkedin": linkedin,
            "github": github
        }

        # Validate inputs
        validation_error = validate_inputs(job_desc, candidate_details)
        if validation_error:
            st.error(validation_error)
        else:
            # Generate and display email
            email_content = generate_email(job_desc, candidate_details)
            
            if email_content:
                st.success("Email generated successfully!")
                st.markdown("---")
                st.subheader("Your Custom Application Email")
                
                with st.container(border=True):
                    st.markdown(email_content)
                
                # Download button
                st.download_button(
                    label="📥 Download Email",
                    data=email_content,
                    file_name=f"Job_Application_{name.replace(' ', '_')}.txt",
                    mime="text/plain",
                    help="Save this email to use in your application"
                )

                # Improvement suggestions
                with st.expander("💡 Tips for Improvement"):
                    st.write("""
                    - **Personalization**: Add a specific reason why you're excited about this company
                    - **Achievements**: Include 1-2 metrics from your experience
                    - **Links**: Double-check your LinkedIn/GitHub URLs
                    - **Follow-up**: Mention when you'll follow up (e.g., 'I'll reach out next Thursday')
                    """)

if __name__ == "__main__":
    # Check for required environment variable
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        st.error("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your environment variables.")
        st.stop()
    
    main()
