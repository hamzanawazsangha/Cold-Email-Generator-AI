# AI Job Application Assistant ✉️🤖

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-app-url.streamlit.app/](https://cold-email-generator-ai.streamlit.app/))
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

An intelligent application that generates **personalized job application emails** using LangChain and Mistral-7B LLM. Developed as my first LangChain project at Ezitech Institute.

## Features ✨

- **AI-Powered Email Generation** - Creates tailored emails based on job descriptions
- **Professional Templates** - Industry-standard email structures
- **Smart Personalization** - Matches candidate skills to job requirements
- **User-Friendly Interface** - Simple form-based input
- **Secure** - No data storage or tracking

## Technology Stack 🛠️

| Component          | Technology |
|--------------------|------------|
| Framework          | LangChain  |
| LLM                | Mistral-7B (via Hugging Face) |
| Web Interface      | Streamlit  |
| Deployment         | Streamlit Cloud |
| Environment        | Python 3.10 |

## Installation 💻

1. Clone the repository:
```bash
git clone https://github.com/hamzanawazsangha/Cold-Email-Generator-AI.git
```
2. Navigate to project directory:
```bash
cd Cold-Email-Generator-AI
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create .env file:
```bash
HUGGINGFACEHUB_API_TOKEN=your_api_token_here
```
5. Run the app:
```bash
streamlit run app.py
```
## Usage Guide 📝
1. Enter Job Description - Paste the complete job posting
2. Fill Your Details - Add your professional information
3. Add Optional Links - Include LinkedIn/GitHub if available
4. Generate Email - Get your tailored application draft
5. Review & Download - Finalize and save the email

## Project Structure 📂
```
├── app.py                # Main application
├── requirements.txt      # Dependency list
├── .env.example          # Environment template
├── assets/               # Screenshots/visuals
└── README.md
```
## Contributing 🤝
- Fork the repository
- Create your feature branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request

Developed with ❤️ at Ezitech Institute
