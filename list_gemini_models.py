import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Gemini Model Browser", layout="wide")
st.title("✨ Gemini Model Browser")

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY in your .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    st.write("Fetching available Gemini models...")

    try:
        # list_models() returns a generator → convert to list
        models = list(genai.list_models())
        
        st.success(f"Found {len(models)} models")

        # Display each model
        for m in models:
            st.subheader(m.name)
            st.write("**Supported generation methods:**")
            st.code(", ".join(m.supported_generation_methods))
            st.markdown("---")

    except Exception as e:
        st.error(f"Error fetching models: {e}")
