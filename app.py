import os
import streamlit as st
from dotenv import load_dotenv
from vector_store import create_vector_store
from qa_chain import ask_question
from sklearn.metrics import f1_score





load_dotenv()
st.title("📄 PDF Q&A Bot")

api_key = st.text_input("OpenAI API Key (optional)", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask a question about the document:")

if uploaded_file and query:
    with st.spinner("Processing..."):
        retriever = create_vector_store(uploaded_file)
        answer = ask_question(retriever, query)
        st.success(answer)

