import streamlit as st
from summarizer import run_rag_pipeline

st.set_page_config(page_title="PDF Summarizer", layout="centered")
st.title(" PDF Summarizer using RAG")

pdf_file = st.file_uploader(" Upload a PDF file", type=["pdf"])
query = st.text_input(" Ask something about the document", "Summarize this document")

if pdf_file and query and st.button("Generate Summary"):
    with st.spinner("Processing..."):
        try:
            result = run_rag_pipeline(pdf_file, query)
            st.success(" Done!")
            st.markdown("### Answer:")
            st.write(result)
        except Exception as e:
            st.error(f" Something went wrong:\n\n{e}")
