import streamlit as st
import os
import PyPDF2
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
from nltk.corpus import stopwords

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def rank_resumes(resumes, job_description):
    texts = [job_description] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity_scores[0]

def main():
    st.title("AI-Powered Resume Screening and Ranking")
    
    job_description = st.text_area("Enter Job Description:")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Rank Resumes"):
        if not job_description:
            st.warning("Please enter a job description.")
            return
        
        resumes_text = []
        filenames = []
        
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            preprocessed_text = preprocess_text(text)
            resumes_text.append(preprocessed_text)
            filenames.append(uploaded_file.name)
        
        scores = rank_resumes(resumes_text, preprocess_text(job_description))
        
        results = pd.DataFrame({"Resume": filenames, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
        
        st.subheader("Ranking Results")
        st.write(results)

if __name__ == "__main__":
    main()
