import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to calculate cosine similarity
def similarity(doc1, doc2): 
    return cosine_similarity ([doc1, doc2])[0][1]

# Main function to check plagiarism and calculate percentile
def check_plagiarism_and_percentile(text1, text2, text3):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2, text3]).toarray()
    plagiarism_scores = []

    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            sim_score = similarity(vectors[i], vectors[j])
            plagiarism_scores.append(sim_score)

    percentile = np.percentile(plagiarism_scores, 90)

    return plagiarism_scores, percentile

# Streamlit UI
st.title("Plagiarism Checker")
st.write("Enter three texts (maximum 5000 characters each) and click 'Check Plagiarism' to find plagiarism.")

# Text inputs for three texts with a character limit
text1 = st.text_area("Enter text 1:", max_chars=5000, height=150)
text2 = st.text_area("Enter text 2:", max_chars=5000, height=150)
text3 = st.text_area("Enter text 3:", max_chars=5000, height=150)

# Button to trigger plagiarism check
check_button = st.button("Check Plagiarism")

# Styling
st.markdown("---")
st.markdown(
    """
    <style>
    .stTextInput>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #d3d3d3;
        padding: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        padding: 10px 15px;
        background-color: #008CBA;
        color: white;
    }
    .stButton>button:hover {
        background-color: #005f73;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Result
if check_button:
    if text1.strip() == "" or text2.strip() == "" or text3.strip() == "":
        st.warning("Please enter text in all three boxes.")
    else:
        plagiarism_scores, _ = check_plagiarism_and_percentile(text1, text2, text3)
        for i, score in enumerate(plagiarism_scores, start=1):
            pair_num = f"{i//2 + 1}" if i % 2 == 0 else f"{i//2 + 1} and {i//2 + 2}"
            st.success(f"percentile of plagiarism for pair {pair_num}: {score:.2f}")