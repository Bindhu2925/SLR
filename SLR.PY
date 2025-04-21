import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import streamlit as st

# Sample dataset (Student Interactions & PCMB Learning Materials)
students = ["S1", "S2", "S3", "S4", "S5"]
subjects = ["Physics", "Chemistry", "Mathematics", "Biology"]
topics = [
    "Newton's Laws, Kinematics, Work & Energy, Gravitation",
    "Periodic Table, Chemical Bonding, Organic Chemistry, Thermodynamics",
    "Trigonometry, Algebra, Calculus, Probability",
    "Genetics, Human Physiology, Ecology, Evolution"
]

# Simulated Ratings (Students Rating Study Topics)
ratings = pd.DataFrame({
    "student_id": np.random.choice(students, 20),
    "subject": np.random.choice(subjects, 20),
    "rating": np.random.randint(3, 6, 20)  # Ratings between 3-5
})

# Create a subjects dataframe
subjects_df = pd.DataFrame({"subject": subjects, "topics": topics})

# 1. CONTENT-BASED FILTERING (TF-IDF + Cosine Similarity)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(subjects_df["topics"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommend(subject, top_n=3):
    """Returns top-N similar subjects based on topic similarity."""
    idx = subjects_df.index[subjects_df["subject"] == subject][0]
    scores = list(enumerate(cosine_sim[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [subjects_df.iloc[i[0]]["subject"] for i in sorted_scores]

# 2. COLLABORATIVE FILTERING (SVD Model)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['student_id', 'subject', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd = SVD()
svd.fit(trainset)

def collaborative_recommend(student_id, top_n=3):
    """Returns top-N recommended subjects based on collaborative filtering."""
    all_subjects = subjects_df["subject"].tolist()
    rated_subjects = ratings[ratings["student_id"] == student_id]["subject"].tolist()
    unrated_subjects = [subject for subject in all_subjects if subject not in rated_subjects]
    
    predictions = [(subject, svd.predict(student_id, subject).est) for subject in unrated_subjects]
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return [pred[0] for pred in sorted_predictions]

# 3. HYBRID RECOMMENDATION FUNCTION
def hybrid_recommend(student_id, subject, top_n=3):
    """Combines content-based and collaborative filtering for recommendations."""
    collab_recs = collaborative_recommend(student_id, top_n=top_n//2)
    content_recs = content_based_recommend(subject, top_n=top_n//2)
    return list(set(collab_recs + content_recs))

# 4. STREAMLIT UI FOR RECOMMENDATIONS
st.title("📚 PCMB Learning Material Recommender")

student_id = st.selectbox("Select Student:", students)
subject = st.selectbox("Select Subject:", subjects)

if st.button("Get Recommendations"):
    recommendations = hybrid_recommend(student_id, subject)
    st.write(f"✅ **Recommended Subjects for {student_id}:**")
    for rec in recommendations:
        st.write(f"- {rec}")
