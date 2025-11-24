import streamlit as st
from googleapiclient.discovery import build
import joblib
import re
# ⬇⬇⬇ ADD THIS (same class used during training)
class TextPreprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # your preprocessing logic
        return X

# ⬆⬆⬆ MUST BE ABOVE joblib.load()
# Load full ML pipeline
pipeline = joblib.load("Malayalam_Cyber_Model.pkl")

# Filtering: remove English comments
def is_valid_comment(text):
    return not bool(re.search(r'[A-Za-z]', text))

API_KEY = "AIzaSyABxi2C5JGjxQVCTWqOTaHBVp0hhx6qzNg"
youtube = build("youtube", "v3", developerKey=API_KEY)

st.title("🔥 YouTube Cyberbullying Detector (Malayalam + Emoji)")

video_id = st.text_input("📌 Enter YouTube Video ID:")

if st.button("Extract & Analyze Comments") and video_id.strip():

    st.info("⏳ Fetching comments from YouTube...")

    comments = []
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText"
    ).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            if is_valid_comment(comment):
                comments.append(comment)

        if "nextPageToken" in results:
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                pageToken=results["nextPageToken"]
            ).execute()
        else:
            break

    st.success(f"✔ Extracted {len(comments)} Malayalam comments.")

    # Predict using pipeline
    predictions = pipeline.predict(comments)

    cyber = [c for c, p in zip(comments, predictions) if p == 1]
    non_cyber = [c for c, p in zip(comments, predictions) if p == 0]

    st.subheader("📊 Summary")
    st.write(f"📝 Total Comments: **{len(comments)}**")
    st.write(f"🚨 Cyberbullying: **{len(cyber)}**")
    st.write(f"✅ Non-Cyberbullying: **{len(non_cyber)}**")

    st.write("---")

    if cyber:
        st.subheader("🚨 Cyberbullying Comments")
        for i, c in enumerate(cyber, 1):
            st.write(f"{i}. {c}")

    st.write("---")

    st.subheader("✅ Non-Cyberbullying Comments")
    for i, c in enumerate(non_cyber, 1):
        st.write(f"{i}. {c}")
