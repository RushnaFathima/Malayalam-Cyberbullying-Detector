import streamlit as st
from googleapiclient.discovery import build
import joblib
import re
from preprocess import preprocess_text  # import preprocessing function

# -------------------------------
# Load trained model + vectorizer
# -------------------------------
model = joblib.load("calibrated.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# -------------------------------
# Comment filter rule:
# 1️⃣ Remove English
# 2️⃣ Keep if contains Malayalam text
# 3️⃣ Remove emoji-only comments
# -------------------------------
def is_valid_comment(text):
    
    # Remove English letters
    cleaned = re.sub(r"[A-Za-z]", "", text).strip()

    # Must contain Malayalam Unicode (U+0D00–U+0D7F)
    has_malayalam = bool(re.search(r'[\u0D00-\u0D7F]', cleaned))

    return has_malayalam


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔥 Malayalam YouTube Cyberbullying Detector")

st.write("This app fetches YouTube comments and classifies them as **Cyberbullying** or **Non-Cyberbullying** using a trained ML model (.pkl).")

api_key = st.text_input("🔑 Enter your YouTube API Key:")
video_id = st.text_input("🎬 Enter YouTube Video ID (example: dQw4w9WgXcQ):")

if st.button("🚀 Analyze Comments"):

    if not api_key.strip():
        st.error("❌ Please enter a valid API key.")
    elif not video_id.strip():
        st.error("❌ Please enter a video ID.")
    else:
        try:
            st.info("⏳ Fetching comments...")

            youtube = build("youtube", "v3", developerKey=api_key)

            comments = []
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText"
            )

            response = request.execute()

            while response:
                for item in response['items']:
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

                    if is_valid_comment(comment):
                        comments.append(comment)

                if "nextPageToken" in response:
                    response = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        textFormat="plainText",
                        pageToken=response["nextPageToken"]
                    ).execute()
                else:
                    break

            if not comments:
                st.warning("⚠ No Malayalam comments detected after filtering.")
                st.stop()

            st.success(f"✔ {len(comments)} valid Malayalam comments extracted.")

            # -------------------------------
            # Apply preprocessing + predict
            # -------------------------------
            cleaned_comments = [preprocess_text(c) for c in comments]

            X_test = vectorizer.transform(cleaned_comments)
            predictions = model.predict(X_test)

            cyber = [c for c, p in zip(comments, predictions) if p == 1]
            safe = [c for c, p in zip(comments, predictions) if p == 0]

            st.write("---")
            st.subheader("📊 Results Summary")
            st.write(f"📝 Total analyzed: **{len(comments)}**")
            st.write(f"🚨 Cyberbullying: **{len(cyber)}**")
            st.write(f"🙂 Non-Cyberbullying: **{len(safe)}**")

            if cyber:
                st.write("---")
                st.subheader("🚨 Cyberbullying Comments")
                for i, c in enumerate(cyber, 1):
                    st.error(f"{i}. {c}")

            st.write("---")
            st.subheader("🙂 Non-Cyberbullying Comments")
            for i, c in enumerate(safe, 1):
                st.success(f"{i}. {c}")

        except Exception as e:
            st.error("❌ Error while accessing YouTube API.")
            st.code(str(e))
