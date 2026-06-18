import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake News Detector")
st.write("AI-powered fake news detection using Hugging Face Transformers")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="hamzab/roberta-fake-news-classification"
    )

classifier = load_model()

text = st.text_area(
    "Paste News Article or Headline",
    height=200,
    placeholder="Enter news text here..."
)

if st.button("Analyze"):

    if len(text.strip()) < 10:
        st.warning("Please enter some news text.")
        st.stop()

    with st.spinner("Analyzing..."):

        result = classifier(text[:512])

        label = result[0]["label"]
        score = result[0]["score"]

        if "FAKE" in label.upper():
            verdict = "FAKE"
            st.error(f"🚨 Verdict: {verdict}")
        else:
            verdict = "REAL"
            st.success(f"✅ Verdict: {verdict}")

        st.metric(
            "Confidence Score",
            f"{score*100:.2f}%"
        )

        st.subheader("Analysis Summary")

        if verdict == "FAKE":
            st.write(
                "The model detected characteristics commonly associated "
                "with misinformation, sensational claims, or unreliable reporting."
            )
        else:
            st.write(
                "The model found linguistic patterns that are more consistent "
                "with legitimate news reporting."
            )

        st.subheader("Verification Tips")

        st.write(
            """
            • Check the original source

            • Verify with trusted news organizations

            • Look for supporting evidence

            • Avoid sharing unverified information
            """
        )