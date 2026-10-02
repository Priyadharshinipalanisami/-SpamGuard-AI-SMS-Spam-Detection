import streamlit as st
import pandas as pd
import joblib
import re
import os
from datetime import datetime


MODEL_PATH = "models/spam_pipeline.pkl"


st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide"
)


st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #64748b;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.spam {
    padding: 20px;
    border-radius: 15px;
    background: #fee2e2;
    border-left: 6px solid #dc2626;
}

.ham {
    padding: 20px;
    border-radius: 15px;
    background: #dcfce7;
    border-left: 6px solid #16a34a;
}

.metric {
    padding: 20px;
    border-radius: 15px;
    background: white;
    text-align: center;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.06);
}

</style>
""", unsafe_allow_html=True)



if not os.path.exists(MODEL_PATH):

    st.error(
        "Model not found. Please run `py src/train.py` first."
    )

    st.stop()


model = joblib.load(MODEL_PATH)


if "history" not in st.session_state:
    st.session_state.history = []



st.markdown(
    '<div class="title">🛡️ SpamGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time SMS Spam Detection using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)



with st.sidebar:

    st.header("⚙️ System")

    st.success("Model Loaded")

    st.write("**Pipeline**")
    st.write("TF-IDF Vectorization")
    st.write("Machine Learning Classifier")

    st.divider()

    st.header("📊 Dataset")

    st.write("SMS Spam Collection")
    st.write("5,572 messages")
    st.write("4,825 Ham")
    st.write("747 Spam")

    st.divider()

    st.info(
        "Enter a new SMS message to perform real-time prediction."
    )



total = len(st.session_state.history)

spam_count = sum(
    1 for x in st.session_state.history
    if x["Prediction"] == "SPAM"
)

ham_count = sum(
    1 for x in st.session_state.history
    if x["Prediction"] == "HAM"
)

avg_confidence = (
    sum(x["Confidence"] for x in st.session_state.history) / total
    if total > 0
    else 0
)


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Messages Analyzed", total)

with c2:
    st.metric("Spam Detected", spam_count)

with c3:
    st.metric("Ham Detected", ham_count)

with c4:
    st.metric(
        "Avg Confidence",
        f"{avg_confidence * 100:.1f}%"
    )


st.write("")


st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📩 Real-Time SMS Analyzer")

message = st.text_area(
    "Enter a new SMS message",
    height=150,
    placeholder=(
        "Example: Congratulations! You have won a free prize. "
        "Click the link to claim now."
    )
)

analyze = st.button(
    "🔍 Analyze Message",
    use_container_width=True
)

st.markdown("</div>", unsafe_allow_html=True)


if analyze:

    if not message.strip():

        st.warning("Please enter an SMS message.")

    else:

        cleaned = message.lower()

        cleaned = re.sub(
            r"http\S+|www\S+",
            " URL ",
            cleaned
        )

        cleaned = re.sub(
            r"\S+@\S+",
            " EMAIL ",
            cleaned
        )

        cleaned = re.sub(
            r"\d+",
            " NUMBER ",
            cleaned
        )

        cleaned = re.sub(
            r"[^a-zA-Z\s]",
            " ",
            cleaned
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned
        ).strip()

        prediction = model.predict([cleaned])[0]

        probabilities = model.predict_proba([cleaned])[0]

        ham_probability = probabilities[0]
        spam_probability = probabilities[1]

        if prediction == 1:

            result = "SPAM"
            confidence = spam_probability

            st.markdown(
                f"""
                <div class="spam">
                    <h2>🚨 SPAM MESSAGE</h2>
                    <h3>Confidence: {confidence * 100:.2f}%</h3>
                    <p>This message has characteristics associated
                    with spam messages.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            result = "HAM"
            confidence = ham_probability

            st.markdown(
                f"""
                <div class="ham">
                    <h2>✅ HAM / SAFE MESSAGE</h2>
                    <h3>Confidence: {confidence * 100:.2f}%</h3>
                    <p>This message appears similar to legitimate
                    messages in the training data.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # Probability columns

        st.subheader("📊 Prediction Probability")

        p1, p2 = st.columns(2)

        with p1:

            st.write(
                f"**Ham:** {ham_probability * 100:.2f}%"
            )

            st.progress(
                float(ham_probability)
            )

        with p2:

            st.write(
                f"**Spam:** {spam_probability * 100:.2f}%"
            )

            st.progress(
                float(spam_probability)
            )


        # Save history

        st.session_state.history.append({
            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Message": message,
            "Prediction": result,
            "Confidence": confidence
        })


if st.session_state.history:

    st.divider()

    st.subheader("📜 Real-Time Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    display_df = history_df.copy()

    display_df["Confidence"] = (
        display_df["Confidence"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False)

    st.download_button(
        "⬇️ Download Prediction History",
        csv,
        "spam_prediction_history.csv",
        "text/csv"
    )


if os.path.exists("models/model_comparison.csv"):

    st.divider()

    st.subheader("📈 Model Performance")

    performance = pd.read_csv(
        "models/model_comparison.csv"
    )

    st.dataframe(
        performance.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        performance.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        ]
    )



st.divider()

st.caption(
    "SpamGuard AI | Machine Learning Based SMS Spam Detection"
)