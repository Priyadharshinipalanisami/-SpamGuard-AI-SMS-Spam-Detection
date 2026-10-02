A machine-learning based SMS Spam Detection System that classifies messages as Spam or Ham (Legitimate) using TF-IDF text vectorization and machine-learning algorithms. The project includes model evaluation, real-time prediction, confidence scores, prediction history, and an interactive Streamlit dashboard.

📌 Project Overview

Spam messages are unwanted or potentially harmful messages that may contain misleading offers, fraudulent links, advertisements, or other suspicious content.

SpamGuard AI automatically analyzes SMS text and predicts whether a message is:

🚨 SPAM
✅ HAM / SAFE

The system compares Multinomial Naive Bayes and Logistic Regression and selects the model based on the F1-score.

🎯 Objectives
Detect spam SMS messages automatically.
Clean and preprocess SMS text.
Convert text into numerical features using TF-IDF.
Train multiple machine-learning classifiers.
Compare models using:
Accuracy
Precision
Recall
F1-score
Save the trained model for reuse.
Provide real-time SMS classification.
Display prediction confidence and probabilities.
Maintain prediction history.
Provide a downloadable CSV of predictions.
✨ Features
Feature	Description
🧹 Text Preprocessing	Cleans URLs, numbers, special characters and unnecessary spaces
🔤 TF-IDF	Converts SMS text into numerical features
🤖 ML Classification	Uses Naive Bayes and Logistic Regression
📊 Model Evaluation	Calculates Accuracy, Precision, Recall and F1-score
⚡ Real-Time Prediction	Classifies newly entered SMS instantly
🎯 Confidence Score	Displays Spam/Ham probability
📜 Prediction History	Stores messages analyzed during the session
📥 CSV Export	Downloads prediction history
📈 Performance Dashboard	Displays model comparison
🌐 Streamlit UI	Interactive web-based interface
🧠 Machine Learning Workflow
                SMS Dataset
                    │
                    ▼
             Data Preprocessing
                    │
                    ▼
              Text Cleaning
                    │
                    ▼
             TF-IDF Vectorizer
                    │
                    ▼
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   Naive Bayes        Logistic Regression
          │                   │
          └─────────┬─────────┘
                    ▼
             Model Evaluation
                    │
                    ▼
              Best Model
                    │
                    ▼
          spam_pipeline.pkl
                    │
                    ▼
          Real-Time SMS Input
                    │
                    ▼
             Spam / Ham
                    │
                    ▼
          Confidence & Result
📂 Project Structure
Spam-Detection-AI/
│
├── data/
│   └── SMSSpamCollection
│
├── models/
│   ├── spam_pipeline.pkl
│   └── model_comparison.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── screenshots/
│   ├── dashboard.png
│   ├── spam_result.png
│   ├── ham_result.png
│   └── model_comparison.png
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
📊 Dataset

This project uses the SMS Spam Collection dataset.

The dataset contains SMS messages labeled as:

ham
spam

The original UCI dataset contains 5,574 SMS messages. Your uploaded dataset contains 5,572 records, which can be used as the training data for this project.

Dataset source: UCI Machine Learning Repository.

SMS Spam Collection — UCI Machine Learning Repository

Dataset format
ham    Go until jurong point, crazy.. Available only ...
spam   Free entry in 2 a wkly comp to win FA Cup...
🛠️ Technologies Used
Programming Language
Python 3
Machine Learning
Scikit-learn
Multinomial Naive Bayes
Logistic Regression
Natural Language Processing
TF-IDF Vectorization
Regular Expression-based text cleaning
Web Application
Streamlit
Data Processing
Pandas
NumPy
Model Storage
Joblib
📦 Installation
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/Spam-Detection-AI.git
cd Spam-Detection-AI
2. Install dependencies
py -m pip install -r requirements.txt

If py is not available, use:

python -m pip install -r requirements.txt
🚀 Train the Model

Run:

py src/train.py

The program will:

Load the SMS dataset.
Clean the text.
Split the dataset into training and testing sets.
Apply TF-IDF.
Train Naive Bayes.
Train Logistic Regression.
Calculate evaluation metrics.
Compare the models.
Select the model with the highest F1-score.
Save the trained pipeline.

The trained model will be saved as:

models/spam_pipeline.pkl

Model comparison will be saved as:

models/model_comparison.csv
⚡ Run Real-Time Prediction

You can test messages through the command line:

py src/predict.py

Example:

============================================================
REAL-TIME SMS SPAM DETECTOR
============================================================

Enter SMS message:
Congratulations! You have won a FREE prize. Claim now!

Possible output:

Prediction : SPAM
Confidence : 98.25%
Ham        : 1.75%
Spam       : 98.25%
🌐 Run Streamlit Dashboard

Start the application:

py -m streamlit run app.py

The application will normally be available at:

http://localhost:8501

The dashboard provides:

Real-time SMS analysis
Spam/Ham classification
Confidence score
Probability visualization
Prediction history
CSV download
Model performance comparison
🧪 Example Predictions
Example 1 — Spam

Input:

Congratulations! You have won a FREE prize of $1000.
Click now to claim your reward!

Expected:

🚨 SPAM MESSAGE
Example 2 — Ham

Input:

Hey, are you coming to college tomorrow?

Expected:

✅ HAM / SAFE MESSAGE
Example 3 — Spam

Input:

WINNER! You have been selected for a special cash reward.
Reply YES now.

Expected:

🚨 SPAM MESSAGE
Example 4 — Ham

Input:

Please send me the assignment when you are free.

Expected:

✅ HAM / SAFE MESSAGE

The exact prediction and confidence depend on the trained model and the message.

📈 Model Evaluation

The system evaluates the classifiers using four important metrics.

Accuracy

Measures the percentage of correctly classified messages.

$$ Accuracy = \frac{TP + TN} {TP + TN + FP + FN} $$
Precision

Measures how many messages predicted as spam are actually spam.

$$ Precision = \frac{TP} {TP + FP} $$
Recall

Measures how many actual spam messages were detected.

$$ Recall = \frac{TP} {TP + FN} $$
F1-Score

Combines precision and recall.

$$ F1 = 2 \times \frac{Precision \times Recall} {Precision + Recall} $$
📊 Confusion Matrix
                    Actual
                 Ham       Spam
              ┌─────────┬─────────┐
Predicted Ham │   TN    │   FN    │
              ├─────────┼─────────┤
Predicted Spam│   FP    │   TP    │
              └─────────┴─────────┘

Where:

TP = Spam correctly classified as spam
TN = Ham correctly classified as ham
FP = Ham incorrectly classified as spam
FN = Spam incorrectly classified as ham
🔬 Text Preprocessing

Before classification, the SMS is cleaned.

The preprocessing pipeline includes:

Original SMS
     ↓
Convert to lowercase
     ↓
Remove URLs
     ↓
Remove email addresses
     ↓
Replace numbers
     ↓
Remove special characters
     ↓
Remove extra spaces
     ↓
Clean SMS

Example:

"Congratulations!!! WIN $1000. Visit www.example.com"

becomes approximately:

"congratulations win NUMBER visit URL"
🔤 TF-IDF

TF-IDF stands for:

Term Frequency–Inverse Document Frequency

It converts text into numerical feature vectors that can be processed by machine-learning algorithms.

The general formula is:

$$ TF-IDF(t,d)=TF(t,d)\times IDF(t) $$

This allows the model to identify words and word combinations that are useful for distinguishing spam from legitimate messages.

🤖 Algorithms
1. Multinomial Naive Bayes

Naive Bayes is a probabilistic classification algorithm commonly used for text classification.

It estimates the probability that a message belongs to the spam or ham class based on its features.

2. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a message belonging to a particular class.

In this project:

0 → Ham
1 → Spam

The two models are evaluated and compared using the same test dataset.

🖥️ Dashboard

The Streamlit dashboard contains:

┌─────────────────────────────────────────────┐
│              🛡️ SpamGuard AI                │
│     Real-Time SMS Spam Detection            │
├──────────┬──────────┬──────────┬────────────┤
│ Messages │   Spam   │   Ham    │ Confidence │
├──────────┴──────────┴──────────┴────────────┤
│                                             │
│ 📩 Real-Time SMS Analyzer                   │
│                                             │
│ Enter SMS message...                        │
│                                             │
│        🔍 Analyze Message                   │
│                                             │
├─────────────────────────────────────────────┤
│ 🚨 SPAM MESSAGE                             │
│ Confidence: XX.XX%                          │
├─────────────────────────────────────────────┤
│ Ham Probability     Spam Probability        │
│      XX%                  XX%                │
├─────────────────────────────────────────────┤
│ 📜 Prediction History                       │
└─────────────────────────────────────────────┘
💾 Saved Model

After training, the complete preprocessing and classification pipeline is stored using Joblib:

models/spam_pipeline.pkl

This allows the application to reuse the trained model without retraining every time.

📥 Prediction History

The application records:

Time
Message
Prediction
Confidence

Example:

Time	Prediction	Confidence
10:30:12	SPAM	98.2%
10:31:05	HAM	96.4%
10:32:44	SPAM	97.8%

The history can be downloaded as:

spam_prediction_history.csv
🔮 Future Enhancements
📱 Integration with mobile SMS applications
📧 Email spam detection
🌐 REST API for external applications
🔄 Continuous model retraining
🧠 Advanced transformer-based NLP models
🌍 Multilingual spam detection
🔗 Malicious URL detection
📊 Advanced analytics dashboard
☁️ Cloud deployment
🔔 Real-time alert system
⚠️ Limitations
The model is trained on historical SMS data.
Prediction confidence is not a guarantee that a message is actually spam.
New spam patterns may not be represented in the training dataset.
Language and writing style can affect classification performance.
A truly live production system would require a secure mechanism for receiving incoming messages and appropriate privacy controls.
🎓 Academic Project

Project Title:

Spam Detection Using Machine Learning

Domain:
Machine Learning / Natural Language Processing

Application:
SMS Spam Classification

Models:

Multinomial Naive Bayes
Logistic Regression

Feature Extraction:
TF-IDF

Frontend:
Streamlit

Language:
Python

👩‍💻 Author

Your Name

MCA – Final Year

📜 License

This project is intended for educational and academic purposes. Check the dataset's licensing terms before redistributing the dataset itself with your repository. The UCI SMS Spam Collection is listed under CC BY 4.0.
