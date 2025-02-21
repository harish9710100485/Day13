Email Conversion Prediction Model
Overview
This project uses machine learning to predict if a customer is likely to buy based on email text. It employs logistic regression and text preprocessing techniques.

Features
Cleans email text by removing special characters and converting to lowercase.
Converts text data into numerical features using TF-IDF Vectorization.
Uses a logistic regression model for classification.
Saves and loads the trained model using joblib.
Installation
bash
Copy
Edit
pip install numpy pandas scikit-learn joblib
Usage
Training the Model
python
Copy
Edit
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Sample dataset
data = {
    "email": [
        "Hello, we are interested in your product, can you share pricing?",
        "Not interested, please do not contact again.",
        "Looking forward to purchasing 50 units next month.",
        "Stop sending me emails!",
        "Can we get a discount on bulk purchase?",
        "Send me more details about the specifications."
    ],
    "converted": [1, 0, 1, 0, 1, 1]  # 1 = Bought, 0 = Not Interested
}

df = pd.DataFrame(data)

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

df["cleaned_email"] = df["email"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_email"], df["converted"], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "email_prediction_model.pkl")
Predicting Conversion
python
Copy
Edit
import joblib

def predict_conversion(email_text):
    email_text = clean_text(email_text)
    model = joblib.load("email_prediction_model.pkl")
    return "Likely to Buy" if model.predict([email_text])[0] == 1 else "Not Interested"

# Example Usage
new_email = "We are considering buying your product, send pricing details."
print(predict_conversion(new_email))
Author:Harish
Intern:Minervasoft
