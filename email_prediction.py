import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Sample dataset (Replace with actual email data)
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

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df["cleaned_email"] = df["email"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_email"], df["converted"], test_size=0.2, random_state=42
)

# Build Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, "email_prediction_model.pkl")

# Function to predict

def predict_conversion(email_text):
    email_text = clean_text(email_text)
    model = joblib.load("email_prediction_model.pkl")
    prediction = model.predict([email_text])
    return "Likely to Buy" if prediction[0] == 1 else "Not Interested"

# Example Usage
new_email = "We are considering buying your product, send pricing details."
print(predict_conversion(new_email))
