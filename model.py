import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data.csv")

# Features
X = df['clean_email']

# Labels
y_category = df['category']
y_urgency = df['urgency']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split (for evaluation)
X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X_vectorized, y_category, test_size=0.2, random_state=42
)

_, _, y_urg_train, y_urg_test = train_test_split(
    X_vectorized, y_urgency, test_size=0.2, random_state=42
)

# Models
category_model = MultinomialNB()
urgency_model = MultinomialNB()

# Train models
category_model.fit(X_train, y_cat_train)
urgency_model.fit(X_train, y_urg_train)

# Accuracy (for dashboard)
cat_accuracy = accuracy_score(y_cat_test, category_model.predict(X_test))
urg_accuracy = accuracy_score(y_urg_test, urgency_model.predict(X_test))


# Prediction functions
def predict_all(email):
    email_vec = vectorizer.transform([email])

    category = category_model.predict(email_vec)[0]
    urgency = urgency_model.predict(email_vec)[0]

    # Confidence (probability)
    cat_prob = max(category_model.predict_proba(email_vec)[0])
    urg_prob = max(urgency_model.predict_proba(email_vec)[0])

    return {
        "category": category,
        "urgency": urgency,
        "cat_conf": round(cat_prob * 100, 2),
        "urg_conf": round(urg_prob * 100, 2)
    }


# Dataset insights
def get_stats():
    return {
        "total_emails": len(df),
        "category_counts": df['category'].value_counts(),
        "urgency_counts": df['urgency'].value_counts(),
        "cat_accuracy": round(cat_accuracy * 100, 2),
        "urg_accuracy": round(urg_accuracy * 100, 2)
    }