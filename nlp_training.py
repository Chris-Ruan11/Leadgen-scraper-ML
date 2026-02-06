import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import joblib

# Load the CSV (after fixing the header)
df = pd.read_csv('labeled_companies_with_text.csv')

# Filter out rows with no scraped_text
df = df[df['scraped_text'].notna() & (df['scraped_text'] != '')]

# Preprocess text: lowercase, remove non-alphabetic chars, etc.
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    return text

df['processed_text'] = df['scraped_text'].apply(preprocess_text)

# Features: TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['processed_text'])
y = df['label_relevance']  # 1 or 0

# Train model (Logistic Regression)
model = LogisticRegression()

# Cross-validation (5-fold) for better evaluation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Full predictions for detailed report
y_pred_cv = cross_val_predict(model, X, y, cv=5)
print("Cross-Validation Report:\n", classification_report(y, y_pred_cv))

# Now train on full data and save (for use in pipeline)
model.fit(X, y)
joblib.dump(model, 'relevance_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Optional: Single split for comparison (your original)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_split = LogisticRegression()
model_split.fit(X_train, y_train)
y_pred_split = model_split.predict(X_test)
print("\nSingle Split Accuracy (for reference):", accuracy_score(y_test, y_pred_split))