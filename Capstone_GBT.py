import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# Load data
df_train = pd.read_csv('data/train/train_raw.csv')
df_test = pd.read_csv('data/test/drugsComTest_raw.csv')

# Combine train and test for consistent vectorization
df = pd.concat([df_train, df_test], ignore_index=True)

# Create sentiment label
df['sentiment'] = (df['rating'] > 5).astype(int)

# Fill missing reviews
df['review'] = df['review'].fillna('')

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(df['review'])
y = df['sentiment'].values

# Split back into train/test
X_train = X[:len(df_train)]
y_train = y[:len(df_train)]
X_test = X[len(df_train):]
y_test = y[len(df_train):]

# Train/test split (optional, for validation)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train Gradient Boosting Classifier
gbt = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbt.fit(X_train, y_train)

# Predict and evaluate
y_pred = gbt.predict(X_test)
y_proba = gbt.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))