from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import os

class BaseSentimentModel:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, sample_weight=None):
        import logging
        logging.info("Training model...")
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        logging.info("Model training complete.")

    def evaluate(self, X_test, y_test, threshold=0.5):
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
        if y_proba is not None:
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        if y_proba is not None:
            print("ROC-AUC:", roc_auc_score(y_test, y_proba))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return y_pred, y_proba

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)