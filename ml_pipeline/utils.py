import logging

def setup_logging(log_file="pipeline.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_model(model_name):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC

    if model_name == "gbt":
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == "logistic":
        return LogisticRegression(max_iter=1000)
    elif model_name == "naive_bayes":
        return MultinomialNB()
    elif model_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "svm":
        return SVC(probability=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")