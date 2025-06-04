import streamlit as st
import pandas as pd
import numpy as np
import logging
from ml_pipeline.base import TextPreprocessor
from ml_pipeline.models import BaseSentimentModel
from ml_pipeline.utils import get_model, setup_logging
from ml_pipeline.hf_sentiment import HFSentimentModel

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

setup_logging("streamlit_app.log")

st.set_page_config(page_title="Drug Sentiment Dashboard", layout="wide")
st.title("💊 Drug Review Sentiment Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    train_file = st.file_uploader("Training Data CSV", type="csv")
    test_file = st.file_uploader("Test Data CSV", type="csv")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

if train_file and test_file:
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # --- Condition Filter ---
    all_conditions = sorted(set(df_train['condition'].dropna().unique()) | set(df_test['condition'].dropna().unique()))
    condition = st.sidebar.selectbox("Filter by Condition", ["All"] + all_conditions)
    if condition != "All":
        df_train = df_train[df_train['condition'] == condition]
        df_test = df_test[df_test['condition'] == condition]

    # --- Key Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Reviews", len(df_train))
    col2.metric("Test Reviews", len(df_test))
    if condition != "All" and not df_train.empty:
        top_drug = df_train['drugName'].value_counts().idxmax()
        col3.metric("Top Drug (Train)", top_drug)
    else:
        col3.metric("Top Drug (Train)", "-")

    st.markdown("---")

    # --- Model Comparison ---
    st.subheader("Model Comparison")
    preprocessor = TextPreprocessor()
    X_train, X_test = preprocessor.fit_transform(df_train['review'], df_test['review'])
    y_train = (df_train['rating'] > 5).astype(int).values
    y_test = (df_test['rating'] > 5).astype(int).values

    model_names = ["gbt", "logistic", "naive_bayes", "random_forest", "svm", "hf_transformer"]
    results = []
    predictions = {}

    for name in model_names:
        if name == "hf_transformer":
            sentiment_model = HFSentimentModel()
            y_pred = sentiment_model.predict(df_test['review'])
            y_proba = None  # Not available for this pipeline
        else:
            model = get_model(name)
            sentiment_model = BaseSentimentModel(model)
            sample_weight = None
            if name == "gbt":
                class_counts = np.bincount(y_train)
                class_weights = {i: sum(class_counts) / (2 * c) for i, c in enumerate(class_counts)}
                sample_weight = np.array([class_weights[y] for y in y_train])
            sentiment_model.train(X_train, y_train, sample_weight=sample_weight)
            y_pred, y_proba = sentiment_model.evaluate(X_test, y_test, threshold=threshold)
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        f1 = f1_score(y_test, y_pred)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "ROC-AUC": roc,
            "F1-score": f1
        })
        predictions[name] = (y_pred, y_proba)

    results_df = pd.DataFrame(results)
    colA, colB = st.columns([2, 1])
    with colA:
        st.dataframe(results_df, use_container_width=True)
    with colB:
        best_model_row = results_df.sort_values("F1-score", ascending=False).iloc[0]
        st.success(f"Best Model: {best_model_row['Model']}\nF1-score: {best_model_row['F1-score']:.3f}")

    best_model_name = best_model_row['Model']
    y_pred, y_proba = predictions[best_model_name]

    st.markdown("---")

    # --- Insights Section ---
    st.subheader("Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class Distribution (Test Data)**")
        st.bar_chart(pd.Series(y_test).value_counts().sort_index().rename({0: "Negative/Neutral", 1: "Positive"}))

        st.markdown("**Top Drugs for This Condition (Train Data)**")
        if condition != "All" and not df_train.empty:
            st.write(df_train['drugName'].value_counts().head(5))
        else:
            st.write("N/A")

    with col2:
        st.markdown("**Confusion Matrix**")
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.markdown("---")

    # --- Trends Section ---
    st.subheader("Trends & Business Insights")
    col3, col4 = st.columns(2)

    with col3:
        if not df_test.empty and 'date' in df_test.columns:
            df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')
            df_test['predicted_sentiment'] = y_pred
            trend = df_test.groupby(df_test['date'].dt.to_period('M'))['predicted_sentiment'].mean()
            st.markdown("**Positive Sentiment Trend Over Time (Test Data)**")
            st.line_chart(trend)
        else:
            st.info("No date column available for trend analysis.")

    with col4:
        if not df_test.empty:
            avg_rating = df_test.groupby('drugName')['rating'].mean().sort_values(ascending=False).head(10)
            st.markdown("**Top 10 Drugs by Average Rating (Test Data)**")
            st.bar_chart(avg_rating)
        else:
            st.info("No test data for average rating.")

    st.markdown("---")

    # --- Download predictions ---
    st.subheader("Download")
    df_test["predicted_sentiment"] = y_pred
    st.download_button("Download Predictions", df_test.to_csv(index=False), "predictions.csv")

    # --- Logs and Details ---
    with st.expander("Show Pipeline Logs"):
        with open("streamlit_app.log") as f:
            st.text(f.read())

else:
    st.info("Please upload both training and test CSV files to begin.")
