from transformers import pipeline

class HFSentimentModel:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.pipe = pipeline("text-classification", model=model_name)

    def predict(self, texts):
        results = self.pipe(list(texts))
        return [1 if r['label'] == 'POSITIVE' else 0 for r in results]