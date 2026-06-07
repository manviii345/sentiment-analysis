# pyrefly: ignore [missing-import]
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

class TextPipeline:
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.current_scores = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)

    def analyze(self, text):
        if not self.model or not self.tokenizer:
            self.load_model()
            
        if not text.strip():
            self.current_scores = {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
            return self.current_scores

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = F.softmax(outputs.logits, dim=-1)[0].numpy()
        
        self.current_scores = {
            'negative': float(probs[0]),
            'neutral': float(probs[1]),
            'positive': float(probs[2])
        }
        
        return self.current_scores

    def get_sentiment_score(self):
        if not self.current_scores:
            return 0.0, 0.0
            
        p_neg = self.current_scores['negative']
        p_neu = self.current_scores['neutral']
        p_pos = self.current_scores['positive']
        
        score = p_pos - p_neg
        
        confidence = max(p_neg, p_neu, p_pos)
        
        return score, confidence
