from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from text_pipeline import TextPipeline
import numpy as np
import time

dataset = [
    ("I am absolutely thrilled with the results of this project.", "positive"),
    ("The new feature works beautifully and saves me so much time.", "positive"),
    ("This is the best experience I have had in a long time.", "positive"),
    ("Great job on the presentation, it was very clear.", "positive"),
    ("I love how fast and responsive the application is.", "positive"),
    ("Not bad at all, actually quite impressive.", "positive"),
    ("I am very disappointed with the poor quality of this item.", "negative"),
    ("This software is full of bugs and crashes constantly.", "negative"),
    ("I hate how long it takes to process a simple request.", "negative"),
    ("Terrible customer service, they were completely unhelpful.", "negative"),
    ("The update ruined the interface and made it unusable.", "negative"),
    ("I guess it could have been worse, but it was still a waste of time.", "negative"),
    ("The package arrived on Tuesday as scheduled.", "neutral"),
    ("The application has a dark mode and a light mode.", "neutral"),
    ("I received the document and will review it later.", "neutral"),
    ("The meeting is scheduled for 3 PM tomorrow.", "neutral"),
    ("The color of the button is blue.", "neutral"),
    ("It is what it is.", "neutral"),
    ("Oh brilliant, another update that breaks everything.", "negative"), 
    ("Yeah right, like that's ever going to happen.", "negative"),
    ("I literally could not care less about this.", "neutral"),
]

pipeline = TextPipeline()
pipeline.load_model()

y_true = []
y_pred = []

start_time = time.time()

for text, true_label in dataset:
    scores = pipeline.analyze(text)
    pred_label = max(scores, key=scores.get)
    y_true.append(true_label)
    y_pred.append(pred_label)

end_time = time.time()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print(classification_report(y_true, y_pred))

with open('metrics_output.txt', 'w') as f:
    f.write(f"{accuracy*100:.2f},{precision*100:.2f},{recall*100:.2f},{f1*100:.2f}")
