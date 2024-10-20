import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load pre-trained FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Function to map FinBERT's multi-class output to binary sentiment (positive/negative)
# Neutral (1) and Positive (2) are both considered as positive (1)
def get_finbert_sentiment_analysis(news):
    # Tokenize the input news
    inputs = tokenizer(news, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()

    # The sentiment classes in FinBERT are: 0 -> negative, 1 -> neutral, 2 -> positive
    # Map it to binary: sentiment == 0 -> negative (-1), sentiment == 1 or 2 -> positive (1)
    predicted_class = np.argmax(probabilities)
    if predicted_class == 0:
        return -1  # Negative sentiment
    else:
        return 1  # Neutral or positive sentiment treated as positive


# Load the actual dataset
file_path = 'BTC_EngNews.csv'
data = pd.read_csv(file_path)

# Get the actual labels (assuming binary sentiment)
y_true = data['Sentiment'].values

# Initialize variables for prediction and timing
y_pred = []
start_time = time.time()  # Start time for total processing
total_news = len(data['Title'])  # Total number of rows

# Start the prediction loop
for i, news in enumerate(data['Title']):
    iter_start_time = time.time()  # Start time for the iteration

    # Get the sentiment prediction using FinBERT model
    sentiment = get_finbert_sentiment_analysis(news)
    y_pred.append(sentiment)

    # Calculate ETA
    iter_end_time = time.time()
    time_per_iteration = iter_end_time - iter_start_time
    remaining_news = total_news - (i + 1)
    eta = remaining_news * time_per_iteration

    # Print progress and ETA
    print(f"Processed {i + 1}/{total_news} articles. ETA: {eta / 60:.2f} minutes.")

# Convert y_pred to a numpy array for comparison
y_pred = np.array(y_pred)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Output the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Create a DataFrame for the actual and predicted values along with the news
results_df = pd.DataFrame({
    'News': data['Title'],  # Include the news articles
    'Actual': y_true,
    'Predicted': y_pred
})

# Save the results to a CSV file
results_csv_path = 'finbert_sentiment_analysis_results.csv'
results_df.to_csv(results_csv_path, index=False)

# Output the total time taken and the path to the saved CSV
total_time = time.time() - start_time
print(f"Total time taken: {total_time / 60:.2f} minutes.")
print(f"Results saved to: {results_csv_path}")
