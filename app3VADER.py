import time
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()


# Function to get sentiment using VADER
def get_sentiment_vader(news):
    # Get the polarity scores
    sentiment_scores = sid.polarity_scores(news)

    # The compound score is a normalized score between -1 (most negative) and 1 (most positive)
    compound_score = sentiment_scores['compound']

    # Map the compound score to binary sentiment (-1 for negative, 1 for positive)
    if compound_score >= 0:
        return 1  # Positive sentiment
    else:
        return -1  # Negative sentiment


# Load the actual dataset
file_path = 'BTC_EngNews.csv'
data = pd.read_csv(file_path)

# Get the actual labels
y_true = data['Sentiment'].values

# Initialize variables for prediction and timing
y_pred = []
start_time = time.time()  # Start time for total processing
total_news = len(data['Title'])  # Total number of rows

# Start the prediction loop
for i, news in enumerate(data['Title']):
    iter_start_time = time.time()  # Start time for the iteration

    # Get the sentiment prediction
    sentiment = get_sentiment_vader(news)
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
results_csv_path = 'sentiment_analysis_results_vader.csv'
results_df.to_csv(results_csv_path, index=False)

# Output the total time taken and the path to the saved CSV
total_time = time.time() - start_time
print(f"Total time taken: {total_time / 60:.2f} minutes.")
print(f"Results saved to: {results_csv_path}")
