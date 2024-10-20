import time
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# LM Studio setup
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

# Function to interact with the LM Studio API and get sentiment analysis
def get_sentiment_analysis(news):
    prompt = f"News: {news}\nProvide only the sentiment score for the news as a number between -1.0 to -0.3 (most negative) ,-0.2 to 0.2 (neutral) and 0.3 to 1 (most positive) .No other text needed"

    try:
        response = requests.post(
            f"{LM_STUDIO_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {LM_STUDIO_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.2-1b-instruct",
                "messages": [
                    {"role": "system",
                     "content": "You are a sentiment analysis agent. Analyze the given news text and provide a sentiment score between -1 and 1."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
        )
        response_json = response.json()
        content = response_json['choices'][0]['message']['content'].strip()

        try:
            # Convert the content to a float value
            sentiment_score = float(content)

            # Classify based on ranges
            if -1 <= sentiment_score <= -0.3:
                return -1  # Negative
            elif -0.2 <= sentiment_score <= 0.2:
                return 0   # Neutral
            elif 0.3 <= sentiment_score <= 1:
                return 1   # Positive
            else:
                return None  # Out of expected range
        except ValueError:
            return None  # If unable to convert to float
    except Exception as e:
        print(f"Error in API request: {str(e)}")
        return None

# Load the new dataset for prediction (CSV file)
file_path = 'BTC_EngNews_classified.csv'
data = pd.read_csv(file_path)

# Assuming the true labels are in a column named 'Classified Sentiment'
y_true = data['Classified Sentiment']

# Initialize variables for prediction and timing
y_pred = []
start_time = time.time()  # Start time for total processing
total_news = len(data['Title'])  # Total number of rows

# Start the prediction loop
for i, news in enumerate(data['Title']):
    iter_start_time = time.time()  # Start time for the iteration

    # Get the sentiment prediction
    sentiment = get_sentiment_analysis(news)
    if sentiment is not None:
        y_pred.append(sentiment)
    else:
        # In case of error or no prediction, assume a default value (e.g., 0 as neutral)
        y_pred.append(0)

    # Calculate ETA
    iter_end_time = time.time()
    time_per_iteration = iter_end_time - iter_start_time
    remaining_news = total_news - (i + 1)
    eta = remaining_news * time_per_iteration

    # Print progress and ETA
    print(f"Processed {i + 1}/{total_news} articles. ETA: {eta / 60:.2f} minutes.")

# Convert y_pred to a numpy array
y_pred = np.array(y_pred)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print out the performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Create a DataFrame for the news and predicted values
results_df = pd.DataFrame({
    'News': data['Title'],  # Include the news articles
    'Predicted Sentiment': y_pred,
    'True Sentiment': y_true
})

# Save the results to a CSV file
results_csv_path = 'sentiment_analysis_results_with_metrics.csv'
results_df.to_csv(results_csv_path, index=False)

# Output the total time taken and the path to the saved CSV
total_time = time.time() - start_time
print(f"Total time taken: {total_time / 60:.2f} minutes.")
print(f"Results saved to: {results_csv_path}")
