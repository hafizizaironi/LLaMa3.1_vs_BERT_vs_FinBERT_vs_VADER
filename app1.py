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
    prompt = f"News: {news}\nProvide only the sentiment score for the news that is either -1 (negative) or 1 (positive). No other text needed"

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
                     "content": "You are a sentiment analysis expert. Analyze the given news text and provide a sentiment score (-1 for negative,1 for positive). No other text needed"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
        )
        response_json = response.json()
        content = response_json['choices'][0]['message']['content'].strip()

        # Ensure the content is either 1 or -1
        if content == '1':
            return 1
        elif content == '-1':
            return -1
        else:
            try:
                # Try converting to float and map ranges
                content_float = float(content)
                if -1 <= content_float < 0:
                    return -1
                elif 0 <= content_float <= 1:
                    return 1
                else:
                    return None
            except ValueError:
                return None
    except Exception as e:
        print(f"Error in API request: {str(e)}")
        return None


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
    sentiment = get_sentiment_analysis(news)
    if sentiment is not None:
        y_pred.append(sentiment)
    else:
        # In case of error or no prediction, assume a default value (e.g., -1)
        y_pred.append(-1)

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
results_csv_path = 'sentiment_analysis_results_full1.csv'
results_df.to_csv(results_csv_path, index=False)

# Output the total time taken and the path to the saved CSV
total_time = time.time() - start_time
print(f"Total time taken: {total_time / 60:.2f} minutes.")
print(f"Results saved to: {results_csv_path}")
