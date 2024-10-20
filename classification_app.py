import time
import pandas as pd
import numpy as np
import requests

# LM Studio setup
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

# Function to interact with the LM Studio API and get sentiment analysis
def get_sentiment_analysis(news):
    prompt = f"News: {news}\nProvide a sentiment score between -1 and 1 (e.g., -0.5 for negative, 0.5 for positive, etc.) with no other text."

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
                     "content": "You are a sentiment analysis agent. Analyze the given news text and provide a sentiment score as a floating-point number between -1 (very negative) and 1 (very positive)."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
        )
        response_json = response.json()
        content = response_json['choices'][0]['message']['content'].strip()

        # Return the content directly, assuming it is a valid score
        try:
            return float(content)  # Try converting to float for a sentiment score
        except ValueError:
            return None  # If the value cannot be converted to a float, return None
    except Exception as e:
        print(f"Error in API request: {str(e)}")
        return None

# Load the new dataset for prediction (CSV file)
file_path = 'BTC_EngNews_Jan-Jul2022.csv'
data = pd.read_csv(file_path)

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
        # In case of error or no prediction, assume a default value (e.g., 0 for neutral)
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

# Create a DataFrame for the news and predicted values
results_df = pd.DataFrame({
    'News': data['Title'],  # Include the news articles
    'Predicted Sentiment': y_pred
})

# Save the results to a CSV file
results_csv_path = 'sentiment_analysis_results_new_2022.csv'
results_df.to_csv(results_csv_path, index=False)

# Output the total time taken and the path to the saved CSV
total_time = time.time() - start_time
print(f"Total time taken: {total_time / 60:.2f} minutes.")
print(f"Results saved to: {results_csv_path}")
