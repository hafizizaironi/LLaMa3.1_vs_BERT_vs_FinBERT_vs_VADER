import pandas as pd
import re

# Load the CSV file
btc_news_file = 'BTC_EngNews_Jan-Jul2022_Compiled.csv'
btc_news_df = pd.read_csv(btc_news_file)

# Function to clean the titles
def clean_title(title):
    # Remove any non-ASCII characters
    cleaned_title = re.sub(r'[^\x00-\x7F]+', '', title)
    # Replace unnecessary whitespace
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
    return cleaned_title

# Apply the cleaning function to the 'Title' column
btc_news_df['Title'] = btc_news_df['Title'].apply(clean_title)

# Save the cleaned dataframe back to CSV
cleaned_file = 'BTC_EngNews_Jan-Jul2022_Cleaned.csv'
btc_news_df.to_csv(cleaned_file, index=False)

print(f"Cleaned file saved at: {cleaned_file}")
