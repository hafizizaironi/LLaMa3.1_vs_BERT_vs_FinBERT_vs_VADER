from flask import Flask, render_template, request, jsonify
import openai
import markdown

app = Flask(__name__)

agent_context = ""


# Configure OpenAI with the local LM Studio setup
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"

def get_sentiment_analysis(news):
    # Prompt to the sentiment model
    prompt = f"News: {news}\nProvide the sentiment score for the news that are only either -1 (negative) or 1 (positive) give simple breakdown."

    # Send the prompt to LM Studio API via the OpenAI client
    try:
        response = openai.ChatCompletion.create(
            model="llama-3.2-1b-instruct",
            messages=[
                {"role": "system", "content": "You are a highly skilled sentiment analysis agent on cryptocurrencies. Your task is to analyze text data and determine the emotional tone expressed within it. Utilize advanced natural language processing (NLP) techniques and machine learning algorithms to provide a sentiment score that is either -1 (negative) or 1 (positive) only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message['content']

        # Format the response with Markdown and return it as HTML-safe
        formatted_content = markdown.markdown(content)
        return formatted_content
    except Exception as e:
        return f"Error in API request: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    news = request.form['news']

    # Get sentiment analysis from LM Studio
    sentiment = get_sentiment_analysis(news)

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
