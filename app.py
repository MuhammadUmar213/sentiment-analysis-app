from transformers import pipeline
import gradio as gr

# Load Hugging Face sentiment analysis model
sentiment = pipeline("sentiment-analysis")

# Define function for prediction
def analyze_sentiment(text):
    result = sentiment(text)[0]
    label = result['label']
    score = round(result['score'], 3)
    return f"Sentiment: {label} (Confidence: {score})"

# Gradio UI setup
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analyzer",
    description="Enter a sentence to see the sentiment."
)

# Launch the app
interface.launch(share=True, debug=True)



