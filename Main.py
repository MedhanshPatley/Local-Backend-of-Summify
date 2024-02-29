from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import json
import joblib
import pickle

app = Flask(__name__)
CORS(app)

# Your existing preprocessing functions

# Load your pre-trained classification model and tokenizer
# Assuming you have pre-trained models named 'model5.h5' and 'tokenizer.pkl'
# Adjust these as needed

# Load the saved model
loaded_model = load_model("model5.h5")

with open("tokenizer.pkl", "rb") as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

# Load MultiLabelBinarizer (if needed)
loaded_mlb = joblib.load("mlb5.pkl")

# Function for preprocessing text
def text_preprocessing(text):
    # Remove non-ASCII characters
    text = ''.join([c if ord(c) < 128 else ' ' for c in text])

    # Remove Roman numerals using a regular expression
    text = re.sub(r'\b[IVXLCDM]+(?:\s+[ivxlcdm]+)?\b', '', text)  # Remove lowercase Roman numerals

    # Convert to lowercase
    text = text.lower()
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.replace('’', '')
    text = text.replace('\n', ' ').replace('-', ' ')
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r',+', ',  ,', text)

    # Tokenize the text and remove stopwords
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    text = ' '.join(words)

    return text

# Function for preprocessing paragraphs
def preprocess_paragraphs(paragraphs):
    preprocessed_paragraphs = []
    for paragraph in paragraphs:
        preprocessed_paragraph = [text_preprocessing(line) for line in paragraph]
        preprocessed_paragraphs.append(preprocessed_paragraph)
    return preprocessed_paragraphs

# Function for splitting text into paragraphs
def split_into_paragraphs(text_list, lines_per_paragraph=8):
    paragraphs = []

    for text in text_list:
        current_paragraph = []
        lines_in_current_paragraph = 0

        for line in text.split("\n"):
            current_paragraph.append(line)
            lines_in_current_paragraph += 1

            if lines_in_current_paragraph == lines_per_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
                lines_in_current_paragraph = 0

        # Add the remaining lines in the current paragraph to the list of paragraphs.
        if current_paragraph:
            paragraphs.append(current_paragraph)

    return paragraphs

# Function to predict labels for text
def predict_labels_for_text(user_text):
    preprocessed_paragraphs = preprocess_paragraphs(split_into_paragraphs(user_text))

    # Predict labels for each paragraph
    predicted_labels = []
    for paragraph in preprocessed_paragraphs:
        preprocessed_text = ' '.join(paragraph)
        text_sequence = loaded_tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(text_sequence, maxlen=200)

        paragraph_labels_bin = loaded_model.predict(padded_sequence)
        paragraph_labels = loaded_mlb.inverse_transform(paragraph_labels_bin > 0.7)

        # Collect labels for the paragraph and flatten the list
        flat_labels = [label for labels in paragraph_labels for label in labels]
        predicted_labels.append(flat_labels)

    return predicted_labels

@app.route('/')
def index():
    return render_template('index.html')  # Display the HTML form

# Define an API endpoint for text classification
@app.route('/classify', methods=['POST'])
def classify_text():
    # Get the text input from the request sent by your mobile app
    data = request.get_json()
    user_text = data.get('user_text', '')  # Assuming the input field is named 'user_text'

    # Call your prediction function with the user's text
    predicted_labels = predict_labels_for_text(user_text)

    # Return the predicted labels as a JSON response
    return jsonify({'labels': predicted_labels[0]})

if __name__ == '__main__':
    app.run(host='192.168.1.15',port=5000)