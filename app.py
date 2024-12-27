import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import bigrams, trigrams
from textblob import TextBlob
from PIL import Image
import numpy as np


"""
This is a simple Flask web application that allows users to upload a CSV file containing text data and generates a wordcloud.

EXAMPLE CALL:
word_freq = process_csv('uploads/sample.csv')
generate_wordcloud(word_freq, 'static/wordcloud.png', top_n=10)
"""
# NLTK Setup
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv'}
WORDCLOUD_FILENAME = 'wordcloud.png'
MASK_IMAGE_PATH = 'static/mask.png'  # Optional for shaped word clouds

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---- MODULE 1: Text Preprocessing ----
def preprocess_text(text):
    """Clean, tokenize, and lemmatize text."""
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return words


# ---- MODULE 2: Extract Actionable Words ----
def extract_actionable_words(words):
    """Filter for feedback-related words (e.g., 'improve', 'delay')."""
    actionable_terms = {'improve', 'fix', 'delay', 'broken', 'issue', 'missing'}
    return [word for word in words if word in actionable_terms]


# ---- MODULE 3: N-Gram Extraction ----
def extract_ngrams(words, n=2):
    """Generate bigrams or trigrams from words."""
    if n == 2:
        return [' '.join(bigram) for bigram in bigrams(words)]
    elif n == 3:
        return [' '.join(trigram) for trigram in trigrams(words)]
    return words


# ---- MODULE 4: Sentiment Analysis ----
def filter_by_sentiment(words, threshold=0):
    """Keep words with sentiment above the threshold."""
    return [word for word in words if TextBlob(word).sentiment.polarity > threshold]


# ---- MODULE 5: Generate Word Frequency ----
def compute_word_frequencies(words):
    """Count word frequencies and filter by frequency > 1."""
    word_freq = Counter(words)
    return {word: freq for word, freq in word_freq.items() if freq > 1}


# ---- MODULE 6: Generate Word Cloud ----
def generate_wordcloud(word_freq, output_path, top_n=15):
    """Generate and save a word cloud from the top N words."""
    try:
        # Mask for shaping the word cloud (optional)
        mask = None
        if os.path.exists(MASK_IMAGE_PATH):
            mask = np.array(Image.open(MASK_IMAGE_PATH))
        
        # Get the top N words
        top_words = dict(Counter(word_freq).most_common(top_n))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            mask=mask,
            contour_color='steelblue'
        ).generate_from_frequencies(top_words)

        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path)
        plt.close()

        return output_path
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None


# ---- MODULE 7: Process CSV ----
def process_csv(file_path, text_column='text'):
    try:
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            text_column = df.columns[0]

        text_data = ' '.join(df[text_column].astype(str))
        words = preprocess_text(text_data)

        # Apply NLP Enhancements
        # words += extract_ngrams(words, n=2)  # Add bigrams
        # words += extract_actionable_words(words)  # Add actionable terms
        # words = filter_by_sentiment(words, threshold=0.1)  # Keep positive words

        word_freq = compute_word_frequencies(words)
        return word_freq
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None


# ---- FLASK ROUTES ----
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            word_freq = process_csv(file_path)
            if not word_freq:
                return render_template('index.html', message="Failed to process CSV.")
            
            output_image_path = os.path.join(app.config['STATIC_FOLDER'], WORDCLOUD_FILENAME)
            generate_wordcloud(word_freq, output_image_path, top_n=20)
            os.remove(file_path)

            return redirect(url_for('show_wordcloud'))
    
    return render_template('index.html')


@app.route('/wordcloud')
def show_wordcloud():
    return render_template('result.html', image_file=WORDCLOUD_FILENAME)


@app.route('/static/<filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
