import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt


# Initialize NLTK stopwords
nltk.download('stopwords')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv'}
WORDCLOUD_FILENAME = 'wordcloud.png'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure upload and static directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_csv(file_path, text_column='text'):
    """
    Process the CSV file to compute word frequencies.

    Parameters:
    - file_path: Path to the uploaded CSV file.
    - text_column: Name of the column containing text data.

    Returns:
    - word_freq: A dictionary with words as keys and their frequencies as values.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the specified text column exists
        if text_column not in df.columns:
            # If not, use the first column
            text_column = df.columns[0]
            print(f"Column '{text_column}' not found. Using first column instead.")

        # Concatenate all text data
        text_data = ' '.join(df[text_column].astype(str))

        # Clean the text (remove punctuation, numbers, etc.)
        text_data = re.sub(r'[^A-Za-z\s]', '', text_data)
        text_data = text_data.lower()

        # Split into words
        words = text_data.split()

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Compute word frequencies
        word_freq = Counter(words)

        return word_freq
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None

def generate_wordcloud(word_freq, output_path):
    """
    Generate a word cloud image from word frequencies.

    Parameters:
    - word_freq: A dictionary with words as keys and their frequencies as values.
    - output_path: Path to save the generated word cloud image.

    Returns:
    - output_path: Path where the image is saved.
    """
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        print(wordcloud)
        # Display the generated image
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)

        # Save the image to the static folder
        plt.savefig(output_path)
        plt.close()

        return output_path
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message="No file part in the request.")
        file = request.files['file']
        # If user does not select file, browser may submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message="No file selected for uploading.")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the CSV and generate word cloud
            word_freq = process_csv(file_path)
            if word_freq is None:
                return render_template('index.html', message="Failed to process the uploaded CSV.")

            output_image_path = os.path.join(app.config['STATIC_FOLDER'], WORDCLOUD_FILENAME)
            print(output_image_path)
            print(word_freq)
            print("Generating word cloud...")
            generate_wordcloud(word_freq, output_image_path)

            # Remove the uploaded file after processing
            os.remove(file_path)

            return redirect(url_for('show_wordcloud'))
        else:
            return render_template('index.html', message="Allowed file type is CSV.")
    return render_template('index.html')

@app.route('/wordcloud')
def show_wordcloud():
    return render_template('result.html', image_file=WORDCLOUD_FILENAME)

# Optional: Serve the wordcloud image directly
@app.route('/static/<filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    # Run the app in debug mode. Disable debug in production.
    app.run(debug=True)
