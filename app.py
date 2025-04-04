# File: app.py (Modified for 3-Class Prediction)

import io
import os
import re
import string
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from werkzeug.utils import secure_filename
import numpy as np
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NLTK Default Path ---
logging.info(f"NLTK will search default paths, including: {nltk.data.path}")

# --- Configuration & Constants ---
# ** Use the filenames from the 3-class training script **
MODEL_FILENAME = 'hate_speech_model_3class.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer_3class.pkl'

# ** Updated labels for 3 classes (MUST match TARGET_CLASS_NAMES order in training) **
# 0: Hate Speech, 1: Offensive Language, 2: Neither
SENTIMENT_LABELS = {
    0: 'Hate Speech',
    1: 'Offensive Language',
    2: 'Neither'
}
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
ASSUMED_TEXT_COLUMN = 'Content' # Column name expected in UPLOADED files
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', b'_fallback_secret_')

# --- Initialize NLTK Components ---
lemmatizer_instance = None
stop_words_set = set()
nltk_data_loaded = False
try:
    stop_words_set = set(stopwords.words('english'))
    lemmatizer_instance = WordNetLemmatizer()
    lemmatizer_instance.lemmatize('tests')
    logging.info(f"NLTK check successful: Found {len(stop_words_set)} stopwords and lemmatizer initialized.")
    nltk_data_loaded = True
except Exception as e_nltk:
    logging.exception(f"FATAL ERROR during NLTK initialization: {e_nltk}")
    logging.error("Check NLTK data installation (stopwords, wordnet, omw-1.4).")
    # App might fail later if this doesn't load

# --- Load Model and Vectorizer ---
model = None
vectorizer = None
model_classes = None # Stores the classes the loaded model predicts (e.g., [0, 1, 2])
model_loaded = False
vectorizer_loaded = False
try:
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    vectorizer_path = os.path.join(script_dir, VECTORIZER_FILENAME)
    logging.info(f"Attempting to load model from: {model_path}")
    logging.info(f"Attempting to load vectorizer from: {vectorizer_path}")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model_loaded = True
        if hasattr(model, 'classes_'):
             model_classes = list(model.classes_)
             logging.info(f"Model loaded successfully. Detected model classes: {model_classes}")
             # ** Validate model classes match expected 3-class scheme **
             if sorted(model_classes) != [0, 1, 2]:
                 logging.warning(f"Loaded model classes {model_classes} DO NOT match expected [0, 1, 2]. Label mapping might be incorrect!")
        else:
             # If classes_ attribute missing, assume the order based on SENTIMENT_LABELS
             model_classes = sorted(SENTIMENT_LABELS.keys())
             logging.warning(f"Model object lacks 'classes_' attribute. Assuming order {model_classes} based on SENTIMENT_LABELS.")
    else:
         logging.error(f"Model file not found at '{model_path}'.")

    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        vectorizer_loaded = True
        logging.info("Vectorizer loaded successfully.")
    else:
        logging.error(f"Vectorizer file not found at '{vectorizer_path}'.")

except Exception as e_load:
    logging.exception(f"FATAL: Error loading 3-class model/vectorizer: {e_load}")


# --- Text Preprocessing Function (Identical to training) ---
def clean_text(text):
    if not nltk_data_loaded or lemmatizer_instance is None:
        logging.error("NLTK components not loaded. Returning original text.")
        return str(text)
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'<', '<', text)
    text = re.sub(r'>', '>', text)
    text = re.sub(r'"', '"', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = ' '.join([lemmatizer_instance.lemmatize(word) for word in text.split() if word not in stop_words_set])
    return cleaned_text

# --- Helper Function for File Type Check (Identical) ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def home():
    logging.info("Serving home page.")
    return render_template('index.html', prediction_text='', probabilities=None, submitted_text='', error_message=None, info_message=None)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received POST request to /predict.")
    prediction_text = ''
    probabilities_dict = None # Will store {Label: Score}
    error_message = None
    info_message = None
    submitted_text = request.form.get('text_input', '').strip()
    file = request.files.get('file_upload')

    # --- Critical Check ---
    if not model_loaded or not vectorizer_loaded:
        logging.error("3-Class Model or Vectorizer is not loaded. Cannot predict.")
        error_message = "Analysis service is temporarily unavailable (model loading failed)."
        return render_template('index.html', error_message=error_message), 503
    if not nltk_data_loaded:
         logging.warning("NLTK data failed to load earlier. Predictions might be inaccurate.")
         # info_message = "Warning: Text processing components may not be fully functional."

    # --- BRANCH 1: File Upload Processing ---
    if file and file.filename != '':
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                logging.info(f"Processing uploaded file: {filename}")
                df = None
                if filename.lower().endswith('.csv'):
                     try: df = pd.read_csv(file.stream)
                     except UnicodeDecodeError: file.stream.seek(0); logging.warning("UTF-8 failed for CSV, trying latin-1"); df = pd.read_csv(file.stream, encoding='latin-1')
                elif filename.lower().endswith(('.xlsx', '.xls')): df = pd.read_excel(file.stream, engine='openpyxl')

                if df is None: raise ValueError("Pandas failed to read file stream.")
                if ASSUMED_TEXT_COLUMN not in df.columns:
                    error_message = f"Required column '{ASSUMED_TEXT_COLUMN}' not found. Found: {', '.join(df.columns)}"
                    return render_template('index.html', error_message=error_message), 400

                results = []
                logging.info(f"Found {len(df)} rows to process in {filename}.")
                for index, row in df.iterrows():
                    original_text = row[ASSUMED_TEXT_COLUMN]
                    pred_label = "Skipped"
                    prob_hate, prob_offensive, prob_neither = None, None, None
                    cleaned_for_output = ""

                    if pd.isna(original_text) or not isinstance(original_text, str) or original_text.strip() == '':
                        pred_label = "Skipped (Invalid/Empty Input)"
                        logging.debug(f"Skipping row {index}: Invalid/Empty text.")
                    else:
                        try:
                            cleaned = clean_text(original_text)
                            vectorized = vectorizer.transform([cleaned])
                            prediction = model.predict(vectorized)[0] # Should be 0, 1, or 2
                            proba = model.predict_proba(vectorized)[0] # Should have 3 probabilities

                            pred_label = SENTIMENT_LABELS.get(prediction, f"Unknown ({prediction})")
                            cleaned_for_output = cleaned

                            # Get probabilities based on the model's actual class order
                            if model_classes is not None and len(proba) == len(model_classes):
                                prob_map = {cls_label: proba[i] for i, cls_label in enumerate(model_classes)}
                                prob_hate = prob_map.get(0, None) # 0 = Hate Speech
                                prob_offensive = prob_map.get(1, None) # 1 = Offensive Language
                                prob_neither = prob_map.get(2, None) # 2 = Neither
                            else:
                                logging.warning(f"Mismatch between proba length ({len(proba)}) and model_classes ({model_classes}) for row {index}.")

                            logging.debug(f"Processed row {index}: Pred={pred_label}, Probs=[H:{prob_hate:.2f}, O:{prob_offensive:.2f}, N:{prob_neither:.2f}]")

                        except Exception as e_row:
                            logging.exception(f"Error processing row {index} from file {filename}: {e_row}")
                            pred_label = f"Error ({type(e_row).__name__})"


                    results.append({
                        'Original_Text': original_text,
                        'Cleaned_Text': cleaned_for_output,
                        'Prediction': pred_label,
                        'Confidence_Hate_Speech': prob_hate,
                        'Confidence_Offensive_Lang': prob_offensive,
                        'Confidence_Neither': prob_neither
                    })

                logging.info(f"Finished processing {len(results)} rows from {filename}.")
                results_df = pd.DataFrame(results)
                output_buffer = io.BytesIO(); results_df.to_csv(output_buffer, index=False, encoding='utf-8', float_format='%.4f'); output_buffer.seek(0)
                logging.info(f"Prepared results CSV for download from file {filename}.")
                return send_file(output_buffer, mimetype='text/csv', as_attachment=True, download_name='multi_class_analysis_results.csv') # New download name

            except Exception as e_file:
                logging.exception(f"Error processing uploaded file '{filename}': {e_file}")
                error_message = f"An error occurred processing the file: {type(e_file).__name__}"
                return render_template('index.html', error_message=error_message), 500

        else: # Invalid file type
            logging.warning(f"Invalid file type uploaded: {file.filename}")
            error_message = "Invalid file type. Allowed types: CSV, XLSX, XLS."
            return render_template('index.html', error_message=error_message), 400

    # --- BRANCH 2: Single Text Input Processing ---
    elif submitted_text:
        try:
            logging.info(f"Processing single text input (length {len(submitted_text)}).")
            cleaned_input = clean_text(submitted_text)
            if not cleaned_input and submitted_text:
                info_message = "Input text became empty after cleaning. No analysis performed."
                return render_template('index.html', submitted_text=submitted_text, info_message=info_message), 200

            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]
            predicted_proba = model.predict_proba(vectorized_input)[0]

            prediction_text = SENTIMENT_LABELS.get(prediction, f"Unknown ({prediction})")
            probabilities_dict = {} # Store {Label_String: Formatted_Score}

            if model_classes is not None and len(predicted_proba) == len(model_classes):
                prob_map = {cls_label: predicted_proba[i] for i, cls_label in enumerate(model_classes)}
                # Populate dict using SENTIMENT_LABELS to ensure correct string keys
                for label_code, label_name in SENTIMENT_LABELS.items():
                    prob_val = prob_map.get(label_code, None)
                    if prob_val is not None:
                        probabilities_dict[label_name] = f"{prob_val:.1%}"
                    else:
                         probabilities_dict[label_name] = "N/A" # Should not happen if model classes are correct
                logging.info(f"Single text prediction: {prediction_text}, Probs: {probabilities_dict}")
            else:
                 logging.error(f"Mismatch between predicted_proba length ({len(predicted_proba)}) and model_classes ({model_classes}). Cannot display probabilities correctly.")
                 prediction_text = "Prediction Error" # Indicate an issue

        except Exception as e_single:
            logging.exception(f"Error during single text prediction: {e_single}")
            error_message = f"An error occurred during analysis: {type(e_single).__name__}"
            prediction_text = "" # Clear prediction on error

        status_code = 500 if error_message and not prediction_text else 200
        return render_template('index.html',
                               prediction_text=prediction_text,
                               probabilities=probabilities_dict,
                               submitted_text=submitted_text,
                               error_message=error_message,
                               info_message=info_message), status_code

    # --- BRANCH 3: No Input Provided ---
    else:
        logging.warning("Predict endpoint called with no text input or file upload.")
        error_message = "Please enter text or upload a file to analyze."
        return render_template('index.html', error_message=error_message), 400

# --- Run the App ---
if __name__ == '__main__':
    if model_loaded and vectorizer_loaded and nltk_data_loaded:
        port = int(os.environ.get('PORT', 5000))
        host = '0.0.0.0'
        is_debug = os.environ.get('FLASK_DEBUG', '0') == '1'
        logging.info(f"Starting Flask application (3-Class Model) on http://{host}:{port} (Debug Mode: {is_debug})")
        app.run(host=host, port=port, debug=is_debug)
    else:
        logging.error("*"*60)
        logging.error("FATAL: Flask app cannot start - critical components failed to load.")
        logging.error(f"Model Loaded: {model_loaded}, Vectorizer Loaded: {vectorizer_loaded}, NLTK Data Loaded: {nltk_data_loaded}")
        logging.error("Check logs for errors (e.g., file not found, NLTK issues).")
        logging.error(f"Ensure '{MODEL_FILENAME}' and '{VECTORIZER_FILENAME}' exist.")
        logging.error("*"*60)
        import sys; sys.exit(1)