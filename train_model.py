# File: train_model.py (Modified for 3-Class Classification)

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NLTK Data Check/Download ---
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('test')
    logging.info("NLTK data ('stopwords', 'wordnet', 'omw-1.4') seems available.")
except LookupError:
    logging.error("*"*60)
    logging.error("FATAL: NLTK data ('stopwords', 'wordnet', 'omw-1.4') not found.")
    logging.error("Please run the NLTK download commands in your terminal:")
    logging.error("python -c \"import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')\"")
    logging.error("*"*60)
    exit(1)

# --- Configuration ---
# Dataset 1 Configuration (Original - MODIFY IF YOURS IS DIFFERENT)
DATASET_PATH_1 = 'HateSpeechDataset.csv'
TEXT_COLUMN_1 = 'Content'       # Text column name in dataset 1
LABEL_COLUMN_1 = 'Label'        # Label column name in dataset 1
# --- MAPPING FOR DATASET 1 (Binary to 3-Class) ---
# Map original 0 (Not Hate) to 2 (Neither)
# Map original 1 (Hate) to 0 (Hate Speech)
label_map_df1 = {
    0: 2, # Not Hate -> Neither
    1: 0  # Hate -> Hate Speech
}

# Dataset 2 Configuration (From hate.csv image)
DATASET_PATH_2 = 'hate.csv'
TEXT_COLUMN_2 = 'tweet'         # Text column name in hate.csv
LABEL_COLUMN_2 = 'class'        # Label column name in hate.csv
# --- MAPPING FOR DATASET 2 (Original 3-Class to Target 3-Class) ---
# Map original 0 (Hate) to 0 (Hate Speech)
# Map original 1 (Offensive) to 1 (Offensive Language)
# Map original 2 (Neither) to 2 (Neither)
label_map_df2 = {
    0: 0,  # Hate -> Hate Speech
    1: 1,  # Offensive -> Offensive Language
    2: 2   # Neither -> Neither
}

# Standard column names to use internally
STD_TEXT_COLUMN = 'text_content'
STD_LABEL_COLUMN = 'temp_label' # Temporary label column before final target

# Final Target variable name and Class names for the MULTI-CLASS model
FINAL_TARGET_COLUMN = 'target'
# ** IMPORTANT: Order must match the numerical labels 0, 1, 2 **
TARGET_CLASS_NAMES = ['Hate Speech', 'Offensive Language', 'Neither']
# Numerical mapping: 0 = Hate Speech, 1 = Offensive Language, 2 = Neither

# Model/Vectorizer output filenames
MODEL_FILENAME = 'hate_speech_model_3class.pkl' # New name for 3-class model
VECTORIZER_FILENAME = 'tfidf_vectorizer_3class.pkl' # New name for vectorizer

# --- Helper Function to Load and Standardize (Same as before) ---
def load_and_standardize(filepath, text_col, label_col, std_text_col, std_label_col, label_map=None):
    """Loads CSV, renames columns, selects relevant ones, and applies label mapping."""
    logging.info(f"Processing dataset: {filepath}")
    if not os.path.exists(filepath):
        logging.error(f"Error: Dataset file not found at '{filepath}'.")
        return None
    try:
        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            logging.warning(f"UTF-8 decoding failed for {filepath}, trying latin-1...")
            df = pd.read_csv(filepath, encoding='latin-1')
        logging.info(f"Loaded {len(df)} rows from {filepath}.")

        if text_col not in df.columns:
            logging.error(f"Text column '{text_col}' not found in {filepath}. Columns: {df.columns.tolist()}")
            return None
        if label_col not in df.columns:
            logging.error(f"Label column '{label_col}' not found in {filepath}. Columns: {df.columns.tolist()}")
            return None

        df = df.rename(columns={text_col: std_text_col, label_col: std_label_col})
        df = df[[std_text_col, std_label_col]]
        logging.info(f"Standardized columns for {filepath} to: {df.columns.tolist()}")

        if label_map:
            original_labels = df[std_label_col].unique()
            logging.info(f"Applying label map to {filepath}. Original labels found: {original_labels}")
            df[std_label_col] = df[std_label_col].map(label_map)
            mapped_labels = df[std_label_col].unique()
            logging.info(f"Labels after mapping: {mapped_labels}")
            rows_before_drop = len(df)
            df.dropna(subset=[std_label_col], inplace=True) # Drop rows where mapping failed
            rows_after_drop = len(df)
            if rows_before_drop > rows_after_drop:
                 logging.warning(f"Dropped {rows_before_drop - rows_after_drop} rows from {filepath} due to failed label mapping.")
        return df

    except Exception as e:
        logging.exception(f"Error processing dataset '{filepath}': {e}")
        return None

# --- Load and Prepare Data ---
df1 = load_and_standardize(DATASET_PATH_1, TEXT_COLUMN_1, LABEL_COLUMN_1,
                           STD_TEXT_COLUMN, STD_LABEL_COLUMN, label_map=label_map_df1)

df2 = load_and_standardize(DATASET_PATH_2, TEXT_COLUMN_2, LABEL_COLUMN_2,
                           STD_TEXT_COLUMN, STD_LABEL_COLUMN, label_map=label_map_df2)

if df1 is None or df2 is None:
    logging.error("Failed to load or standardize one or both datasets. Exiting.")
    exit(1)

# Combine DataFrames
logging.info("Combining standardized datasets...")
df_combined = pd.concat([df1, df2], ignore_index=True)
logging.info(f"Combined dataset created with {len(df_combined)} rows initially.")
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)


# --- Clean NAs and Filter FINAL target labels (0, 1, 2) ---
logging.info("Cleaning NAs and filtering final labels (0, 1, 2) in combined data...")
initial_rows_combined = len(df_combined)
df_combined.dropna(subset=[STD_TEXT_COLUMN, STD_LABEL_COLUMN], inplace=True)

# Ensure only target labels 0, 1, 2 remain
df_combined[STD_LABEL_COLUMN] = pd.to_numeric(df_combined[STD_LABEL_COLUMN], errors='coerce')
df_combined.dropna(subset=[STD_LABEL_COLUMN], inplace=True)
df_combined = df_combined[df_combined[STD_LABEL_COLUMN].isin([0.0, 1.0, 2.0])] # Keep 0, 1, 2

# Convert the final label column to integer
try:
    df_combined[FINAL_TARGET_COLUMN] = df_combined[STD_LABEL_COLUMN].astype(int)
    logging.info(f"Final target column '{FINAL_TARGET_COLUMN}' created.")
except ValueError as e:
    logging.error(f"Could not convert final labels to integers after filtering. Error: {e}")
    logging.error(f"Unique values remaining in '{STD_LABEL_COLUMN}': {df_combined[STD_LABEL_COLUMN].unique()}")
    exit(1)

rows_after_cleaning = len(df_combined)
removed_count = initial_rows_combined - rows_after_cleaning
logging.info(f"Rows after cleaning NAs and filtering for valid target labels (0, 1, 2): {rows_after_cleaning}")
if removed_count > 0: logging.warning(f"Removed {removed_count} rows during cleaning/filtering.")
if rows_after_cleaning == 0:
     logging.error("Error: No data remaining after cleaning and filtering labels.")
     exit(1)

# Final check of 3-class label distribution
logging.info("\nCombined Final Label distribution (0=Hate, 1=Offensive, 2=Neither):")
print(df_combined[FINAL_TARGET_COLUMN].value_counts(normalize=True).sort_index())


# --- Text Preprocessing (Identical function as before) ---
logging.info("Preprocessing text on combined dataset...")
lemmatizer = WordNetLemmatizer()
stop_words_list = stopwords.words('english')

def clean_text(text):
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
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words_list])
    return cleaned_text

df_combined['cleaned_text'] = df_combined[STD_TEXT_COLUMN].apply(clean_text)
logging.info("Text preprocessing complete.")
print("\nSample of original vs cleaned text:")
print(df_combined[[STD_TEXT_COLUMN, 'cleaned_text']].head())


# --- Feature Engineering (TF-IDF - Identical) ---
logging.info("Applying TF-IDF Vectorizer on combined data...")
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df_combined['cleaned_text'])
y = df_combined[FINAL_TARGET_COLUMN] # Use the final 3-class target column
logging.info(f"TF-IDF matrix shape: {X_tfidf.shape}")


# --- Train-Test Split (Stratify on 3 classes) ---
logging.info("Splitting combined data into train and test sets (stratified)...")
try:
    # Check if all 3 classes have at least 2 samples for stratification
    min_class_count = y.value_counts().min()
    if min_class_count < 2:
         logging.warning(f"The smallest class has only {min_class_count} samples. Stratification might behave unexpectedly or fail if < 2.")
         # Consider alternative splitting if a class has < 2 samples. Proceeding with caution.
         if min_class_count < 1:
              raise ValueError("At least one class has zero samples after cleaning. Cannot train.")

    test_size = 0.2
    # Adjust test_size if any class has very few samples
    smallest_allowed_test_count = 1 # Need at least 1 test sample per class if possible
    required_test_size_per_class = smallest_allowed_test_count / y.value_counts()
    max_required_test_size = required_test_size_per_class.max()

    if test_size < max_required_test_size and max_required_test_size < 1.0:
         # Adjust test size upwards if needed, but cap it (e.g., at 0.5)
         test_size = min(0.5, max_required_test_size * 1.1) # Add slight buffer
         logging.warning(f"Adjusting test_size to ~{test_size:.3f} to ensure all classes representation in test set.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y,
        test_size=test_size,
        random_state=42,
        stratify=y # Stratify based on the 3-class target
    )
    logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    logging.info("Train set label distribution (%):\n" + str(y_train.value_counts(normalize=True).sort_index()))
    logging.info("Test set label distribution (%):\n" + str(y_test.value_counts(normalize=True).sort_index()))

except ValueError as e:
    logging.exception(f"Error during train/test split: {e}")
    logging.error("Check the final combined label distribution:")
    print(y.value_counts().sort_index())
    exit(1)


# --- Model Training (Logistic Regression supports multi-class OVR by default) ---
logging.info("Training Logistic Regression model for 3-class classification...")
# class_weight='balanced' helps with imbalanced classes in multi-class settings too
model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    solver='liblinear', # Works for multi-class (OVR)
    max_iter=1000
    # multi_class='ovr' # Explicitly set One-vs-Rest (often default)
    # multi_class='multinomial' # Alternative, might be better/worse
)
model.fit(X_train, y_train)
logging.info("Model training complete.")


# --- Evaluation (Using 3-class names) ---
logging.info("Evaluating model on combined test set...")
try:
    y_pred = model.predict(X_test)
    logging.info("\nClassification Report (0=Hate, 1=Offensive, 2=Neither):")
    report = classification_report(y_test, y_pred, target_names=TARGET_CLASS_NAMES, zero_division=0)
    print(report)
    # Optional: Save report
    # with open("classification_report_3class.txt", "w") as f:
    #     f.write(report)

except Exception as e:
    logging.exception(f"Error during evaluation: {e}")


# --- Save Model and Vectorizer ---
logging.info(f"Saving model to '{MODEL_FILENAME}' and vectorizer to '{VECTORIZER_FILENAME}'...")
try:
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(vectorizer, VECTORIZER_FILENAME)
    logging.info("Model and vectorizer saved successfully.")
except Exception as e:
    logging.exception(f"Error saving model/vectorizer: {e}")

logging.info("\n--- Training script finished (3-class model) ---")