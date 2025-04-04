#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "---> Installing requirements..."
pip install -r requirements.txt

echo "---> Downloading NLTK data..."
# Download NLTK data to the default Render location (/opt/render/nltk_data)
# app.py relies on NLTK finding this automatically in its search paths.
python -m nltk.downloader -d /opt/render/nltk_data stopwords wordnet omw-1.4

echo "---> Build finished!"