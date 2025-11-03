# SEO Content Quality & Duplicate Detector

This Streamlit app performs SEO content analysis, feature extraction, duplicate detection, and trains a quality model.

## Files in this folder
- `app.py` : The main Streamlit application.
- `requirements.txt` : Python dependencies.
- `README.md` : This file.

## How to run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place your dataset in one of these locations:
   - `/mnt/data/data.csv`
   - `./data.csv`
   The CSV must contain columns: `url` and `html_content`.

3. Run the app:
   ```
   streamlit run app.py
   ```

The app will create `extracted_content.csv` and `quality_model.pkl` as part of the workflow when you run the relevant steps inside the app.