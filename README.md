# SEO Content Quality & Duplicate Detector

A powerful Streamlit-based tool for comprehensive SEO content analysis. Combines machine learning-based quality scoring, duplicate content detection, and readability analysis to help content teams maintain high standards and identify optimization opportunities.

## Quick Start

```bash
git clone https://github.com/Bull9016/seo-content-detector
cd seo-content-detector
pip install -r requirements.txt
streamlit run app.py
```

## Setup Instructions

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   - Place your dataset as `data.csv` in the project root
   - Required columns: `url`, `html_content`
   - Max file size: 100MB

3. **Run Analysis**:
   ```bash
   streamlit run app.py
   ```

## Key Technical Decisions

1. **Library Selection**:
   - `BeautifulSoup4`: Robust HTML parsing with customizable content extraction
   - `sentence-transformers`: State-of-the-art text embeddings for similarity detection
   - `scikit-learn`: Production-ready ML implementation for quality scoring
   - `textstat`: Comprehensive readability metrics

2. **HTML Parsing Strategy**:
   - Prioritize main content areas (`<article>`, `<main>`, `#content`)
   - Filter short paragraphs (<50 chars) to reduce noise
   - Clean navigation, headers, footers for better text quality
   - Preserve sentence structure for readability analysis

3. **Content Quality Model**:
   - Random Forest Classifier with 5 key features
   - Features: word count, sentence complexity, readability scores
   - Three-class labeling: High/Medium/Low quality
   - 70/30 train-test split with stratification

4. **Duplicate Detection**:
   - Sentence-BERT embeddings (all-MiniLM-L6-v2)
   - Cosine similarity with 0.85 threshold
   - Optimized for both exact and near-duplicate detection

## Results Summary

### Model Performance
- Quality Classification Accuracy: 78%
- F1-Score: 0.77 (weighted average)
- Baseline Improvement: +14% over word-count only

### Content Analysis
- Duplicate Detection: Identifies both exact and near-duplicates (>85% similarity)
- Quality Distribution: 25% High, 45% Medium, 30% Low quality
- Average Processing Time: ~2 seconds per URL

## Limitations

1. **Content Extraction**:
   - May miss content in highly dynamic JavaScript-rendered pages
   - Assumes standard HTML structure for main content areas

2. **Language Support**:
   - Optimized for English content
   - Readability scores most accurate for English text

3. **Resource Requirements**:
   - Initial embedding computation can be memory-intensive
   - Dataset size limited to 100MB for web interface

## Live Demo

Access the live demo at: [SEO Content Analyzer Demo](https://daawhpkpszaag7t3qdhhqv.streamlit.app/)
