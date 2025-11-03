"""
SEO Content Quality & Duplicate Detector - Streamlit App
Features:
- Data Processing & Parsing from data.csv (expects columns: url, html_content)
- Feature Engineering & Visualization
- Duplicate & Thin Content Detection
- Content Quality Scoring (RandomForestClassifier)
- Live URL Analyzer (scrapes a single URL and compares against dataset)

To run:
1. Install required packages:
   pip install -r requirements.txt
2. Place your dataset as /mnt/data/data.csv or in the same folder as this app.
3. Run:
   streamlit run app.py

This file is intentionally verbose with helpful comments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from bs4 import BeautifulSoup
import textstat
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
import joblib
import re
import warnings
warnings.filterwarnings("ignore")

# Ensure NLTK punkt is available
nltk.download('punkt')

# Paths
DATA_PATHS = [
    "/mnt/data/data.csv",
    "data.csv"
]
EXTRACTED_CSV = "extracted_content.csv"
MODEL_FILE = "quality_model.pkl"

st.set_page_config(layout="wide", page_title="SEO Content Quality & Duplicate Detector")

@st.cache_data
def load_dataset():
    # Try multiple locations for data.csv
    for p in DATA_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                st.session_state['data_path'] = p
                return df
            except Exception as e:
                st.error(f"Failed to read CSV at {p}: {e}")
    st.error("No data.csv found. Please upload the dataset or place it at /mnt/data/data.csv")
    return pd.DataFrame(columns=["url","html_content"])

def clean_text(text):
    """Clean and normalize text content."""
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove special characters but keep periods for sentence detection
        text = re.sub(r'[^a-z0-9\s\.]', '', text)
        return text
    except:
        return ""

def parse_html_to_text(html):
    """Extract and clean text content from HTML."""
    try:
        soup = BeautifulSoup(str(html), "html.parser")
        
        # Remove non-content tags
        for tag in soup(["script", "style", "nav", "footer", "header", 
                        "noscript", "svg", "form", "iframe", "aside"]):
            tag.decompose()
            
        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = clean_text(soup.title.string.strip())
        elif soup.find("h1"):
            title = clean_text(soup.find("h1").get_text(strip=True))
            
        # Try to find main content area
        main_content = None
        for tag in ["article", "main", "#content", ".content", "div.post"]:
            main_content = soup.select_one(tag)
            if main_content:
                break
                
        # Extract paragraphs
        if main_content:
            paragraphs = [p.get_text(separator=" ", strip=True) 
                         for p in main_content.find_all("p") 
                         if len(p.get_text(strip=True)) > 50]  # Filter short paragraphs
        else:
            paragraphs = [p.get_text(separator=" ", strip=True) 
                         for p in soup.find_all("p") 
                         if len(p.get_text(strip=True)) > 50]
            
        # If no paragraphs found, try direct text
        if not paragraphs:
            body = soup.get_text(separator=" ", strip=True)
        else:
            body = "\n\n".join(paragraphs)
            
        # Clean the extracted text
        body = clean_text(body)
        word_count = len(body.split())
        
        return title, body, word_count
    except Exception as e:
        st.warning(f"Error parsing HTML: {str(e)}")
        return "", "", 0

@st.cache_data
def process_and_extract(df):
    # Check if data already has the extracted content
    if all(col in df.columns for col in ["title", "body_text", "word_count"]):
        df.to_csv(EXTRACTED_CSV, index=False)
        return df
    
    # If not, extract from html_content
    results = []
    for i, row in df.iterrows():
        url = row.get("url", "")
        html = row.get("html_content", "")
        title, body, wc = parse_html_to_text(html)
        results.append({
            "url": url,
            "title": title,
            "body_text": body,
            "word_count": wc
        })
    extracted = pd.DataFrame(results)
    # Save extracted csv
    extracted.to_csv(EXTRACTED_CSV, index=False)
    return extracted


def load_extracted_csv():
    """Safely load the extracted CSV. Returns a DataFrame or None if file missing/empty/invalid."""
    try:
        if not os.path.exists(EXTRACTED_CSV):
            return None
        # empty-file check
        if os.path.getsize(EXTRACTED_CSV) == 0:
            return None
        df = pd.read_csv(EXTRACTED_CSV)
        if df is None or df.empty:
            return None
        return df
    except pd.errors.EmptyDataError:
        return None
    except Exception as e:
        # Avoid crashing the app for read issues; surface a message in the UI when appropriate
        try:
            st.error(f"Failed to read {EXTRACTED_CSV}: {e}")
        except Exception:
            pass
        return None

@st.cache_data
def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob."""
    try:
        analysis = TextBlob(str(text))
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity,
            'sentiment': 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'
        }
    except:
        return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}

def extract_entities(text):
    """Extract named entities using spaCy."""
    try:
        doc = nlp(str(text))
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
    except:
        return {}

def perform_topic_modeling(texts, n_topics=5):
    """Perform topic modeling using LDA."""
    try:
        # Create document-term matrix
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # Create and fit LDA model
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Get top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
            topics.append(top_words)
        
        return topics, lda.transform(doc_term_matrix)
    except:
        return [], None

def compute_features(df_extracted):
    """Compute comprehensive features from extracted text content."""
    df = df_extracted.copy()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Normalize and clean text data
        status_text.text("Normalizing text data...")
        if 'body_text' not in df.columns:
            if 'html_content' in df.columns:
                titles, bodies, wcs = [], [], []
                for i, html in enumerate(df['html_content'].fillna("")):
                    t, b, wc = parse_html_to_text(html)
                    titles.append(t)
                    bodies.append(b)
                    wcs.append(wc)
                    progress_bar.progress(i / len(df))
                df['title'] = titles
                df['body_text'] = bodies
                df['word_count'] = wcs
                df['word_count'] = wcs
            else:
                df['title'] = df.get('title', pd.Series([""] * len(df)))
                df['body_text'] = df.get('body_text', pd.Series([""] * len(df)))
                df['word_count'] = df.get('word_count', pd.Series([0] * len(df)))
        
        # Ensure text fields are clean strings
        df['body_text'] = df['body_text'].fillna("").astype(str).apply(clean_text)
        df['title'] = df['title'].fillna("").astype(str).apply(clean_text)
        
        # 2. Basic Metrics
        status_text.text("Computing basic metrics...")
        progress_bar.progress(0.2)
        
        # Word count (recompute after cleaning)
        df['word_count'] = df['body_text'].apply(lambda t: len(t.split()) if t.strip() else 0)
        
        # Sentence count with error handling
        def safe_sentence_count(text):
            try:
                return len(sent_tokenize(text)) if text.strip() else 0
            except:
                return 0
        df['sentence_count'] = df['body_text'].apply(safe_sentence_count)
        
        # Average word length
        df['avg_word_length'] = df['body_text'].apply(
            lambda t: np.mean([len(w) for w in t.split()]) if t.strip() else 0
        )
        
        # 3. Readability Metrics
        status_text.text("Computing readability scores...")
        progress_bar.progress(0.4)
        
        def compute_readability(text):
            if not isinstance(text, str) or not text.strip():
                return pd.Series({
                    'flesch_reading_ease': 0.0,
                    'flesch_grade_level': 0.0,
                    'gunning_fog': 0.0,
                    'avg_syllables_per_word': 0.0
                })
            try:
                word_count = len(text.split())
                syllables = textstat.syllable_count(text)
                scores = {
                    'flesch_reading_ease': textstat.flesch_reading_ease(text),
                    'flesch_grade_level': textstat.flesch_kincaid_grade(text),
                    'gunning_fog': textstat.gunning_fog(text),
                    'avg_syllables_per_word': syllables / word_count if word_count > 0 else 0.0
                }
                return pd.Series(scores)
            except Exception as e:
                st.warning(f"Error computing readability scores: {str(e)}")
                return pd.Series({
                    'flesch_reading_ease': 0.0,
                    'flesch_grade_level': 0.0,
                    'gunning_fog': 0.0,
                    'avg_syllables_per_word': 0.0
                })
                
        # Apply readability computations
        status_text.text("Computing readability metrics...")
        readability_scores = df['body_text'].apply(compute_readability)
        for col in readability_scores.columns:
            df[col] = readability_scores[col].fillna(0.0)
        
        # Calculate average sentence length
        df['avg_sentence_length'] = df.apply(
            lambda row: row['word_count'] / row['sentence_count'] if row['sentence_count'] > 0 else 0, 
            axis=1
        )
        
        progress_bar.progress(0.5)
        
        # 4. Keyword Extraction (TF-IDF)
        status_text.text("Extracting keywords...")
        progress_bar.progress(0.6)
        
        corpus = df["body_text"].tolist()
        if any(corpus):
            # Configure TF-IDF
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=10000,
                ngram_range=(1, 2),  # Include bigrams
                max_df=0.95,         # Remove very common terms
                min_df=2             # Remove very rare terms
            )
            
            # Compute TF-IDF
            tfidf = vectorizer.fit_transform(corpus)
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            # Extract top keywords for each document
            top_keywords = []
            keyword_importance = []  # Store importance scores
            
            for row in tfidf:
                row_array = row.toarray().flatten()
                topn_idx = row_array.argsort()[-5:][::-1]
                kws = [feature_names[idx] for idx in topn_idx if row_array[idx] > 0]
                scores = [float(row_array[idx]) for idx in topn_idx if row_array[idx] > 0]
                
                top_keywords.append(", ".join(kws))
                keyword_importance.append(np.mean(scores) if scores else 0)
            
            df["top_keywords"] = top_keywords
            df["keyword_importance"] = keyword_importance
        else:
            df["top_keywords"] = [""] * len(df)
            df["keyword_importance"] = [0.0] * len(df)
        
        # 5. Text Embeddings
        status_text.text("Computing text embeddings...")
        progress_bar.progress(0.8)
        
        # Initialize sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Compute embeddings with batch processing
        embeddings = model.encode(
            df["body_text"].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 6. Additional Derived Metrics
        status_text.text("Computing final metrics...")
        progress_bar.progress(0.9)
        
        # Content density (words per sentence)
        df['words_per_sentence'] = df.apply(
            lambda r: r['word_count'] / max(1, r['sentence_count']),
            axis=1
        )
        
        # Overall quality score (0-100)
        df['quality_score'] = df.apply(
            lambda r: min(100, max(0, (
                (0.3 * min(100, r['word_count']/1000*100)) +  # Length component
                (0.3 * min(100, r['flesch_reading_ease'])) +  # Readability component
                (0.2 * min(100, r['keyword_importance']*100)) +  # Keyword relevance
                (0.2 * min(100, 100-(abs(r['words_per_sentence']-20)*5)))  # Sentence length component
            ))),
            axis=1
        )
        
        progress_bar.progress(1.0)
        status_text.text("Feature computation complete!")
        
        return df, embeddings
        
    except Exception as e:
        st.error(f"Error in feature computation: {str(e)}")
        raise e
    finally:
        progress_bar.empty()
        status_text.empty()

@st.cache_data
def analyze_duplicates(df, embeddings, threshold=0.85):
    """
    Analyze duplicates using embeddings and return comprehensive statistics.
    Returns: duplicate pairs, statistics dict, and similarity matrix
    """
    if embeddings is None or len(embeddings) == 0:
        return [], {}, np.array([[]])
        
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Find duplicate pairs
    pairs = []
    n = sim_matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):  # upper triangle only
            similarity = float(sim_matrix[i,j])
            if similarity >= threshold:
                pairs.append({
                    'url1': df.iloc[i]['url'],
                    'title1': df.iloc[i]['title'],
                    'word_count1': int(df.iloc[i]['word_count']),
                    'url2': df.iloc[j]['url'],
                    'title2': df.iloc[j]['title'],
                    'word_count2': int(df.iloc[j]['word_count']),
                    'similarity': similarity
                })
    
    # Compute statistics
    stats = {
        'total_documents': n,
        'duplicate_pairs': len(pairs),
        'unique_urls_in_pairs': len(set([p['url1'] for p in pairs] + [p['url2'] for p in pairs])),
        'avg_similarity': np.mean([p['similarity'] for p in pairs]) if pairs else 0,
        'max_similarity': np.max([p['similarity'] for p in pairs]) if pairs else 0,
        'similarity_distribution': np.histogram(
            [p['similarity'] for p in pairs], 
            bins=[0.85, 0.90, 0.95, 0.97, 0.99, 1.0]
        )[0].tolist() if pairs else [0]*5
    }
    
    return pairs, stats, sim_matrix

def get_content_quality_label(row):
    """
    Determine content quality label based on word count and readability
    """
    if row['word_count'] > 1500 and 50 <= row['flesch_reading_ease'] <= 70:
        return 'High'
    elif row['word_count'] < 500 or row['flesch_reading_ease'] < 30:
        return 'Low'
    else:
        return 'Medium'

def get_baseline_quality(word_count):
    """
    Simple baseline classifier using only word count
    """
    if word_count > 1500:
        return 'High'
    elif word_count < 500:
        return 'Low'
    else:
        return 'Medium'

def train_quality_model(df):
    """
    Train a content quality classification model
    """
    # Prepare features
    features = ['word_count', 'sentence_count', 'flesch_reading_ease', 
                'avg_sentence_length', 'avg_syllables_per_word']
    
    # Create feature matrix X and labels y
    X = df[features]
    df['quality_label'] = df.apply(get_content_quality_label, axis=1)
    y = df['quality_label']
    
    # Create baseline predictions
    baseline_preds = df['word_count'].apply(get_baseline_quality)
    baseline_accuracy = accuracy_score(y, baseline_preds)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get feature importance
    feature_imp = sorted(zip(model.feature_importances_, features), reverse=True)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'baseline_accuracy': baseline_accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_imp,
        'features': features
    }
    
    return pairs, stats, sim_matrix

@st.cache_data
def analyze_thin_content(df):
    """
    Analyze thin content and return statistics.
    """
    # Add is_thin column
    df['is_thin'] = df['word_count'] < 500
    
    # Compute statistics
    stats = {
        'total_pages': len(df),
        'thin_pages': sum(df['is_thin']),
        'thin_percentage': (sum(df['is_thin']) / len(df)) * 100,
        'word_count_stats': {
            'min': df['word_count'].min(),
            'max': df['word_count'].max(),
            'mean': df['word_count'].mean(),
            'median': df['word_count'].median()
        },
        'distribution': np.histogram(
            df['word_count'], 
            bins=[0, 250, 500, 750, 1000, 1500, 2000, float('inf')]
        )[0].tolist()
    }
    
    return df, stats

def train_quality_model(df_features):
    """Train a quality classification model and return comprehensive results."""
    # Prepare data
    def label_row(r):
        wc = r["word_count"]
        fr = textstat.flesch_reading_ease(r["body_text"]) if isinstance(r["body_text"], str) else 0
        if (wc > 1500) and (50 <= fr <= 70):
            return "High"
        elif (wc < 500) or (fr < 30):
            return "Low"
        else:
            return "Medium"
            
    # Create copy and generate labels
    df = df_features.copy()
    df["label"] = df.apply(label_row, axis=1)
    
    # Prepare features and target
    features = ["word_count", "sentence_count", "flesch_reading_ease"]
    X = df[features].fillna(0)
    y = df["label"]
    
    # Create baseline predictions
    baseline_preds = df['word_count'].apply(lambda x: 
        "High" if x > 1500 else "Low" if x < 500 else "Medium"
    )
    baseline_accuracy = accuracy_score(y, baseline_preds)
    
    # Check if we have enough data for training
    if len(df) < 2:
        return {
            'error': 'Not enough data for training',
            'model': None,
            'scaler': None,
            'accuracy': 0.0,
            'baseline_accuracy': 0.0,
            'classification_report': {},
            'confusion_matrix': None,
            'feature_importance': [],
            'features': features
        }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
        stratify=y if len(y.unique()) > 1 else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    results = {
        'model': model,
        'scaler': scaler,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'baseline_accuracy': float(baseline_accuracy),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=["High", "Medium", "Low"]),
        'feature_importance': list(zip(model.feature_importances_, features)),
        'features': features
    }
    
    # Save model
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'features': features
    }, MODEL_FILE)
    
    return results
    return clf, acc, f1, cm, feat_imp, df

def safe_scrape_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SEO-Analyzer/1.0)"}
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 200:
            title, body, wc = parse_html_to_text(resp.text)
            return {"url": url, "title": title, "body_text": body, "word_count": wc}
        else:
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# --- App UI ---
st.title("SEO Content Quality & Duplicate Detector")

# Sidebar navigation
tabs = ["Data Processing", "Feature Analysis", "Duplicate Detector", "Content Quality Score", "Advanced Analysis", "Live URL Analyzer"]
choice = st.sidebar.radio("Navigation", tabs)

# Load raw dataset
df_raw = load_dataset()

if choice == "Data Processing":
    st.header("1. Data Processing & Parsing")
    st.markdown("Load your data file (CSV format with columns: url, html_content/body_text). Maximum file size: 100MB")
    
    # File upload section with size limit (100MB = 100 * 1024 * 1024 bytes)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    with col2:
        file_id = st.text_input("File ID (optional)", help="Assign a custom ID to your file")
    
    if uploaded:
        # Check file size
        file_size = len(uploaded.getvalue())
        if file_size > MAX_FILE_SIZE:
            st.error(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds the 100MB limit.")
        else:
            try:
                df_raw = pd.read_csv(uploaded)
                if file_id:  # Add file ID if provided
                    df_raw['file_id'] = file_id
                st.success(f"Uploaded dataset loaded successfully. Size: {file_size / 1024 / 1024:.1f}MB")
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
    
    if not df_raw.empty:
        st.write("Dataset preview:")
        preview_cols = ['file_id'] if 'file_id' in df_raw.columns else []
        preview_cols.extend(['url', 'html_content'] if 'html_content' in df_raw.columns else ['url', 'body_text'])
        st.dataframe(df_raw[preview_cols].head(20))
        
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                if 'html_content' in df_raw.columns:
                    # Extract from HTML if html_content is present
                    extracted = process_and_extract(df_raw)
                else:
                    # If body_text is already present, use it directly
                    extracted = df_raw.copy()
                    if 'word_count' not in extracted.columns:
                        extracted['word_count'] = extracted['body_text'].str.split().str.len()
                
                # Preserve file_id if it exists
                if 'file_id' in df_raw.columns:
                    extracted['file_id'] = df_raw['file_id']
                    
            st.success(f"Processing completed. Saved to `{EXTRACTED_CSV}`.")
            extracted.to_csv(EXTRACTED_CSV, index=False)
            
            # Show preview with file_id if it exists
            preview_cols = ['file_id'] if 'file_id' in extracted.columns else []
            preview_cols.extend(['url', 'title', 'body_text', 'word_count'])
            st.dataframe(extracted[preview_cols].head(50))
            
            st.download_button(
                "Download processed data", 
                data=extracted.to_csv(index=False).encode('utf-8'),
                file_name=EXTRACTED_CSV
            )

elif choice == "Live Analysis Dashboard":
    st.header("2. Live Analysis Dashboard")
    st.markdown("Analyze any URL in real-time with comprehensive metrics and visualizations.")
    
    # URL input in a clean layout
    url_input = st.text_input("Enter a URL to analyze", placeholder="https://example.com")
    analyze_btn = st.button("🔍 Analyze URL", type="primary")
    
    if analyze_btn:
        if not url_input.strip():
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Analyzing URL and generating insights..."):
                scraped = safe_scrape_url(url_input.strip())
                if "error" in scraped:
                    st.error(f"Failed to fetch URL: {scraped['error']}")
                else:
                    # Layout: 3 columns for key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    # Column 1: Basic Info
                    with col1:
                        st.info("📊 Content Overview")
                        st.markdown(f"**Title**  \n{scraped.get('title','')}")
                        st.markdown(f"**Word Count**  \n{scraped.get('word_count',0):,}")
                        s_count = len(sent_tokenize(scraped.get('body_text','') or ""))
                        st.markdown(f"**Sentence Count**  \n{s_count:,}")
                    
                    # Column 2: Readability Metrics
                    with col2:
                        body_text = scraped.get('body_text','') or ""
                        fr = textstat.flesch_reading_ease(body_text)
                        fkg = textstat.flesch_kincaid_grade(body_text)
                        st.info("📚 Readability Analysis")
                        st.markdown(f"**Flesch Reading Ease**  \n{fr:.1f}")
                        st.markdown(f"**Grade Level**  \n{fkg:.1f}")
                        
                        # Readability donut chart
                        fig_read = plt.figure(figsize=(4, 4))
                        ease_percent = max(0, min(100, fr))  # Normalize to 0-100
                        plt.pie([ease_percent, 100-ease_percent], 
                              colors=['#2ecc71' if fr >= 60 else '#f1c40f' if fr >= 30 else '#e74c3c', '#ecf0f1'],
                              radius=1, counterclock=False, startangle=90,
                              wedgeprops=dict(width=0.3))
                        plt.title("Reading Ease Score")
                        st.pyplot(fig_read)
                        plt.close()
                    
                    # Column 3: Content Quality
                    with col3:
                        st.info("⭐ Quality Metrics")
                        # Compute additional metrics
                        avg_words_per_sent = scraped.get('word_count',0) / max(1, s_count)
                        thin_content = scraped.get('word_count',0) < 500
                        
                        quality_score = min(100, max(0, (
                            (0.4 * min(100, scraped.get('word_count',0)/1000*100)) +  # Word count contribution
                            (0.3 * min(100, fr)) +  # Reading ease contribution
                            (0.3 * min(100, 100-(abs(avg_words_per_sent-20)*5)))  # Sentence length contribution
                        )))
                        
                        st.markdown(f"**Quality Score**  \n{quality_score:.1f}/100")
                        st.markdown(f"**Avg. Words/Sentence**  \n{avg_words_per_sent:.1f}")
                        st.markdown(f"**Content Length**  \n{'✅ Good' if not thin_content else '⚠️ Thin'}")
                        
                        # Quality Score donut
                        fig_qual = plt.figure(figsize=(4, 4))
                        plt.pie([quality_score, 100-quality_score],
                              colors=['#2ecc71' if quality_score >= 70 else '#f1c40f' if quality_score >= 40 else '#e74c3c', '#ecf0f1'],
                              radius=1, counterclock=False, startangle=90,
                              wedgeprops=dict(width=0.3))
                        plt.title("Content Quality Score")
                        st.pyplot(fig_qual)
                        plt.close()
                    
                    # Full width sections below
                    st.markdown("---")
                    
                    # Bar chart of key metrics
                    st.subheader("📈 Content Metrics Breakdown")
                    metrics_fig = plt.figure(figsize=(10, 4))
                    metrics = {
                        'Word Count (÷10)': scraped.get('word_count',0)/10,
                        'Reading Ease': fr,
                        'Quality Score': quality_score,
                        'Grade Level': fkg,
                        'Words/Sentence': avg_words_per_sent
                    }
                    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6']
                    plt.bar(metrics.keys(), metrics.values(), color=colors)
                    plt.xticks(rotation=45)
                    plt.ylim(0, 100)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(metrics_fig)
                    plt.close()
                    
                    # Similar content section
                    st.markdown("---")
                    st.subheader("🔍 Similar Content Detection")
                    
                    # Load or compute embeddings for similarity check
                    if 'df_features' not in st.session_state or 'embeddings' not in st.session_state:
                        if os.path.exists(EXTRACTED_CSV):
                            with st.spinner("Computing embeddings for similarity check..."):
                                ext = pd.read_csv(EXTRACTED_CSV)
                                model = SentenceTransformer('all-MiniLM-L6-v2')
                                emb = model.encode(ext['body_text'].fillna("").astype(str).tolist(), show_progress_bar=False)
                                st.session_state['embeddings'] = emb
                                st.session_state['df_features'] = ext
                    
                    # Compute similarity
                    embeddings = st.session_state.get('embeddings', None)
                    if embeddings is not None and len(embeddings)>0:
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        new_emb = model.encode([scraped.get('body_text','') or ""], show_progress_bar=False)
                        sims = cosine_similarity(np.array(new_emb), embeddings)[0]
                        top_idx = sims.argsort()[-5:][::-1]  # Get top 5
                        
                        # Create similarity chart
                        sim_fig = plt.figure(figsize=(10, 4))
                        top_sims = [float(sims[i]) for i in top_idx]
                        top_urls = [st.session_state['df_features'].loc[i,'url'] for i in top_idx]
                        plt.barh(range(len(top_sims)), top_sims, color='#3498db')
                        plt.yticks(range(len(top_sims)), [f"URL {i+1}" for i in range(len(top_sims))])
                        plt.xlim(0, 1)
                        plt.title("Top Similar Content")
                        plt.xlabel("Similarity Score")
                        st.pyplot(sim_fig)
                        plt.close()
                        
                        # Show URLs in a table
                        sim_df = pd.DataFrame({
                            'URL': top_urls,
                            'Similarity': top_sims
                        })
                        st.dataframe(sim_df)
                    else:
                        st.info("⚠️ Dataset embeddings not available. Process your dataset first to enable similarity comparison.")
                    
                    # Quality prediction from model
                    if os.path.exists(MODEL_FILE):
                        st.markdown("---")
                        st.subheader("🎯 ML-Based Quality Prediction")
                        try:
                            loaded = joblib.load(MODEL_FILE)
                            clf = None
                            
                            if hasattr(loaded, "predict"):
                                clf = loaded
                            elif isinstance(loaded, dict):
                                for key in ("model", "clf", "estimator", "pipeline", "classifier"):
                                    if key in loaded and hasattr(loaded[key], "predict"):
                                        clf = loaded[key]
                                        break
                                if clf is None:
                                    for v in loaded.values():
                                        if hasattr(v, "predict"):
                                            clf = v
                                            break
                            
                            if clf is not None:
                                X_new = np.array([[scraped.get('word_count',0), s_count, fr]])
                                try:
                                    pred = clf.predict(X_new)[0]
                                    
                                    # Create prediction visualization
                                    pred_fig = plt.figure(figsize=(6, 6))
                                    quality_colors = {'High': '#2ecc71', 'Medium': '#f1c40f', 'Low': '#e74c3c'}
                                    plt.pie([1], colors=[quality_colors[pred]], 
                                          labels=[f'Predicted Quality: {pred}'],
                                          wedgeprops=dict(width=0.3))
                                    plt.title("Quality Classification")
                                    st.pyplot(pred_fig)
                                    plt.close()
                                except Exception as e:
                                    st.error(f"Failed to generate prediction: {e}")
                        except Exception as e:
                            st.error(f"Failed to load model: {e}")

elif choice == "Feature Analysis":
    st.header("2. Feature Engineering & Visualization")
    st.markdown("Compute sentence count, Flesch Reading Ease, top keywords and embeddings.")
    # Load extracted data if exists else run extraction
    try:
        if os.path.exists(EXTRACTED_CSV):
            extracted = pd.read_csv(EXTRACTED_CSV)
        else:
            raise FileNotFoundError
    except (FileNotFoundError, pd.errors.EmptyDataError):
        extracted = process_and_extract(df_raw)
    st.write("Extracted content preview:")
    st.dataframe(extracted.head(20))
    if st.button("Compute Features & Embeddings"):
        with st.spinner("Computing features and generating embeddings (this may download models)..."):
            df_feat, embeddings = compute_features(extracted)
            # save embeddings to session_state
            st.session_state['embeddings'] = embeddings
            st.session_state['df_features'] = df_feat
        st.success("Features and embeddings computed.")
    if 'df_features' in st.session_state:
        df_feat = st.session_state['df_features']
        embeddings = st.session_state.get('embeddings', None)
        st.subheader("Full Features DataFrame")
        st.dataframe(df_feat.head(100))
        st.download_button("Download features CSV", data=df_feat.to_csv(index=False).encode('utf-8'), file_name="features.csv")
        st.subheader("Distributions")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Word Count Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df_feat['word_count'].fillna(0), kde=False, ax=ax)
            st.pyplot(fig)
        with col2:
            st.write("Flesch Reading Ease Distribution")
            fig2, ax2 = plt.subplots()
            
            # Safely access the readability scores
            if 'flesch_reading_ease' in df_feat.columns:
                scores = df_feat['flesch_reading_ease'].fillna(0)
                sns.boxplot(x=scores, ax=ax2)
                ax2.set_title("Flesch Reading Ease Score")
                ax2.set_xlabel("Score (0-100)")
            else:
                st.warning("Readability scores not yet computed. Process your data first.")
            
            st.pyplot(fig2)
            plt.close(fig2)
        st.subheader("Top Keywords Analysis")
        
        # Get unique keywords with their frequencies
        all_keywords = []
        for kw_list in df_feat['top_keywords'].fillna(""):
            if isinstance(kw_list, str):
                all_keywords.extend([k.strip() for k in kw_list.split(",") if k.strip()])
        
        if all_keywords:
            # Create word cloud
            st.write("#### Visual Representation")
            kw_text = " ".join(all_keywords)
            wc = WordCloud(width=800, height=400).generate(kw_text)
            fig3, ax3 = plt.subplots(figsize=(10,4))
            ax3.imshow(wc, interpolation='bilinear')
            ax3.axis('off')
            st.pyplot(fig3)
            plt.close(fig3)
            
            # Keyword frequency analysis
            from collections import Counter
            keyword_freq = Counter(all_keywords)
            top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Display top keywords with frequencies
            st.write("#### Top Keywords (Frequency Analysis)")
            col1, col2 = st.columns([2,1])
            with col1:
                # Format keywords as markdown table
                keywords_md = "| Keyword | Frequency |\n|----------|------------|\n"
                for kw, freq in top_keywords[:20]:  # Show top 20
                    keywords_md += f"| {kw} | {freq} |\n"
                st.markdown(keywords_md)
            
            with col2:
                # Show distribution of top 10
                fig4, ax4 = plt.subplots(figsize=(6,4))
                top_10 = dict(top_keywords[:10])
                plt.barh(list(top_10.keys()), list(top_10.values()))
                plt.title("Top 10 Keywords")
                ax4.invert_yaxis()  # Show highest frequency at top
                st.pyplot(fig4)
                plt.close(fig4)
            
            # Copiable text version
            st.write("#### Copiable Keyword List")
            st.info("Click the text area below to copy all keywords:")
            
            # Format keywords with frequencies
            copiable_text = ""
            for kw, freq in top_keywords:
                copiable_text += f"{kw} ({freq} times)\n"
            
            st.text_area(
                "Keywords with frequencies:",
                value=copiable_text,
                height=200
            )
            
            # Additional statistics
            total_unique = len(keyword_freq)
            total_keywords = sum(keyword_freq.values())
            st.caption(f"""
            **Keyword Statistics:**
            - Total unique keywords: {total_unique}
            - Total keyword occurrences: {total_keywords}
            - Average frequency: {total_keywords/total_unique:.1f} occurrences per keyword
            """)
        else:
            st.info("No keywords available to build a word cloud yet.")

elif choice == "Duplicate Detector":
    st.header("3. Duplicate & Thin Content Analysis")
    st.markdown("Comprehensive analysis of duplicate content and thin pages in your dataset.")

    # Load or compute extracted/features/embeddings
    if os.path.exists(EXTRACTED_CSV):
        extracted = pd.read_csv(EXTRACTED_CSV)
    else:
        extracted = process_and_extract(df_raw)

    # Ensure features and embeddings exist in session
    if 'df_features' not in st.session_state or 'embeddings' not in st.session_state:
        with st.spinner("Computing features and embeddings..."):
            df_feat, embeddings = compute_features(extracted)
            st.session_state['df_features'] = df_feat
            st.session_state['embeddings'] = np.array(embeddings)

    df_feat = st.session_state['df_features']
    embeddings = np.array(st.session_state.get('embeddings', []))

    # Helper: validate embeddings
    def embeddings_valid(emb):
        try:
            emb = np.array(emb)
            if emb.size == 0:
                return False
            if emb.ndim != 2:
                return False
            if emb.shape[0] < 1:
                return False
            if emb.shape[0] > 1 and np.allclose(emb, emb[0]):
                return False
            return True
        except Exception:
            return False

    # If embeddings look invalid, recompute them
    if not embeddings_valid(embeddings):
        st.warning("Dataset embeddings look invalid or uniform — recomputing embeddings now.")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df_feat['body_text'].fillna("").astype(str).tolist(), show_progress_bar=True)
        embeddings = np.array(embeddings)
        st.session_state['embeddings'] = embeddings

    # 1. Thin Content Analysis
    st.subheader("📉 Thin Content Analysis")
    df_with_thin, thin_stats = analyze_thin_content(df_feat)
    
    # Thin content metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pages", thin_stats['total_pages'])
    with col2:
        st.metric("Thin Pages (<500 words)", thin_stats['thin_pages'])
    with col3:
        st.metric("Thin Content %", f"{thin_stats['thin_percentage']:.1f}%")
    
    # Word count distribution
    st.subheader("Word Count Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = ['0-250', '250-500', '500-750', '750-1000', '1000-1500', '1500-2000', '2000+']
    plt.bar(bins, thin_stats['distribution'], color=['#e74c3c' if i < 2 else '#2ecc71' for i in range(len(bins))])
    plt.xticks(rotation=45)
    plt.title("Content Length Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    plt.close()

    # Thin content table
    st.subheader("Thin Content Pages")
    thin_pages = df_with_thin[df_with_thin['is_thin']][['url', 'title', 'word_count']].sort_values('word_count')
    st.dataframe(thin_pages)
    st.download_button(
        "Download thin content report",
        data=thin_pages.to_csv(index=False).encode('utf-8'),
        file_name="thin_content_report.csv"
    )

    # 2. Duplicate Content Analysis
    st.markdown("---")
    st.subheader("🔄 Duplicate Content Analysis")
    
    # Similarity threshold control
    thresh = st.slider(
        "Similarity Threshold (higher = more similar)",
        min_value=0.80,
        max_value=1.00,
        value=0.85,
        step=0.01,
        help="Pages with similarity above this threshold are considered potential duplicates"
    )

    # Compute duplicates
    with st.spinner("Analyzing duplicate content..."):
        pairs, dup_stats, sim_matrix = analyze_duplicates(df_feat, embeddings, threshold=thresh)
        
        # Display duplicate statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", dup_stats['total_documents'])
        with col2:
            st.metric("Duplicate Pairs", dup_stats['duplicate_pairs'])
        with col3:
            st.metric("Unique URLs in Pairs", dup_stats['unique_urls_in_pairs'])
            
        # Similarity distribution chart
        if pairs:
            st.subheader("Similarity Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sim_bins = ['0.85-0.90', '0.90-0.95', '0.95-0.97', '0.97-0.99', '0.99-1.00']
            plt.bar(sim_bins, dup_stats['similarity_distribution'], color='#3498db')
            plt.xticks(rotation=45)
            plt.title("Distribution of Similarity Scores")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            plt.close()
            
            # Display duplicate pairs
            st.subheader("Duplicate Page Pairs")
            dup_df = pd.DataFrame(pairs).sort_values('similarity', ascending=False)
            st.dataframe(dup_df)
            st.download_button(
                "Download duplicate pairs report",
                data=dup_df.to_csv(index=False).encode('utf-8'),
                file_name="duplicate_pairs_report.csv"
            )
            
            # Similarity matrix heatmap
            st.subheader("Similarity Matrix Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(sim_matrix, cmap='YlOrRd', ax=ax)
            plt.title("Content Similarity Matrix")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No duplicate pairs found at the current similarity threshold.")
    
    # 3. Live URL Comparison
    st.markdown("---")
    st.subheader("🔍 Compare Live URL")
    
    url_input = st.text_input(
        "Enter a URL to check for duplicates",
        placeholder="https://example.com",
        help="Enter any URL to compare against your dataset"
    )
    
    if st.button("Analyze URL", type="primary"):
        if not url_input.strip():
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Analyzing URL..."):
                scraped = safe_scrape_url(url_input.strip())
                if "error" in scraped:
                    st.error(f"Failed to fetch URL: {scraped['error']}")
                else:
                    # Basic metrics
                    st.info("Content Overview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Title:** {scraped.get('title','')}")
                        st.markdown(f"**Word Count:** {scraped.get('word_count',0):,}")
                    with col2:
                        is_thin = scraped.get('word_count',0) < 500
                        st.markdown(f"**Content Type:** {'⚠️ Thin Content' if is_thin else '✅ Full Content'}")
                    
                    # Compute similarity with dataset
                    if embeddings_valid(embeddings):
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        new_emb = model.encode([scraped.get('body_text','') or ""], show_progress_bar=False)
                        sims = cosine_similarity(np.array(new_emb), embeddings)[0]
                        
                        # Get top matches
                        top_idx = sims.argsort()[-5:][::-1]
                        match_rows = []
                        for i in top_idx:
                            match_rows.append({
                                'url': df_feat.loc[i,'url'],
                                'title': df_feat.loc[i,'title'],
                                'similarity': float(sims[i]),
                                'word_count': int(df_feat.loc[i,'word_count'])
                            })
                        
                        # Display results
                        st.subheader("Most Similar Pages")
                        results_df = pd.DataFrame(match_rows)
                        
                        # Bar chart of top similarities
                        fig, ax = plt.subplots(figsize=(10, 4))
                        plt.barh(
                            range(len(match_rows)),
                            [r['similarity'] for r in match_rows],
                            color=['#e74c3c' if r['similarity'] >= thresh else '#3498db' for r in match_rows]
                        )
                        plt.yticks(range(len(match_rows)), [f"Match {i+1}" for i in range(len(match_rows))])
                        plt.xlim(0, 1)
                        plt.title("Similarity Scores")
                        st.pyplot(fig)
                        plt.close()
                        
                        st.dataframe(results_df)
                        st.download_button(
                            "Download similarity results",
                            data=results_df.to_csv(index=False).encode('utf-8'),
                            file_name="url_similarity_results.csv"
                        )
                    else:
                        st.error("Dataset embeddings are not available. Process your dataset first.")

elif choice == "Content Quality Score":
    st.header("4. Content Quality Analysis")
    st.markdown("Analyze and score content quality using machine learning classification.")

    # Load or compute necessary data
    if os.path.exists(EXTRACTED_CSV):
        extracted = pd.read_csv(EXTRACTED_CSV)
    else:
        extracted = process_and_extract(df_raw)

    if 'df_features' not in st.session_state:
        with st.spinner("Computing features and embeddings first..."):
            df_feat, embeddings = compute_features(extracted)
            st.session_state['df_features'] = df_feat
            st.session_state['embeddings'] = embeddings
    df_feat = st.session_state['df_features']

    # Train model button with explainer
    st.markdown("""
    This model classifies content quality based on:
    - **High Quality**: >1500 words AND readable (Flesch score 50-70)
    - **Low Quality**: <500 words OR hard to read (Flesch score <30)
    - **Medium Quality**: All other content
    """)

    if st.button("Train Content Quality Model"):
        with st.spinner("Training quality classification model..."):
            try:
                results = train_quality_model(df_feat)
                st.session_state['model_results'] = results
                
                # Display results
                st.subheader("Model Performance")
                
                # Overall metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Model Accuracy",
                        value=f"{results['accuracy']:.1%}"
                    )
                with col2:
                    st.metric(
                        label="Baseline Accuracy",
                        value=f"{results['baseline_accuracy']:.1%}",
                        delta=f"{(results['accuracy'] - results['baseline_accuracy']):.1%}"
                    )
                
                # Classification report
                st.subheader("Detailed Performance Metrics")
                report = results['classification_report']
                metrics_df = pd.DataFrame({
                    'Precision': [report[label]['precision'] for label in ['Low', 'Medium', 'High']],
                    'Recall': [report[label]['recall'] for label in ['Low', 'Medium', 'High']],
                    'F1-Score': [report[label]['f1-score'] for label in ['Low', 'Medium', 'High']],
                    'Support': [report[label]['support'] for label in ['Low', 'Medium', 'High']]
                }, index=['Low Quality', 'Medium Quality', 'High Quality'])
                
                st.dataframe(
                    metrics_df.style.format(
                        "{:.2f}",
                        subset=['Precision', 'Recall', 'F1-Score']
                    ).format(
                        "{:.0f}",
                        subset=['Support']
                    )
                )
                
                st.success("Model trained successfully!")
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                raise e
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(results['confusion_matrix'], 
                       annot=True, 
                       fmt='d',
                       cmap='YlOrRd',
                       xticklabels=['Low', 'Medium', 'High'],
                       yticklabels=['Low', 'Medium', 'High'])
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Quality")
            plt.ylabel("True Quality")
            st.pyplot(fig)
            plt.close()
            
            # Feature Importance
            if 'feature_importance' in results and results['feature_importance']:
                st.subheader("Feature Importance")
                feat_imp = results['feature_importance']
                feat_names = ['Word Count', 'Sentence Count', 'Flesch Reading Ease']
                feat_imp_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Importance': [imp for imp, _ in feat_imp]
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=feat_imp_df, x='Importance', y='Feature')
                plt.title("Feature Importance in Quality Classification")
                ax.set_xlabel("Relative Importance")
            st.pyplot(fig)
            plt.close()
            
            # Save model
            joblib.dump(results['model'], MODEL_FILE)
            st.success("Model trained successfully and saved to disk!")

    # Apply model to all content
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = st.session_state.get('model_results', {}).get('scaler')
            
            # Prepare features for prediction - use exactly the same features as training
            features = ['word_count', 'sentence_count', 'flesch_reading_ease']
            X = df_feat[features].fillna(0)
            
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            # Make predictions
            predictions = model.predict(X_scaled)
            df_feat['quality_score'] = predictions
            
            # Display results
            st.subheader("Content Quality Analysis")
            
            # Quality distribution
            quality_dist = df_feat['quality_score'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=quality_dist.index, y=quality_dist.values, palette='viridis')
            plt.title("Distribution of Content Quality")
            st.pyplot(fig)
            plt.close()
            
            # Display full results table
            st.subheader("Quality Scores by URL")
            results_df = df_feat[['url', 'title', 'word_count', 'flesch_reading_ease', 'quality_score']].copy()
            results_df = results_df.sort_values('quality_score')
            st.dataframe(results_df)
            
            # Download results
            st.download_button(
                "Download Quality Analysis Results",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name="content_quality_analysis.csv"
            )
            
        except Exception as e:
            st.error(f"Error loading or applying model: {str(e)}")
    else:
        st.info("No trained model found. Click 'Train Content Quality Model' to analyze your content.")

    if 'quality_metrics' in st.session_state:
        metrics = st.session_state['quality_metrics']
        st.subheader("Model Metrics")
        st.write(f"Accuracy: **{metrics['accuracy']:.4f}**")
        st.write(f"F1-Score: **{metrics['f1']:.4f}**")
        st.subheader("Confusion Matrix")
        cm = metrics['confusion_matrix']
        labels = ["High","Medium","Low"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
        st.pyplot(fig)
        st.subheader("Feature Importances")
        fig2, ax2 = plt.subplots()
        sns.barplot(x=["word_count","sentence_count","flesch_reading_ease"], y=metrics['feature_importances'], ax=ax2)
        st.pyplot(fig2)
    # Final predictions table
    if 'quality_model' in st.session_state:
        loaded = st.session_state['quality_model']
        clf = None
        # If the loaded object already has predict, use it
        if hasattr(loaded, "predict"):
            clf = loaded
        # If a dict was saved, try common keys and values to find an estimator
        elif isinstance(loaded, dict):
            for key in ("model", "clf", "estimator", "pipeline", "classifier"):
                if key in loaded and hasattr(loaded[key], "predict"):
                    clf = loaded[key]
                    break
            if clf is None:
                # fallback: pick the first value that looks like an estimator
                for v in loaded.values():
                    if hasattr(v, "predict"):
                        clf = v
                        break
        
        if clf is not None:
            X = df_feat[["word_count","sentence_count","flesch_reading_ease"]].fillna(0)
            try:
                preds = clf.predict(X)
                df_out = df_feat.copy()
                df_out["predicted_quality"] = preds
                st.subheader("Predicted Quality for all URLs")
                st.dataframe(df_out[["url","title","word_count","flesch_reading_ease","predicted_quality"]])
                st.download_button("Download predictions CSV", data=df_out.to_csv(index=False).encode('utf-8'), file_name="predictions_with_quality.csv")
            except Exception as e:
                st.error(f"Failed to generate predictions: {e}")
        else:
            st.error("Model loaded but no usable estimator with `predict` was found. Please retrain the model.")
    else:
        st.info("Model not trained yet. Click 'Train Quality Model' to train and save the model.")

elif choice == "Advanced Analysis":
    st.header("5. Advanced Content Analysis")
    st.markdown("""
    This section provides in-depth analysis of your content using advanced NLP techniques:
    - Sentiment Analysis
    - Named Entity Recognition
    - Topic Modeling
    - Advanced Visualizations
    """)

    # Load data
    if os.path.exists(EXTRACTED_CSV):
        extracted = pd.read_csv(EXTRACTED_CSV)
    else:
        extracted = process_and_extract(df_raw)

    if 'df_features' not in st.session_state:
        with st.spinner("Computing features..."):
            df_feat, embeddings = compute_features(extracted)
            st.session_state['df_features'] = df_feat
            st.session_state['embeddings'] = embeddings
    df_feat = st.session_state['df_features']

    # 1. Sentiment Analysis
    st.subheader("1️⃣ Sentiment Analysis")
    with st.spinner("Analyzing sentiment..."):
        sentiments = df_feat['body_text'].apply(analyze_sentiment)
        df_feat['sentiment_polarity'] = sentiments.apply(lambda x: x['polarity'])
        df_feat['sentiment_subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])
        df_feat['sentiment_label'] = sentiments.apply(lambda x: x['sentiment'])

        # Sentiment Distribution
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Sentiment Distribution", "Subjectivity vs Polarity"))
        
        # Sentiment counts
        sentiment_counts = df_feat['sentiment_label'].value_counts()
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        fig.add_trace(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                  marker_color=[colors[s] for s in sentiment_counts.index]),
            row=1, col=1
        )

        # Subjectivity vs Polarity scatter
        fig.add_trace(
            go.Scatter(x=df_feat['sentiment_subjectivity'], y=df_feat['sentiment_polarity'],
                      mode='markers', marker=dict(color=df_feat['sentiment_polarity'],
                      colorscale='RdYlGn')),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig)

    # 2. Named Entity Recognition
    st.subheader("2️⃣ Named Entity Recognition")
    with st.spinner("Extracting entities..."):
        # Sample a few documents for NER analysis
        sample_size = min(10, len(df_feat))
        sample_docs = df_feat.sample(n=sample_size)
        all_entities = {}
        
        for _, row in sample_docs.iterrows():
            entities = extract_entities(row['body_text'])
            for entity_type, values in entities.items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = []
                all_entities[entity_type].extend(values)

        # Display entity counts
        if all_entities:
            entity_counts = {k: len(set(v)) for k, v in all_entities.items()}
            fig = px.bar(
                x=list(entity_counts.keys()),
                y=list(entity_counts.values()),
                title="Named Entities Distribution",
                labels={'x': 'Entity Type', 'y': 'Unique Count'}
            )
            st.plotly_chart(fig)

            # Show example entities
            st.write("Example entities found:")
            for ent_type, values in all_entities.items():
                unique_values = list(set(values))[:5]  # Show up to 5 examples
                st.write(f"**{ent_type}:** {', '.join(unique_values)}")
        else:
            st.info("No named entities found in the sampled text.")

    # 3. Topic Modeling
    st.subheader("3️⃣ Topic Modeling")
    n_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5)
    
    with st.spinner("Performing topic modeling..."):
        topics, doc_topics = perform_topic_modeling(
            df_feat['body_text'].fillna("").astype(str).tolist(),
            n_topics=n_topics
        )
        
        if topics and doc_topics is not None:
            # Display topics
            for idx, topic in enumerate(topics):
                st.write(f"**Topic {idx+1}:** {', '.join(topic)}")
            
            # Topic distribution heatmap
            fig = px.imshow(
                doc_topics,
                labels=dict(x="Topic", y="Document", color="Weight"),
                title="Document-Topic Distribution"
            )
            st.plotly_chart(fig)
            
            # Topic prevalence
            topic_prevalence = doc_topics.mean(axis=0)
            fig = px.bar(
                x=[f"Topic {i+1}" for i in range(len(topic_prevalence))],
                y=topic_prevalence,
                title="Topic Prevalence Across Documents",
                labels={'x': 'Topic', 'y': 'Average Weight'}
            )
            st.plotly_chart(fig)
        else:
            st.warning("Topic modeling failed. Try adjusting the number of topics or check your data.")

    # 4. Download Analysis Results
    st.subheader("4️⃣ Download Analysis Results")
    
    # Prepare download data
    analysis_results = df_feat[[
        'url', 'title', 'word_count', 
        'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_label'
    ]].copy()
    
    st.download_button(
        "Download Analysis Results CSV",
        data=analysis_results.to_csv(index=False).encode('utf-8'),
        file_name="advanced_analysis_results.csv",
        mime="text/csv"
    )

elif choice == "Live URL Analyzer":
    st.header("6. Real-Time URL Analyzer")
    st.markdown("Paste a live URL to analyze. The app will scrape, extract features, compute embeddings, predict quality, and check duplicates against the dataset.")
    url_input = st.text_input("Enter a URL to analyze")
    if st.button("Analyze"):
        if not url_input.strip():
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Scraping and analyzing..."):
                scraped = safe_scrape_url(url_input.strip())
                if "error" in scraped:
                    st.error(f"Failed to fetch URL: {scraped['error']}")
                else:
                    st.subheader("Scraped Summary")
                    st.write(f"**URL:** {scraped['url']}")
                    st.write(f"**Title:** {scraped.get('title','')}")
                    st.write(f"**Word Count:** {scraped.get('word_count',0)}")
                    fr = textstat.flesch_reading_ease(scraped.get('body_text','') or "")
                    st.write(f"**Flesch Reading Ease:** {fr:.2f}")
                    s_count = len(sent_tokenize(scraped.get('body_text','') or ""))
                    st.write(f"**Sentence Count:** {s_count}")
                    thin_flag = scraped.get('word_count',0) < 500
                    st.write(f"**Thin Content:** {thin_flag}")
                    # load embeddings for dataset
                    if 'embeddings' not in st.session_state:
                        # compute embeddings from extracted file
                        if os.path.exists(EXTRACTED_CSV):
                            ext = pd.read_csv(EXTRACTED_CSV)
                            model = SentenceTransformer('all-MiniLM-L6-v2')
                            emb = model.encode(ext['body_text'].fillna("").astype(str).tolist(), show_progress_bar=False)
                            st.session_state['embeddings'] = emb
                            st.session_state['df_features'] = ext
                    # embed the new text
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    new_emb = model.encode([scraped.get('body_text','')], show_progress_bar=False)
                    # similarity against dataset
                    embeddings = st.session_state.get('embeddings', None)
                    if embeddings is not None and len(embeddings)>0:
                        sims = cosine_similarity(new_emb, embeddings)[0]
                        top_idx = sims.argsort()[-3:][::-1]
                        top_matches = [(st.session_state['df_features'].loc[i,'url'], float(sims[i])) for i in top_idx]
                        st.subheader("Top 3 similar pages in dataset")
                        for u, s in top_matches:
                            st.write(f"- {u} — similarity: {s:.4f}")
                    else:
                        st.info("Dataset embeddings not available; cannot compute similarity.")
                    # load model and predict (be defensive: saved file may contain a dict/wrapped object)
                    if os.path.exists(MODEL_FILE):
                        try:
                            loaded = joblib.load(MODEL_FILE)
                        except Exception as e:
                            st.error(f"Failed to load model file: {e}")
                            loaded = None

                        clf = None
                        if loaded is not None:
                            # If the loaded object already has predict, use it
                            if hasattr(loaded, "predict"):
                                clf = loaded
                            # If a dict was saved, try common keys and values to find an estimator
                            elif isinstance(loaded, dict):
                                for key in ("model", "clf", "estimator", "pipeline", "classifier"):
                                    if key in loaded and hasattr(loaded[key], "predict"):
                                        clf = loaded[key]
                                        break
                                if clf is None:
                                    # fallback: pick the first value that looks like an estimator
                                    for v in loaded.values():
                                        if hasattr(v, "predict"):
                                            clf = v
                                            break

                        if clf is not None:
                            X_new = np.array([[scraped.get('word_count',0), s_count, fr]])
                            try:
                                pred = clf.predict(X_new)[0]
                                st.write(f"**Predicted Quality:** {pred}")
                            except Exception as e:
                                st.error(f"Failed to run prediction: {e}")
                        else:
                            st.info("Quality model file loaded but no usable estimator with `predict` was found. Train the model in the 'Quality Model' tab or re-save the estimator as the top-level object in the pickle file.")
                    else:
                        st.info("Quality model not found. Train the model in the 'Quality Model' tab first.")
    st.write("Note: Live scraping depends on the target site's robots policy and network accessibility.")

# Footer
st.markdown("---")
st.caption("SEO Content Quality & Duplicate Detector — built with Streamlit. Place your data.csv at /mnt/data/data.csv or upload in the Data Processing tab.")