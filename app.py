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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
import joblib
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

def parse_html_to_text(html):
    try:
        soup = BeautifulSoup(str(html), "html.parser")
        # remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        # Heuristic: main text - join all <p> tags else body text
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        if paragraphs:
            body = "\n\n".join(paragraphs)
        else:
            body = soup.get_text(separator=" ", strip=True)
        # clean multiple spaces
        body = " ".join(body.split())
        word_count = len(body.split())
        return title, body, word_count
    except Exception as e:
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
def compute_features(df_extracted):
    df = df_extracted.copy()
    # Normalize extracted dataframe: ensure title, body_text, word_count exist
    if 'body_text' not in df.columns:
        # Try to extract from html_content if present
        if 'html_content' in df.columns:
            titles = []
            bodies = []
            wcs = []
            for html in df['html_content'].fillna(""):
                t, b, wc = parse_html_to_text(html)
                titles.append(t)
                bodies.append(b)
                wcs.append(wc)
            df['title'] = titles
            df['body_text'] = bodies
            df['word_count'] = wcs
        else:
            # No html or body_text: create empty defaults
            df['title'] = df.get('title', pd.Series([""] * len(df)))
            df['body_text'] = df.get('body_text', pd.Series([""] * len(df)))
            df['word_count'] = df.get('word_count', pd.Series([0] * len(df)))
    else:
        # Ensure body_text is string
        df['body_text'] = df['body_text'].fillna("").astype(str)
        # Ensure title exists
        if 'title' not in df.columns:
            df['title'] = ""
        # Compute word_count if missing or zero
        if 'word_count' not in df.columns or df['word_count'].isnull().all():
            df['word_count'] = df['body_text'].apply(lambda t: len(t.split()) if isinstance(t, str) and t.strip() else 0)

    # sentence_count
    df["sentence_count"] = df["body_text"].apply(lambda t: len(sent_tokenize(t)) if isinstance(t, str) and t.strip() else 0)
    # flesch_reading_ease
    def safe_flesch(text):
        try:
            return textstat.flesch_reading_ease(text) if text and text.strip() else 0.0
        except:
            return 0.0
    df["flesch_reading_ease"] = df["body_text"].apply(safe_flesch)
    # TF-IDF top keywords (top 5)
    corpus = df["body_text"].fillna("").astype(str).tolist()
    if any(corpus):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        tfidf = vectorizer.fit_transform(corpus)
        feature_names = np.array(vectorizer.get_feature_names_out())
        top_keywords = []
        for row in tfidf:
            if isinstance(row, np.ndarray):
                row = row
            else:
                row = row.toarray().flatten()
            topn_idx = row.argsort()[-5:][::-1]
            kws = [feature_names[idx] for idx in topn_idx if row[idx] > 0]
            top_keywords.append(", ".join(kws))
    else:
        top_keywords = [""] * len(df)
    df["top_keywords"] = top_keywords
    # embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df["body_text"].fillna("").astype(str).tolist(), show_progress_bar=True)
    df_embeddings = np.array(embeddings)
    return df, df_embeddings

@st.cache_data
def compute_similarity_matrix(embeddings):
    if embeddings is None or len(embeddings)==0:
        return np.array([[]])
    sim = cosine_similarity(embeddings)
    return sim

def train_quality_model(df_features):
    # synthetic labels
    def label_row(r):
        wc = r["word_count"]
        fr = r["flesch_reading_ease"]
        if (wc > 1500) and (50 <= fr <= 70):
            return "High"
        elif (wc < 500) or (fr < 30):
            return "Low"
        else:
            return "Medium"
    df = df_features.copy()
    df["label"] = df.apply(label_row, axis=1)
    X = df[["word_count", "sentence_count", "flesch_reading_ease"]].fillna(0)
    y = df["label"]
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(y.unique())>1 else None)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # save
    joblib.dump(clf, MODEL_FILE)
    # metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=["High","Medium","Low"])
    feat_imp = clf.feature_importances_
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
tabs = ["Data Processing", "Feature Analysis", "Duplicate Detector", "Quality Model", "Live URL Analyzer"]
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
            sns.boxplot(x=df_feat['flesch_reading_ease'].fillna(0), ax=ax2)
            st.pyplot(fig2)
        st.subheader("Top Keywords Word Cloud")
        # build combined keywords text
        kw_text = " ".join(df_feat['top_keywords'].fillna("").astype(str).tolist())
        if kw_text.strip():
            wc = WordCloud(width=800, height=400).generate(kw_text)
            fig3, ax3 = plt.subplots(figsize=(10,4))
            ax3.imshow(wc, interpolation='bilinear')
            ax3.axis('off')
            st.pyplot(fig3)
        else:
            st.info("No keywords available to build a word cloud yet.")

elif choice == "Duplicate Detector":
    st.header("3. Duplicate & Thin Content Detector")
    st.markdown("Use this page to find duplicates in your dataset or compare a live URL against the dataset.")

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

    # helper: validate embeddings
    def embeddings_valid(emb):
        try:
            emb = np.array(emb)
            if emb.size == 0:
                return False
            if emb.ndim != 2:
                return False
            # more than one vector
            if emb.shape[0] < 1:
                return False
            # check if all rows are nearly identical -> bad encoding
            if emb.shape[0] > 1 and np.allclose(emb, emb[0]):
                return False
            return True
        except Exception:
            return False

    # If embeddings look invalid, recompute them explicitly
    if not embeddings_valid(embeddings):
        st.warning("Dataset embeddings look invalid or uniform — recomputing embeddings now.")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df_feat['body_text'].fillna("").astype(str).tolist(), show_progress_bar=True)
        embeddings = np.array(embeddings)
        st.session_state['embeddings'] = embeddings

    # Thin content table
    st.subheader("Thin Content (word_count < 500)")
    thin = df_feat[df_feat['word_count'] < 500][['url','title','word_count']]
    st.dataframe(thin)

    # Live-URL based duplicate detection (preferred)
    st.subheader("Compare a Live URL against the dataset")
    live_url = st.text_input("Enter a live URL to compare (optional)")
    if st.button("Analyze Live URL"):
        if not live_url.strip():
            st.error("Please enter a valid URL to analyze.")
        else:
            with st.spinner("Scraping and computing similarity..."):
                scraped = safe_scrape_url(live_url.strip())
                if 'error' in scraped:
                    st.error(f"Failed to fetch URL: {scraped['error']}")
                else:
                    # ensure embeddings exist and valid
                    if not embeddings_valid(embeddings):
                        st.error("Dataset embeddings are not available or invalid; compute features first.")
                    else:
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        new_emb = model.encode([scraped.get('body_text','') or ""], show_progress_bar=False)
                        sims = cosine_similarity(np.array(new_emb), embeddings)[0]
                        top_idx = sims.argsort()[-10:][::-1]
                        match_rows = []
                        for i in top_idx:
                            match_rows.append({'url': df_feat.loc[i,'url'], 'title': df_feat.loc[i,'title'], 'similarity': float(sims[i]), 'word_count': int(df_feat.loc[i,'word_count'])})
                        st.subheader("Top matches from dataset")
                        st.dataframe(pd.DataFrame(match_rows))

    st.markdown("---")
    # Dataset vs Dataset duplicate detection (opt-in)
    st.subheader("Find duplicate pairs within dataset (optional)")
    if st.button("Find duplicate pairs in dataset"):
        if not embeddings_valid(embeddings):
            st.error("Dataset embeddings are not available or invalid; compute features first.")
        else:
            with st.spinner("Computing similarity matrix and extracting pairs..."):
                sim_matrix = compute_similarity_matrix(embeddings)
                st.write("Adjust the similarity threshold to list pairs considered duplicates.")
                thresh = st.slider("Similarity Threshold", min_value=0.70, max_value=1.00, value=0.85, step=0.01, key='dup_thresh')
                pairs = []
                n = sim_matrix.shape[0]
                for i in range(n):
                    for j in range(i+1, n):
                        if sim_matrix[i,j] >= thresh:
                            pairs.append((df_feat.loc[i,'url'], df_feat.loc[j,'url'], float(sim_matrix[i,j])))
                if pairs:
                    dup_df = pd.DataFrame(pairs, columns=["url1","url2","similarity"]).sort_values("similarity", ascending=False)
                    st.dataframe(dup_df)
                    st.download_button("Download duplicate pairs CSV", data=dup_df.to_csv(index=False).encode('utf-8'), file_name="duplicate_pairs.csv")
                else:
                    st.info("No duplicate pairs found at this threshold.")
                st.subheader("Similarity Matrix Heatmap")
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(sim_matrix, ax=ax)
                st.pyplot(fig)

elif choice == "Quality Model":
    st.header("4. Content Quality Scoring (Machine Learning)")
    # Ensure features
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
    if st.button("Train Quality Model (RandomForest)"):
        with st.spinner("Training model..."):
            clf, acc, f1, cm, feat_imp, df_with_labels = train_quality_model(df_feat)
            st.session_state['quality_model'] = clf
            st.session_state['quality_metrics'] = {"accuracy":acc, "f1":f1, "confusion_matrix":cm, "feature_importances":feat_imp}
        st.success("Model trained and saved to disk.")
    if os.path.exists(MODEL_FILE):
        if 'quality_model' not in st.session_state:
            try:
                st.session_state['quality_model'] = joblib.load(MODEL_FILE)
            except:
                pass
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

elif choice == "Live URL Analyzer":
    st.header("5. Real-Time URL Analyzer")
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