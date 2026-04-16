import streamlit as st
import whisper
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import spacy
from transformers import pipeline
from textblob import TextBlob
from pdf_gen import generate_pdf_report
import re

st.set_page_config(page_title="Meeting Minutes Generator", page_icon="🎤", layout="wide")

# ======================== STYLING ========================
st.markdown("""
    <style>
        body {
            background: #f5f7fa;
        }

        .stApp {
            background: #f5f7fa;
        }
        .main-header-full {
            width: 100%;
            background: #1e2a3a;
            color: white;
            padding: 18px;
            font-size: 22px;
            font-weight: 600;
            border-radius: 8px;
            text-align: left;
            margin-bottom: 15px;
        }

        .section-header {
            background: #1e293b;
            color: white;
            padding: 16px;
            border-radius: 6px;
            margin: 20px 0;
            font-size: 22px;
            font-weight: 600;
        }

        .insight-card {
            background: white;
            border-left: 4px solid #2563eb;
            padding: 12px;
            border-radius: 6px;
            margin: 10px 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        .action-item-card {
            background: #eef2ff;
            border-left: 4px solid #2563eb;
            padding: 12px;
            border-radius: 6px;
            margin: 10px 0;
        }

        .metric-box {
            background: #1e293b;
            color: white;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }

        .success-badge {
            background: #16a34a;
            color: white;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }

        .neutral-badge {
            background: #eab308;
            color: black;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }

        .negative-badge {
            background: #dc2626;
            color: white;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }

        button {
            border-radius: 6px !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ======================== MODEL LOADING (CACHED) ========================
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_summarization_pipeline():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

@st.cache_resource
def load_keyword_pipeline():
    return pipeline("feature-extraction", model="bert-base-uncased", device=-1)

@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

# ======================== UTILITY FUNCTIONS ========================
def extract_action_items(text, nlp):
    """Extract action items using spaCy NER and pattern matching"""
    doc = nlp(text)
    
    action_items = []
    sentences = text.split('.')
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        lower_sent = sent.lower()
        if any(greet in lower_sent for greet in ["good morning", "good afternoon", "hello everyone"]):
            continue
        if ('will' in lower_sent or 'needs to' in lower_sent or 'is assigned to' in lower_sent):
            person = None
            for ent in nlp(sent).ents:
                if ent.label_ == "PERSON":
                    person = ent.text
                    break
            if not person:
                words = sent.split()
                if words:
                    first_word = words[0]
                    if first_word[0].isupper():
                        person = first_word
            
            deadline = extract_deadline(sent)
            priority = determine_priority(sent)
            
            action_items.append({
                'Person': person or 'Unassigned',
                'Task': sent.strip(),
                'Deadline': deadline or 'N/A',
                'Priority': priority
            })
    
    return action_items

def extract_deadline(text):
    patterns = [
        r'by\s+([A-Za-z]+\s+\d{1,2},\s+\d{4}(?:\s+at\s+\d{1,2}\s*(?:AM|PM))?)',
        r'before\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',
        r'by\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None

def determine_priority(text):
    """Determine priority level"""
    urgent_keywords = ['urgent', 'asap', 'critical', 'immediately', 'today', 'now']
    high_keywords = ['important', 'must', 'required', 'deadline']
    
    lower_text = text.lower()
    if any(kw in lower_text for kw in urgent_keywords):
        return 'Urgent'
    elif any(kw in lower_text for kw in high_keywords):
        return 'High'
    else:
        return 'Medium'

def extract_keywords(text):
    """Extract keywords from text"""
    blob = TextBlob(text)
    nouns = [word for word, tag in blob.tags if tag.startswith('NN')]
    keywords = list(set(nouns))[:10]
    return keywords if keywords else ['meeting', 'discussion']

def get_sentiment(text):
    """Analyze sentiment of text"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "Positive 😊", polarity
        elif polarity < -0.1:
            return "Negative 😞", polarity
        else:
            return "Neutral 😐", polarity
    except:
        return "Neutral 😐", 0

def highlight_important_sentences(text):
    """Highlight decision/important sentences"""
    decision_keywords = ['decided', 'agreed', 'decided to', 'must', 'required', 'will', 'action item', 'critical', 'approved', 'rejected']
    
    sentences = text.split('.')
    important = []
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if any(keyword in sent.lower() for keyword in decision_keywords):
            important.append(sent)
    
    return important[:5]

def generate_summary(text, summarization_model):
    """Generate summary using transformer"""
    try:
        words = text.split()
        if len(words) < 50:
            return "Text too short for summarization. Summary: " + text[:100]
        
        max_length = max(50, len(words) // 4)
        min_length = max(30, len(words) // 8)
        
        summary = summarization_model(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summary: {text[:200]}..."

# ======================== MAIN APP ========================
def main():
    st.sidebar.title("Meeting AI")
    st.sidebar.caption("NLP Powered System")

    page = st.sidebar.radio("Navigation", [
        "Upload",
        "Transcript",
        "Summary",
        "Action Items",
        "Insights",
        "Export"
])
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <h1 style='
            background-color:#1e2a3a;
            color:white;
            padding:20px;
            border-radius:10px;
            text-align:left;
            font-size:28px;
            font-weight:700;
        '>
        🎤 Automatic Meeting Minutes Generator using NLP
        </h1>
        """, unsafe_allow_html=True)
        st.markdown("*Transform your meetings into actionable insights with AI-powered intelligence*")
    
    st.divider()
    
    # Initialize session state
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'action_items' not in st.session_state:
        st.session_state.action_items = []
    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = None
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []
    
    # ======================== UPLOAD SECTION ========================
    if page == "Upload":
        with st.container():
            st.markdown('<div class="section-header">Upload Audio File</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                uploaded_file = st.file_uploader(
                    "Drag and drop your audio file here",
                    type=["wav", "mp3", "flac", "ogg", "m4a"],
                    label_visibility="collapsed"
                )

            with col2:
                st.info("Supported formats: WAV, MP3, FLAC, OGG, M4A\nMax size: 200MB")

            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")

                # Process audio
                with st.spinner("Transcribing audio..."):
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)

                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Transcribe
                    whisper_model = load_whisper_model()
                    result = whisper_model.transcribe(temp_path)

                    st.session_state.transcript = result["text"]

                    # Cleanup
                    os.remove(temp_path)

                    st.success("Transcription complete!")
    # ======================== TRANSCRIPT SECTION ========================
    if st.session_state.transcript:
        st.divider()
        with st.container():
            st.markdown('<div class="section-header">📝 Meeting Transcript</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                edited_transcript = st.text_area(
                    "Edit transcript if needed:",
                    value=st.session_state.transcript,
                    height=200,
                    label_visibility="collapsed"
                )
                st.session_state.transcript = edited_transcript
            
            with col2:
                word_count = len(st.session_state.transcript.split())
                st.metric("📊 Word Count", word_count)
                st.metric("⏱️ Estimated Duration", f"{word_count // 130} min")
        
        # ======================== SUMMARY SECTION ========================
        st.divider()
        with st.container():
            st.markdown('<div class="section-header">✨ Smart Summary</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Generate Summary", use_container_width=True, type="primary"):
                with st.spinner("⏳ Generating summary..."):
                    summarization_model = load_summarization_pipeline()
                    st.session_state.summary = generate_summary(st.session_state.transcript, summarization_model)
                    st.success("✅ Summary generated!")
            
            if st.session_state.summary:
                st.info(st.session_state.summary)
                
                # Key Highlights
                st.subheader("📌 Key Highlights")
                highlights = highlight_important_sentences(st.session_state.transcript)
                
                for i, highlight in enumerate(highlights, 1):
                    st.markdown(f"• {highlight}")
        
        # ======================== ACTION ITEMS SECTION ========================
        st.divider()
        with st.container():
            st.markdown('<div class="section-header">✅ Action Items</div>', unsafe_allow_html=True)
            
            if st.button("🔍 Extract Action Items", use_container_width=True, type="primary"):
                with st.spinner("🔎 Extracting action items..."):
                    nlp = load_spacy_model()
                    st.session_state.action_items = extract_action_items(st.session_state.transcript, nlp)
                    st.success(f"✅ Extracted {len(st.session_state.action_items)} action items")
            
            if st.session_state.action_items:
                df_actions = pd.DataFrame(st.session_state.action_items)
                st.dataframe(df_actions, use_container_width=True, hide_index=True)
        
        # ======================== INSIGHTS SECTION ========================
        st.divider()
        with st.container():
            st.markdown('<div class="section-header">🧠 Meeting Intelligence</div>', unsafe_allow_html=True)
            
            if st.button("💡 Analyze Insights", use_container_width=True, type="primary"):
                with st.spinner("🔍 Analyzing insights..."):
                    # Sentiment
                    sentiment, polarity = get_sentiment(st.session_state.transcript)
                    st.session_state.sentiment = sentiment
                    
                    # Keywords
                    st.session_state.keywords = extract_keywords(st.session_state.transcript)
                    
                    st.success("✅ Analysis complete!")
            
            if st.session_state.sentiment or st.session_state.keywords:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("😊 Sentiment Analysis")
                    sentiment_text = st.session_state.sentiment or "Click analyze button"
                    
                    if "Positive" in sentiment_text:
                        st.markdown('<span class="success-badge">POSITIVE</span>', unsafe_allow_html=True)
                    elif "Negative" in sentiment_text:
                        st.markdown('<span class="negative-badge">NEGATIVE</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="neutral-badge">NEUTRAL</span>', unsafe_allow_html=True)
                    
                    st.write(sentiment_text)
                
                with col2:
                    st.subheader("🏷️ Key Topics")
                    if st.session_state.keywords:
                        keywords_text = ", ".join(st.session_state.keywords)
                        st.info(keywords_text)
            
            # Important Sentences
            st.subheader("💬 Important Statements")
            if st.session_state.transcript:
                important = highlight_important_sentences(st.session_state.transcript)
                if important:
                    for imp in important:
                        st.markdown(f'<div class="insight-card">💭 {imp}</div>', unsafe_allow_html=True)
                else:
                    st.info("No decision statements found in transcript")
        
        # ======================== EXPORT SECTION ========================
        st.divider()
        with st.container():
            st.markdown('<div class="section-header">📥 Export Report</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Download as TXT", use_container_width=True):
                    txt_content = f"""
MEETING MINUTES REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRANSCRIPT:
{st.session_state.transcript}

SUMMARY:
{st.session_state.summary}

ACTION ITEMS:
"""
                    if st.session_state.action_items:
                        for item in st.session_state.action_items:
                            txt_content += f"\n- Person: {item['Person']}\n  Task: {item['Task']}\n  Deadline: {item['Deadline']}\n  Priority: {item['Priority']}\n"
                    
                    st.download_button(
                        label="✅ Download TXT",
                        data=txt_content,
                        file_name=f"meeting_minutes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("📊 Download as CSV", use_container_width=True):
                    if st.session_state.action_items:
                        df = pd.DataFrame(st.session_state.action_items)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="✅ Download CSV",
                            data=csv,
                            file_name=f"action_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No action items to export")
            
            with col3:
                if st.button("📑 Download as PDF", use_container_width=True):
                    try:
                        pdf_data = generate_pdf_report(
                            transcript=st.session_state.transcript,
                            summary=st.session_state.summary,
                            action_items=st.session_state.action_items,
                            sentiment=st.session_state.sentiment,
                            keywords=st.session_state.keywords
                        )
                        st.download_button(
                            label="✅ Download PDF",
                            data=pdf_data,
                            file_name=f"meeting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF generation in progress. Error: {str(e)}")
        
        st.divider()
        st.markdown("---")
        st.markdown("*Powered by OpenAI Whisper, Hugging Face Transformers & spaCy NLP*")
        st.markdown("*🚀 Production-Ready Meeting Intelligence Platform*")

if __name__ == "__main__":
    main()
