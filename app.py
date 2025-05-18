import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Define supported language codes and their full names
LANGUAGES = {
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ur': 'Urdu',
    'pt': 'Portuguese',
    'nl': 'Dutch'
}

# Create reverse mapping for language selection
LANGUAGE_CODES = {v: k for k, v in LANGUAGES.items()}

# Function to load the appropriate MarianMT model and tokenizer
@st.cache_resource
def load_model(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Streamlit app layout
st.title("🌍 Multilingual Translator")
st.markdown("Translate text between multiple languages using Hugging Face's MarianMT models.")

# Language selection
col1, col2 = st.columns(2)
with col1:
    src_language = st.selectbox("Select source language:", list(LANGUAGES.values()), index=0)
with col2:
    tgt_language = st.selectbox("Select target language:", list(LANGUAGES.values()), index=1)

# Text input
text_to_translate = st.text_area("Enter text to translate:", height=150)

# Translate button
if st.button("Translate"):
    if not text_to_translate.strip():
        st.warning("Please enter text to translate.")
    else:
        src_code = LANGUAGE_CODES[src_language]
        tgt_code = LANGUAGE_CODES[tgt_language]
        try:
            tokenizer, model = load_model(src_code, tgt_code)
            inputs = tokenizer(text_to_translate, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.success("Translation:")
            st.write(translated_text)
        except Exception as e:
            st.error(f"Translation failed: {e}")
