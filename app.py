import streamlit as st
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Title & Description
st.set_page_config(page_title="LinguaNova", page_icon="馃實", layout="centered")

st.markdown("## 馃實 LinguaNova: Universal Translator")
st.markdown("Translate **any language** into English or your choice. Type or upload 鈥� we decode the world for you.")

# Caching model
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Language map
lang_map = {
    'bn': 'Bengali',
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'hi': 'Hindi',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-cn': 'Chinese',
    'ru': 'Russian',
    'ar': 'Arabic'
}

# Translation function
def translate(text, src_lang, tgt_lang="English"):
    input_text = f"translate {src_lang} to {tgt_lang}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=1024,
        num_beams=4,
        early_stopping=True
    )
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

# Sidebar
st.sidebar.title("Settings")
target_language = st.sidebar.selectbox("Translate to:", ["English", "Bengali", "French", "Spanish", "German", "Hindi"])

# Main Input Area
tab1, tab2 = st.tabs(["鉁嶏笍 Type or Paste", "馃搧 Upload File"])

with tab1:
    user_text = st.text_area("Enter text here (any language)", height=250)

with tab2:
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        file_text = uploaded_file.read().decode("utf-8")
        st.text_area("File content preview", value=file_text, height=250)
        user_text = file_text

if st.button("馃寪 Translate"):
    if not user_text.strip():
        st.warning("Please provide some input.")
    else:
        with st.spinner("Detecting language and translating..."):
            try:
                detected_code = detect(user_text)
                detected_lang = lang_map.get(detected_code, detected_code)
            except:
                detected_lang = "Unknown"

            st.info(f"Detected Language: **{detected_lang}**")

            if detected_lang == "Unknown":
                st.error("Could not detect language. Try entering more text.")
            else:
                translation = translate(user_text, detected_lang, target_language)
                st.success("Translation Complete:")
                st.text_area("Translated Text", value=translation, height=200)

# Optional: Add history
if 'history' not in st.session_state:
    st.session_state.history = []

if user_text:
    st.session_state.history.append((user_text, translation))
    if len(st.session_state.history) > 5:
        st.session_state.history.pop(0)

with st.expander("馃晿 Translation History (Last 5)"):
    for i, (src, trans) in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**{i+1}.** `{src[:40]}...` 鈫� `{trans[:60]}...`")

# Footer
st.markdown("---")
st.markdown("Created with love by **Sugar & ChatGPT**")
