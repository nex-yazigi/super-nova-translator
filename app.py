import streamlit as st
from googletrans import Translator, LANGUAGES
import nltk
from nltk.tokenize import sent_tokenize
from gtts import gTTS
import base64
from io import BytesIO

# Download NLTK data
nltk.download('punkt')

# Initialize Translator
translator = Translator()

# Set page config
st.set_page_config(page_title="LinguaNova GT+", page_icon="🌍", layout="centered")

st.title("🌍 LinguaNova GT+ (Google Translate Supreme)")
st.markdown("Translate up to 100,000 characters, hear it, copy it, download it — multilingual magic!")

# Language selection
lang_names = [LANGUAGES[key].capitalize() for key in LANGUAGES]
lang_codes = {v.capitalize(): k for k, v in LANGUAGES.items()}

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("From:", ["Auto Detect"] + sorted(lang_names))
with col2:
    tgt_langs = st.multiselect("To (Select multiple):", sorted(lang_names), default=["English"])

# Input tabs
tab1, tab2 = st.tabs(["✍️ Paste Text", "📄 Upload File"])
text = ""

with tab1:
    input_text = st.text_area("Paste your content here", height=300, max_chars=100000)
    text = input_text.strip()

with tab2:
    uploaded_file = st.file_uploader("Upload a .txt file (max ~100k characters)", type=["txt"])
    if uploaded_file:
        file_text = uploaded_file.read().decode("utf-8")[:100000]
        st.text_area("File Content Preview", value=file_text, height=300)
        text = file_text.strip()

# Sentence splitting function
def split_into_chunks(text, max_chars=4500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_chars:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Text-to-Speech and audio download helpers
def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def get_audio_download_link(mp3_fp, filename):
    b64 = base64.b64encode(mp3_fp.read()).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Audio</a>'
    return href

# Translation button and process
if st.button("🌐 Translate"):
    if not text:
        st.warning("Please provide some text or upload a file.")
    else:
        with st.spinner("Translating and processing..."):
            try:
                src = 'auto' if src_lang == "Auto Detect" else lang_codes[src_lang]
                chunks = split_into_chunks(text)
                results = {}

                for tgt_lang in tgt_langs:
                    dest = lang_codes[tgt_lang]
                    translated_chunks = []
                    for chunk in chunks:
                        trans = translator.translate(chunk, src=src, dest=dest)
                        translated_chunks.append(trans.text)
                    final_result = "\n\n".join(translated_chunks)
                    results[tgt_lang] = final_result

                for lang in results:
                    st.subheader(f"Translated to {lang}")
                    st.text_area("Result", results[lang], height=250, key=f"area_{lang}")
                    st.download_button("⬇️ Download TXT", results[lang], file_name=f"translated_{lang}.txt")
                    mp3_fp = text_to_speech(results[lang], lang_codes[lang])
                    st.audio(mp3_fp, format="audio/mp3")
                    st.markdown(get_audio_download_link(mp3_fp, f"speech_{lang}.mp3"), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# Footer
st.markdown("---")
st.caption("Created with care by Sugar & ChatGPT")
