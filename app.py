import streamlit as st
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- CONFIG ---
st.set_page_config(page_title="LinguaNova", page_icon="🌐", layout="centered")

# --- LANGUAGES ---
lang_map = {
    'bn': 'Bengali', 'en': 'English', 'fr': 'French', 'es': 'Spanish',
    'de': 'German', 'hi': 'Hindi', 'ja': 'Japanese', 'ko': 'Korean',
    'zh-cn': 'Chinese', 'ru': 'Russian', 'ar': 'Arabic'
}
reverse_lang_map = {v: k for k, v in lang_map.items()}

# --- LOAD MODEL ---
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- TRANSLATION FUNCTION ---
def translate(text, src_lang_name, tgt_lang_name):
    src_code = reverse_lang_map.get(src_lang_name, "en")
    tgt_code = reverse_lang_map.get(tgt_lang_name, "en")
    input_text = f"translate {src_code} to {tgt_code}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=1024, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- SIDEBAR ---
st.sidebar.title("Settings")
target_language = st.sidebar.selectbox("Translate to:", list(reverse_lang_map.keys()))

# --- UI ---
st.markdown("## 🌐 LinguaNova: Universal Translator")
st.markdown("Translate **any language** into your selected one. Type or upload — we decode the world for you.")

tab1, tab2 = st.tabs(["✍️ Type or Paste", "📤 Upload File"])

user_text = ""
with tab1:
    user_text = st.text_area("Enter text here (any language)", height=250)

with tab2:
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        file_text = uploaded_file.read().decode("utf-8")
        st.text_area("File content preview", value=file_text, height=250)
        user_text = file_text

# --- TRANSLATE BUTTON ---
if st.button("🔄 Translate"):
    if not user_text.strip():
        st.warning("Please provide some input.")
    else:
        with st.spinner("Detecting language and translating..."):
            try:
                detected_code = detect(user_text)
                detected_lang = lang_map.get(detected_code, "Unknown")
            except:
                detected_lang = "Unknown"

            st.info(f"Detected Language: **{detected_lang}**")

            if detected_lang == "Unknown":
                st.error("Could not detect language. Try entering more text.")
            else:
                translation = translate(user_text, detected_lang, target_language)
                st.success("Translation Complete:")
                st.text_area("Translated Text", value=translation, height=200)

                # Save to session history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append((user_text, translation))
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop(0)

# --- HISTORY ---
if 'history' in st.session_state:
    with st.expander("🕘 Translation History (Last 5)"):
        for i, (src, trans) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**{i+1}.** `{src[:40]}...` ➡️ `{trans[:60]}...`")

# --- FOOTER ---
st.markdown("---")
st.markdown("Created with love by **Sugar & ChatGPT**")
