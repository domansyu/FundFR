import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Fundbüro KI", layout="centered")

st.title("Fundbüro KI (Hugging Face)")

# Modell laden
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

classifier = load_model()

# Upload
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", width=300)

    if st.button("Analysieren"):
        with st.spinner("KI analysiert..."):
            result = classifier(image)

        st.write("### Ergebnis:")
        for r in result[:3]:
            st.write(f"{r['label']} ({round(r['score']*100,2)}%)")
