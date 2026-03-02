import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from supabase import create_client
import uuid

# Supabase Secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Modell laden
@st.cache_resource
def load_ml_model():
    model = load_model("keras_model.h5", compile=False)
    labels = open("labels.txt", "r").readlines()
    return model, labels

model, class_names = load_ml_model()

# Farb-Erkennung
def detect_color(image):
    image = image.resize((100, 100))
    img_array = np.array(image)
    avg_color = img_array.mean(axis=0).mean(axis=0)
    r, g, b = avg_color

    if r > 150 and g < 100:
        return "Rot"
    elif g > 150:
        return "Grün"
    elif b > 150:
        return "Blau"
    else:
        return "Unbekannt"

# Klassifikation
def classify_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index][2:].strip()
    confidence = float(prediction[0][index])

    return class_name, confidence

# Upload zu Supabase
def upload_image(file):
    file_bytes = file.getvalue()
    file_name = f"{uuid.uuid4()}.jpg"

    supabase.storage.from_("images").upload(file_name, file_bytes)
    public_url = supabase.storage.from_("images").get_public_url(file_name)

    return public_url

# UI
st.title("Lost & Found KI")

menu = st.sidebar.selectbox("Menü", ["Suchen", "Fund einstellen"])

if menu == "Fund einstellen":
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image)

        if st.button("Analysieren & Speichern"):
            item, confidence = classify_image(image)
            color = detect_color(image)

            image_url = upload_image(uploaded_file)

            supabase.table("items").insert({
                "item": item,
                "color": color,
                "image_url": image_url
            }).execute()

            st.success(f"Gespeichert: {item} - {color}")

if menu == "Suchen":
    search = st.text_input("Suche nach Kleidung")

    if st.button("Suchen"):
        result = supabase.table("items").select("*").ilike("item", f"%{search}%").execute()

        for item in result.data:
            st.write(f"{item['item']} - {item['color']}")
            st.image(item["image_url"])
