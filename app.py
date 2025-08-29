import streamlit as st
import pandas as pd
from nutrition_extractor import run_pipeline

st.title("Nutritional Label Extractor")
st.write("Upload a food packet image to extract nutritional information.")

# Load API key from secrets
API_KEY = st.secrets["gemini"]["api_key"]

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and API_KEY:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Extracting nutritional information..."):
        parsed_data, nutrition_table = run_pipeline(uploaded_file.read(), api_key=API_KEY)

    st.success("Extraction complete!")

    st.subheader("Extracted Nutritional Information (Table)")
    df = pd.DataFrame(nutrition_table)
    st.dataframe(df, use_container_width=True)

    st.subheader("Raw JSON Output")
    st.json(parsed_data)

elif uploaded_file is not None and not API_KEY:
    st.warning("API Key not found. Please add it to `.streamlit/secrets.toml`.")

else:
    st.info("Please upload an image to begin.")
