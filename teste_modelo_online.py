import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import torch
import shap
from streamlit_shap import st_shap
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from google.cloud import storage
import os
import tempfile
import matplotlib

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
import os
import tempfile

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
import os
import tempfile

# Function to download model files from Google Cloud Storage
def download_model_files():
    # Load credentials directly from Streamlit secrets
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    # Create the Google Cloud Storage client using the credentials
    client = storage.Client(credentials=credentials, project=st.secrets["gcp_service_account"]["project_id"])
    bucket_name = "model_caed"  # Replace with your bucket name
    bucket = client.bucket(bucket_name)

    # Define the files to download and where to store them
    files = ["config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"]
    temp_model_dir = tempfile.mkdtemp()  # Temporary directory for model files

    # Download each file
    for file_name in files:
        blob = bucket.blob(file_name)
        blob.download_to_filename(os.path.join(temp_model_dir, file_name))
        st.write(f"Downloaded {file_name} from Google Cloud Storage to {temp_model_dir}")

    return temp_model_dir


# Download model files to a temporary directory
model_path = download_model_files()

# Initialize the model and tokenizer using the downloaded files
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=4)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prediction function
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    elif not isinstance(texts, list):
        raise ValueError("Unsupported format for 'texts' in 'predict_proba'")
    
    # Tokenize the text and convert to tensors
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=100
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Pass through the model and obtain logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate class probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.cpu().numpy()

# Streamlit interface
st.title("Análise sobre P. Pedag.")

# Text input field for the user to type a sentence
input_text = st.text_area("Digite a frase para classificação e análise do modelo (refl. sobre p. pedag.):")

# Button to execute classification and SHAP analysis
if st.button("Analisar"):
    if input_text.strip() == "":
        st.warning("Por favor, digite uma frase para análise.")
    else:
        # Display the entered text
        st.write("Frase para análise:", input_text)

        # Predict the class and display the probability of each class
        probabilities = predict_proba([input_text])[0]
        class_id = probabilities.argmax()
        st.write(f"Classe prevista: {class_id} \n \n - (Output/Classe 0) Valor 0 se apenas citou as práticas pedagógicas, sem justificar \n - (Output/Classe 1) Valor 1 se citou as práticas pedagógicas e justificou de maneira difusa  \n - (Output/Classe 2) Valor 2 se citou as práticas pedagógicas e apresentou um argumento estruturado do porquê do uso dela  \n - (Output/Classe 3) Valor 3 se citou as práticas pedagógicas, apresentou um argumento estruturado do porquê do uso dela e apresentou uma autoavaliação da própria prática")
        st.write("Probabilidades por classe:")
        for i, prob in enumerate(probabilities):
            st.write(f"Classe {i}: {prob:.2f}")

        # Generate SHAP explanation
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predict_proba, masker=masker)
        shap_values = explainer([input_text])

        st.write(
            f"Explicação para a frase na classe {class_id}: \n\n"
            "- **Setas vermelhas**: indicam as palavras que contribuem positivamente para classificar a frase nesta categoria.\n"
            "- **Setas azuis**: indicam as palavras que têm menor contribuição ou uma influência negativa para a classificação.\n"
            "- **Interatividade**: ao passar o mouse sobre uma seta, a palavra correspondente é sublinhada, e ao passar o mouse sobre a palavra, a seta correspondente é destacada.\n\n"
            "Essas setas vermelhas e azuis ajudam a entender quais palavras influenciam a predição e em qual direção.\n\n"
            "**Dica**: Se o fundo da janela estiver preto, clique em 'Settings' (configurações) no canto superior direito e altere para o modo de visualização 'Light' para melhorar a visibilidade das setas."
        )

        
        # Display SHAP text plot
        #st_shap(shap.plots.text(shap_values[0]), width=800, height=400)
        # Display SHAP text plot with additional styles
        shap_html = shap.plots.text(shap_values[0], display=False)

        # Define custom CSS to force colors on tokens
        custom_css = """
        <style>
        .shap span[data-shap] {
            color: inherit !important; /* Ensure SHAP token colors show */
        }
        </style>
        """

        # Combine custom CSS with the SHAP HTML output
        html_content = f"<html><head>{shap.getjs()}{custom_css}</head><body>{shap_html}</body></html>"

        # Display in Streamlit
        components.html(html_content, height=400, scrolling=True)

