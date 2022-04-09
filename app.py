import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
import yaml

from PIL import Image
from yaml.loader import SafeLoader

with open('models/app.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

MODEL_PATH_FAMILY = data['MODEL_PATH_FAMILY']
MODEL_PATH_MANUFACTURER = data['MODEL_PATH_MANUFACTURER']
MODEL_PATH_VARIANT = data['MODEL_PATH_VARIANT']
SVM_PATH_FAMILY = data['SVM_PATH_FAMILY']
SVM_PATH_MANUFACTURER = data['SVM_PATH_MANUFACTURER']
SVM_PATH_VARIANT = data['SVM_PATH_VARIANT']
IMAGE_WIDTH = data['IMAGE_WIDTH']
IMAGE_HEIGHT = data['IMAGE_HEIGHT']
IMAGE_DEPTH = data['IMAGE_DEPTH']

with open('models/cat.yaml') as f:
    data_cat = yaml.load(f, Loader=SafeLoader)
    
CAT_FAMILY = data_cat['categories_family']
CAT_MANUFACTURER = data_cat['categories_manufacturer']
CAT_VARIANT = data_cat['categories_variant']

def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)
    

def predict_image(path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    print(images.shape)
    prediction_vector = model.predict(images)
    predict_proba = np.max(prediction_vector)
    predicted_classes = np.argmax(prediction_vector, axis=1)
    return predicted_classes[0], predict_proba, prediction_vector


def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)
    
model = st.sidebar.radio('Quelle méthode de prédiction voulez-vous utiliser ?',("Réseaux de neurones", "SVM"))
if model =="Réseaux de neurones":
    option = st.sidebar.selectbox('Quel modèle voulez-vous utiliser ?',("Choisissez un modèle", MODEL_PATH_FAMILY, MODEL_PATH_MANUFACTURER, MODEL_PATH_VARIANT))

    if option != "Choisissez un modèle": 
        model = load_model(option)
        model.summary()

        st.title("Identification d'avion")

        uploaded_file = st.file_uploader("Charger une image d'avion") #, accept_multiple_files=True)#

        if uploaded_file:
            loaded_image = load_image(uploaded_file)
            st.image(loaded_image)


        predict_btn = st.button("Identifier", disabled=(uploaded_file is None))
        if predict_btn:
            prediction = predict_image(uploaded_file, model)
            if option == MODEL_PATH_FAMILY  :
                st.write(f"C'est un : {CAT_FAMILY[prediction[0]]}")
                st.write(f"Avec une probabilité de : {round(prediction[1]*100,2)}%")
                st.write(f"Le code correspondant est le : {(prediction[0])}")
                st.title("Graphique de la distribution des probabilités")
                st.bar_chart(pd.DataFrame(prediction[2]).T)
                st.write("Légende du graphique")
                pred = pd.DataFrame({'catégorie':CAT_FAMILY, 'code':range(0,len(CAT_FAMILY))}).set_index('catégorie')
                st.write(pred)

            if option == MODEL_PATH_MANUFACTURER  :
                st.write(f"C'est un : {CAT_MANUFACTURER[prediction[0]]}")   
                st.write(f"Avec une probabilité de : {round(prediction[1]*100,2)}%")
                st.write(f"Le code correspondant est le : {(prediction[0])}")
                st.title("Graphique de la distribution des probabilités")
                st.bar_chart(pd.DataFrame(prediction[2]).T)
                st.write("Légende du graphique")
                pred=pd.DataFrame({'catégorie':CAT_MANUFACTURER,
                                   'code':range(0,len(CAT_MANUFACTURER))}).set_index('catégorie')
                st.write(pred)

            if option == MODEL_PATH_VARIANT  :
                st.write(f"C'est un : {CAT_VARIANT[prediction[0]]}")   
                st.write(f"Avec une probabilité de : {round(prediction[1]*100,2)}%")
                st.write(f"Le code correspondant est le : {(prediction[0])}")
                st.title("Graphique de la distribution des probabilités")
                st.bar_chart(pd.DataFrame(prediction[2]).T)
                st.write("Légende du graphique")
                pred=pd.DataFrame({'catégorie':CAT_VARIANT,
                                   'code':range(0,len(CAT_VARIANT))}).set_index('catégorie')
                st.write(pred)
if model =="SVM":
    st.write("La prédiction est encore en construction...merci de patienter ! ")
