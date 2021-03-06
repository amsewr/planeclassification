import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
import yaml
import pickle

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

def predict_image_bySVM(path, model):
    """Predict plane identification from image thanks to a SVM
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (sklearn.models): SVM sklearn model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    print(images.shape)
    predicted_classes = model.predict(images)
    return predicted_classes[0]


def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)

def load_model_svm(path):
    """Load sklearn model for prediction
    """
    return pickle.load(open(path, 'rb'))

st.title("Identification d'avion")
uploaded_file = st.file_uploader("Charger une image d'avion") 
if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)
    
method = st.sidebar.radio('Quelle m??thode de pr??diction voulez-vous utiliser ?',("R??seaux de neurones", "SVM"))

if method =="R??seaux de neurones":
    option = st.sidebar.selectbox('Que voulez-vous identifier ?',("S??lectionner votre cible", MODEL_PATH_FAMILY, MODEL_PATH_MANUFACTURER, MODEL_PATH_VARIANT))
    
    if option =="S??lectionner votre cible":
        st.write("S??lectionner votre cible")
    
    if option != "S??lectionner votre cible": 
        model = load_model(option)
        model.summary()

        predict_btn = st.button("Identifier", disabled=(uploaded_file is None))
        if predict_btn:
            prediction = predict_image(uploaded_file, model)
            if option == MODEL_PATH_FAMILY  :
                st.write(f"C'est un : {CAT_FAMILY[prediction[0]]}")
                st.write(f"Avec une probabilit?? de : {round(prediction[1]*100,2)}%")
                st.write(f"Le code correspondant est le : {(prediction[0])}")
                st.title("Graphique de la distribution des probabilit??s")
                st.bar_chart(pd.DataFrame(prediction[2]).T)
                st.write("L??gende du graphique")
                pred = pd.DataFrame({'cat??gorie':CAT_FAMILY, 'code':range(0,len(CAT_FAMILY))}).set_index('cat??gorie')
                st.write(pred)

            if option == MODEL_PATH_MANUFACTURER  :
                st.write(f"C'est un : {CAT_MANUFACTURER[prediction[0]]}")   
                st.write(f"Avec une probabilit?? de : {round(prediction[1]*100,2)}%")
                st.write(f"Le code correspondant est le : {(prediction[0])}")
                st.title("Graphique de la distribution des probabilit??s")
                st.bar_chart(pd.DataFrame(prediction[2]).T)
                st.write("L??gende du graphique")
                pred=pd.DataFrame({'cat??gorie':CAT_MANUFACTURER,
                                   'code':range(0,len(CAT_MANUFACTURER))}).set_index('cat??gorie')
                st.write(pred)

            if option == MODEL_PATH_VARIANT  :
                st.write(f"C'est un : {CAT_VARIANT[prediction[0]]}")   
                st.write(f"Avec une probabilit?? de : {round(prediction[1]*100,2)}%")
                st.write(f"Le code correspondant est le : {(prediction[0])}")
                st.title("Graphique de la distribution des probabilit??s")
                st.bar_chart(pd.DataFrame(prediction[2]).T)
                st.write("L??gende du graphique")
                pred=pd.DataFrame({'cat??gorie':CAT_VARIANT,
                                   'code':range(0,len(CAT_VARIANT))}).set_index('cat??gorie')
                st.write(pred)
if method =="SVM":
    option = st.sidebar.selectbox('Que voulez-vous identifier ?',("S??lectionner votre cible", SVM_PATH_FAMILY, SVM_PATH_MANUFACTURER))
    
    if option =="S??lectionner votre cible":
        st.write("S??lectionner votre cible")
    
    if option != "S??lectionner votre cible": 
        #model = load_model_svm(option)
        
        predict_btn = st.button("Identifier", disabled=(uploaded_file is None))
        if predict_btn:
            #prediction = predict_image_bySVM(uploaded_file, model)
            if option == SVM_PATH_FAMILY:
                st.write("La pr??diction de cette cible n'est pas encore disponible. Vous pouvez toujours essayer avec les r??seaux de neurones!")
                #st.write(f"C'est un : {CAT_FAMILY[prediction[0]]}")

            if option == SVM_PATH_MANUFACTURER:
                st.write("La pr??diction de cette cible n'est pas encore disponible. Vous pouvez toujours essayer avec les r??seaux de neurones!")
                #st.write(f"C'est un : {CAT_MANUFACTURER[prediction[0]]}")   
             
               

