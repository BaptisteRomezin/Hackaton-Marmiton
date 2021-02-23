import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from pickle import load
from sklearn.pipeline import Pipeline

@st.cache()
def import_model():
    pipeline=Pipeline()
    pipeline.named_steps['standardize'].model=load(open('C:/Users/Baptiste/Desktop/scaler.pkl', 'rb'))
    pipeline.named_steps['mlp'].model=load_model('C:/Users/Baptiste/Desktop/kera_model.h5')

    return pipeline

pipeline=import_model()

temps_prep=int(st.sidebar.slider("Temps de préparation",min_value=0, max_value=50,step=5,value=25))
nb_ustensile=int(st.sidebar.slider("Nombre d'ustensiles",min_value=0, max_value=14,step=1,value=7))
nb_ingredients=int(st.sidebar.slider("Nombre d'ingrédients",min_value=2, max_value=25,step=2,value=15))
temps_cuisson=int(st.sidebar.slider("Temps de cuisson",min_value=0, max_value=50,step=5,value=25))
nb_etapes=int(st.sidebar.slider("Nombre d'étapes",min_value=1, max_value=65,step=5,value=25))
budget=int(st.sidebar.slider("Budget",min_value=1, max_value=3,step=1,value=2))
difficulté=int(st.sidebar.slider("Difficulté",min_value=1, max_value=4,step=1,value=2))



st.write(pipeline.predict[temps_prep,nb_ustensile,nb_ingredients,temps_cuisson,nb_etapes,budget,difficulté])


'C:/Users/Baptiste/Desktop/scaler.pkl'