import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from tensorflow.keras.models import load_model

new_title = '<h1 style="font-family:sans-serif; color:NAVY; font-size: 50px; align ="right">Potato Leaf Disease Classification</h1>'
st.markdown(new_title, unsafe_allow_html=True)

st.sidebar.subheader("SELECT A MODEL OF YOUR CHOICE")
x = st.sidebar.selectbox(label = 'MODEL',options = ["VGG16","VGG19","RESNET50","MOBILENET"])
st.write("\n\n\n\n\n")
m = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">INTRODUCTION</h2>'
st.markdown(m , unsafe_allow_html = True)
st.markdown("The two most common leaf diseases found in potatoes are early blight and late blight. A leaf that has been infected has no cure and the disease is contagious across the plant, so the only option is to remove the infected leaf.\n ")

m2 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">SAMPLE DATASET</h2>'
st.markdown(m2 , unsafe_allow_html = True)
imgsd = Image.open("sampledataset.png")
st.image(imgsd)
st.markdown("The dataset for the potato leaf diseases was obtained from the PlantVillage dataset. The datasets include over 50,000 images of 14 crops such as potatoes, grapes, tomato apples etc.")
st.markdown("From the 14 classes of data we have focused only the 3 Potato leaf classes.")

m2 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">DATA PREPROCESSING</h2>'
st.markdown(m2 , unsafe_allow_html = True)
st.markdown("To balance the data RandomOverSampler was used and to avoid overfitting of data; data augmentation using Image Data Generator was used")
imgsd = Image.open("dataprep3.png")
st.image(imgsd)

m2 = '<h2 style="font-family:sans-serif; color: BLACK; font-size: 20px; align ="right">In the project 4 different models were compared namely VGG16 , VGG19, RESNET50, MOBILENET</h2>'
st.markdown(m2 , unsafe_allow_html = True)
m2 = '<h2 style="font-family:sans-serif; color: BLACK; font-size: 20px; align ="right">The proposed model is RESNET50</h2>'
st.markdown(m2 , unsafe_allow_html = True)
if x == "VGG16":
  
    m3 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">MODEL DETAILS</h2>'
    st.markdown(m3 , unsafe_allow_html = True)
    if st.button("Click here to see model details"):
        df = pd.read_csv("classification_report_vgg16.csv")
        img1 = Image.open("vgg16.png")
        img2 = Image.open("vgg16results.png")
        st.subheader("Model architecture")
        st.image(img1)
        st.subheader("Model accuracy plots")
        st.image(img2)
        st.subheader("Classification Report")
        st.write(df)
        
if x == "VGG19":
    m3 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">MODEL DETAILS</h2>'
    st.markdown(m3 , unsafe_allow_html = True)
    if st.button("Click here to see model details"):
        df = pd.read_csv("classification_report_vgg19.csv")
        img1 = Image.open("vgg19.png")
        img2 = Image.open("vgg19results.png")
        st.subheader("Model architecture")
        st.image(img1)
        st.subheader("Model accuracy plots")
        st.image(img2)
        st.subheader("Classification Report")
        st.write(df)
        
if x == "RESNET50":
  #insert model comaprison df 
    m3 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">MODEL DETAILS</h2>'
    st.markdown(m3 , unsafe_allow_html = True)
    if st.button("Click here to see model details"):
        df = pd.read_csv("classification_report_resnet.csv")
        img1 = Image.open("resnet_model.png")
        img2 = Image.open("resnetresults.png")
        st.subheader("Model architecture")
        st.image(img1)
        st.subheader("Model accuracy plots")
        st.image(img2)
        st.subheader("Classification Report")
        st.write(df)

if x == "MOBILENET":
    m3 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">MODEL DETAILS</h2>'
    st.markdown(m3 , unsafe_allow_html = True)
    if st.button("Click here to see model details"):
        df = pd.read_csv("classification_report_mobilenet.csv")
        img1 = Image.open("mobilenet.png")
        img2 = Image.open("mobilenetresults.png")
        st.subheader("Model architecture")
        st.image(img1)
        st.subheader("Model accuracy plots")
        st.image(img2)
        st.subheader("Classification Report")
        st.write(df)
        
classes = {0 : "Healthy",
           1: "Late Blight",
           2 : "Early Blight"}

m2 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">PREDICTION:</h2>'
st.markdown(m2 , unsafe_allow_html = True)
if x == "VGG16":
    img = st.file_uploader(label = "")
    if st.button("Predict"):
        if img is not None:
            im = Image.open(img)
            im = im.resize((128,128))
            pix_val = list(im.getdata())
            pix_val_flat = [x for sets in pix_val for x in sets]
            Data = np.array(pix_val_flat).reshape(-1,128,128,3)
            model3 = load_model("vgg16.h5")
            ans = model3.predict(Data)[0]
            index = ans.argmax(axis = 0)
            st.write("The uploaded leaf images is predicted to be of the class : " + classes[index])
            st.write("CONFIDENCE =",max(ans)*100, "%")
            
if x == "VGG19":
    img = st.file_uploader(label = "")
    if st.button("Predict"):
        if img is not None:
            im = Image.open(img)
            im = im.resize((128,128))
            pix_val = list(im.getdata())
            pix_val_flat = [x for sets in pix_val for x in sets]
            Data = np.array(pix_val_flat).reshape(-1,128,128,3)
            model3 = load_model("vgg19.h5")
            ans = model3.predict(Data)[0]
            index = ans.argmax(axis = 0)
            st.write("The uploaded leaf images is predicted to be of the class" + classes[index])
            st.write("CONFIDENCE =",max(ans)*100, "%")
if x == "RESNET50":
    img = st.file_uploader(label = "")
    if st.button("Predict"):
        if img is not None:
            im = Image.open(img)
            im = im.resize((128,128))
            pix_val = list(im.getdata())
            pix_val_flat = [x for sets in pix_val for x in sets]
            Data = np.array(pix_val_flat).reshape(-1,128,128,3)
            model3 = load_model("resnet50.h5")
            ans = model3.predict(Data)[0]
            index = ans.argmax(axis = 0)
            st.write("The uploaded leaf images is predicted to be of the class : " + classes[index])
            st.write("CONFIDENCE =",max(ans)*100, "%")
if x == "MOBILENET":
    img = st.file_uploader(label = "")
    if st.button("Predict"):
        if img is not None:
            im = Image.open(img)
            im = im.resize((128,128))
            pix_val = list(im.getdata())
            pix_val_flat = [x for sets in pix_val for x in sets]
            Data = np.array(pix_val_flat).reshape(-1,128,128,3)
            model3 = load_model("mobilenet.h5")
            ans = model3.predict(Data)[0]
            index = ans.argmax(axis = 0)
            st.write("The uploaded leaf images is predicted to be of the class : " + classes[index])
            st.write("CONFIDENCE =",max(ans)*100, "%")
            
      

st.sidebar.markdown("Developed by: ")
st.sidebar.markdown("Adhitya Narayan 123003004 ")
st.sidebar.markdown("Rakshit Acharya 123003203 ")