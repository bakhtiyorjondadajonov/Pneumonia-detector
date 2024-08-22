import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
  
    </style>
   
    <h2 class="centered-title">X-Ray Pneumonia Detector (XPD)🩺🫁🩻</h2>
    <p>This AI model is designed to diagnose pneumonia from chest X-ray images using deep learning techniques. 
    
    </br>
    Leveraging the power of convolutional neural networks (CNNs) and the Fastai library, the model has been trained on a large dataset of labeled chest X-rays to accurately identify signs of pneumonia.
    </br>
    </br>
    </p>
    """, 
    unsafe_allow_html=True
)
st.subheader("Why we need this model:")
st.write(
    "Pneumonia is responsible for over 15% of deaths among children under 5 globally, with 920,000 fatalities in 2015 alone. In the U.S., it led to more than 500,000 emergency visits and over 50,000 deaths that same year, ranking among the top 10 causes of death. "
    "Despite its prevalence, pneumonia is difficult to diagnose accurately, requiring expert analysis of chest X-rays (CXR) along with clinical history and tests. The diagnosis can be complicated by other conditions like pulmonary edema, hemorrhage, or lung cancer, which also appear as opacities on CXRs. "
    "Comparing CXRs over time and considering clinical symptoms can help in diagnosis."
)
st.write('This AI model streamlines pneumonia diagnosis by quickly and accurately analyzing chest X-rays, reducing human error, and providing crucial support in resource-limited settings.')
st.subheader("How to use:")
st.write("Just upload a X-Ray image, and it will determine whether the patient has pneumonia or not.")
url = "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
st.write("You can find sample data to test the model here [link](%s)" % url)
file=st.file_uploader("Upload your image",["jpg","jpeg","png","gif","svg"])
model=load_learner("./Pneumonia_analizer.pkl")
if(file):
    img=PILImage.create(file)
    prediction,prediction_id,probability=model.predict(img)
    st.image(file)
    if(prediction=="PNEUMONIA"):
        st.error(f'Prediction result: {prediction}')
    else:
        st.success(f'Prediction result: {prediction}')
    prob=f'Probability: {probability[prediction_id]*100:.1f}%'
    st.info(prob)
    fig=px.bar(x=probability*100,y=model.dls.vocab)
    st.plotly_chart(fig)