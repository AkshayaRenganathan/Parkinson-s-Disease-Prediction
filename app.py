from unittest import result
import streamlit as st
import pandas as pd
import time
import numpy as np
import pandas as pd
import os, sys
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import pickle 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler((-1,1))

loaded_model= pickle.load(open('train.sav','rb'))

def prediction(value):
    df=pd.read_csv('./parkinsons.csv')
    df.head()
    features=df.loc[:,df.columns!='status'].values[:,1:]

    labels=df.loc[:,'status'].values

    print(labels[labels==1].shape[0], labels[labels==0].shape[0])


    scaler=MinMaxScaler((-1,1))

    x=scaler.fit_transform(features)

    y=labels

    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)



    model=XGBClassifier(eval_metric='mlogloss')

    model.fit(x_train,y_train)

    user_input=value
 
    in_data= np.asarray(tuple(user_input))
#reshape and scale the input array
    in_data_re = in_data.reshape(1,-1)
    in_data_sca = scaler.transform(in_data_re)
#print the predicted output for input array
    prediction=model.predict(in_data_sca) 
    print(in_data_sca)
    print(prediction)
    return prediction
    
   

st.write("<h2 style='text-align:center ; font-weight:bold'>Parkinson's Disease Prediction using Biomedical Voice Measurements</h2>",unsafe_allow_html=True)

def about():
  st.write("<h3 style:'color:blue>About Parkinson's Disease 🧠",unsafe_allow_html=True)
  st.write("Parkinson’s disease is a common chronic degenerative disorder of the central nervous system. It is a disabling disease of the ageing population and affects mobility and locomotion. It is classified as a movement disorder. The motor symptoms of Parkinson’s disease results from progressive damage of dopamine-generating cells of “substantia nigra”, of the basal ganglia, a part of the brain situated below the cerebral cortex and is called the mid-brain.")
  st.image("./parkinsons-disease.jpg" ,width=500)

def Predict():
    st.write("<br/>",unsafe_allow_html=True)
    st.write("<h3 style='text-align:center'>Lets predict if you are affected <br/> with  Parkinson's disease 🧠 <br/> OR  <br/>You are Healthy ❤️</h3>",unsafe_allow_html=True)
    st.write("<br/>",unsafe_allow_html=True)
    value= st.text_input("Enter Voice feature's extracted attributes: ")
    res=(map(float,value.rstrip().split(',')))
    result=prediction(res)
    if st.button("Predict"):
       if result == 1:
           st.error("Person is Predicted with Parkinson's Disease 😢 ")
       else:
           st.success('Person is Healthy 😃')


def XGBoost():
    st.header("XGBoost")
    st.write("<br/>",unsafe_allow_html=True)
    st.subheader("Accuracy of XGBoost Algorithm is :")
    with st.spinner('loading...'):
     time.sleep(1)
    st.success("94.87 %")
    st.write("<br/>",unsafe_allow_html=True)
    st.subheader("Mean Absolute Error of XGBoost Algorithm is :")
    with st.spinner('loading...'):
     time.sleep(1)
    st.error("5.13 %")
    st.write("<br/> ",unsafe_allow_html=True)
    st.subheader("Confusion Matrix :")
    df = pd.DataFrame({'Predicted Healthy': [5, 0],
                   'Predicted Parkinsons': [2, 32],},                 
                  index=['True Healthy', 'True Parkinsons'])
    st.table(df)
   

def SVM():
    st.header("Support Vector Machine(SVM)")
    st.write("<br/>",unsafe_allow_html=True)
    st.subheader("Accuracy of SVM Algorithm is :")
    with st.spinner('loading...'):
     time.sleep(1)
    st.success("87.18 %")
    st.write("<br/>",unsafe_allow_html=True)
    st.subheader("Mean Absolute Error of SVM Algorithm is :")
    with st.spinner('loading...'):
     time.sleep(1)
    st.error("12.82 %")
    st.write("<br/> ",unsafe_allow_html=True)
    st.subheader("Confusion Matrix :")
    df = pd.DataFrame({'Predicted Healthy': [2, 0],
                   'Predicted Parkinsons': [5, 32],},                 
                  index=['True Healthy', 'True Parkinsons'])
    st.table(df)


def KNN():
    st.header("K-nearest neighbors (KNN)")
    st.write("<br/>",unsafe_allow_html=True)
    st.subheader("Accuracy of KNN Algorithm is :")
    with st.spinner('loading...'):
     time.sleep(1)
    st.success("89.74 %")
    st.write("<br/>",unsafe_allow_html=True)
    st.subheader("Mean Absolute Error of KNN Algorithm is :")
    with st.spinner('loading...'):
     time.sleep(1)
    st.error("10.26 %")
    st.write("<br/> ",unsafe_allow_html=True)
    st.subheader("Confusion Matrix :")
    df = pd.DataFrame({'Predicted Healthy': [4, 1],
                   'Predicted Parkinsons': [3, 31],},                 
                  index=['True Healthy', 'True Parkinsons'])
    st.table(df)


def Randomforest():
    st.header("Random Forest")
    st.write("<br/>",unsafe_allow_html=True)
    st.subheader("Accuracy of Random Forest Algorithm is :")
    with st.spinner('loading...'):
     time.sleep(1)
    st.success("92.31%")
    st.write("<br/>",unsafe_allow_html=True)
    st.subheader("Mean Absolute Error of Random Forest Algorithm is :")
    with st.spinner('loading...'):
     time.sleep(1)
    st.error("7.69%")
    st.write("<br/> ",unsafe_allow_html=True)
    st.subheader("Confusion Matrix :")
    df = pd.DataFrame({'Predicted Healthy': [4, 1],
                   'Predicted Parkinsons': [3, 31],},                 
                  index=['True Healthy', 'True Parkinsons'])
    st.table(df)


def Final():
    st.header("Comparing All Algorithms..")
    st.write("<br/>",unsafe_allow_html=True)
    with st.spinner('loading...'):
     time.sleep(1)
    
    df = pd.DataFrame({'MLA used': ['XGBClassifier', 'SVC','KNeighborsClassifier','RandomForestClassifier	'],
                   'Train Accuracy': ['100.0000', '81.5068','90.4110','100.0000'],
                   'Test Accuracy': ['94.87','87.18','89.74','92.31'],
                   'Precission': ['0.925000','0.822222','0.871795','0.923077'],
                   'Recall': ['1.000000','1.000000','0.918919','0.972973'],
                   'AUC': ['0.875000','0.666667','0.751126','0.861486'],
                   },                 
                  index=['1', '2','3','4'])
    st.table(df)
    

option = st.sidebar.radio(
     'Choose from here..!',
     ('About', 'Predict', 'XGBoost','SVM','KNN','Random Forest','Final Touch'))


if option == "About":
    about()
if option == "Predict":
    Predict()
if option == "XGBoost":
    XGBoost()     
if option == "SVM":
    SVM()   
if option == "KNN":
    KNN()   
if option == "Random Forest":
    Randomforest()
if option == "Final Touch":
    Final()    
