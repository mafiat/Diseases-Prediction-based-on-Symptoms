import streamlit as st 
import pandas as pd
import numpy as np
import joblib as jb

# load the model 
model_RFC = jb.load('model_RFC.joblib')

# Header of Web page
st.header(body='Disease Prediction System based on Symptoms')

# devide the page into 3 col
col1, col2, col3 = st.columns(3)


# in 1st column
with col1:
    Sym_1 = st.text_input('Symptom 1') 
    Sym_4 = st.text_input('Symptom 4')
    Sym_7 = st.text_input('Symptom 7')
    Sym_10 = st.text_input('Symptom 10')
    Sym_13 = st.text_input('Symptom 13')
    Sym_16 = st.text_input('Symptom 16')
    
with col2:
    Sym_2 = st.text_input('Symptom 2') 
    Sym_5 = st.text_input('Symptom 5')
    Sym_8 = st.text_input('Symptom 8')
    Sym_11 = st.text_input('Symptom 11')
    Sym_14 = st.text_input('Symptom 14')
    Sym_17 = st.text_input('Symptom 17')

with col3:
    Sym_3 = st.text_input('Symptom 3') 
    Sym_6 = st.text_input('Symptom 6')
    Sym_9 = st.text_input('Symptom 9')
    Sym_12 = st.text_input('Symptom 12')
    Sym_15 = st.text_input('Symptom 15')
    

# code for prediction
def prediction(Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17):
    
    # input data
    data = [Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17]

    # symptom ranking dataset
    severity = pd.read_csv('D:\Diseases Prediction Based on symtomps\Dataset\Symptom-severity.csv')
    severity['Symptom'] = severity['Symptom'].str.replace("_"," ")
    
    # load the severity data into array
    sym = np.array(severity['Symptom'])
    weight = np.array(severity['weight'])
    
    # encode the categorical data
    for i in range(len(data)):
        for j in range(len(sym)):
            if data[i] == sym[j] :
                data[i] = weight[j]

    # make the prediction
    pred = model_RFC.predict([data])
    
    # return the prediction
    return pred[0]
    
dia_prediction = ''

# submit button
if st.button('Make Prediction'):
    dia_prediction = prediction(Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17)
    
st.success(dia_prediction)

#TODO : data append function 