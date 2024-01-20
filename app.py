#TODO : data append function -- > Done

import streamlit as st 
import pandas as pd
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit_option_menu as som

# load the model 
model_RFC = jb.load('model_RFC.joblib')


# slider 
with st.sidebar:
    # Options
    menu_option = ['Prediction', 'Add Data','Train Model']
    
    # selecte Option
    selected_option = som.option_menu('Disease Prediction System Based on Symptoms',options= menu_option , icons = ['hospital','database-fill-add','train-front'], menu_icon='bandaid')
 

# Prediction page
if selected_option == 'Prediction':
    
    # Header of Web page
    st.header(body='Disease Prediction System based on Symptoms')
    
    # devide the page into 3 col
    col1, col2, col3 = st.columns(3)


    # in 1st column
    with col1:
        Sym_1 = st.text_input('Symptom 1',0) 
        Sym_4 = st.text_input('Symptom 4',0)
        Sym_7 = st.text_input('Symptom 7',0)
        Sym_10 = st.text_input('Symptom 10',0)
        Sym_13 = st.text_input('Symptom 13',0)
        Sym_16 = st.text_input('Symptom 16',0)

    with col2:
        Sym_2 = st.text_input('Symptom 2',0) 
        Sym_5 = st.text_input('Symptom 5',0)
        Sym_8 = st.text_input('Symptom 8',0)
        Sym_11 = st.text_input('Symptom 11',0)
        Sym_14 = st.text_input('Symptom 14',0)
        Sym_17 = st.text_input('Symptom 17',0)

    with col3:
        Sym_3 = st.text_input('Symptom 3',0) 
        Sym_6 = st.text_input('Symptom 6',0)
        Sym_9 = st.text_input('Symptom 9',0)
        Sym_12 = st.text_input('Symptom 12',0)
        Sym_15 = st.text_input('Symptom 15',0)


    # code for prediction
    def prediction(Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17):

        # input data
        data = [Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17]

        # removing white space if have and handling the case error
        for i in range(len(data)):
            if data[i]!=0:
                data[i] = str(data[i]).lower().strip()

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
  
# Add Data page  
elif selected_option == 'Add Data':
    
    # Header 
    st.title('Your Contribution is Valueable!!')
    st.write(' ##### Provide data here')
    
    # devide the page into 3 col
    col1, col2, col3 = st.columns(3)

    #TODO : Add the instruction to fill the data
    
    # enter the label
    label = st.text_input('Label')
    # in 1st column
    with col1:
        Sym_1 = st.text_input('Symptom 1',0) 
        Sym_4 = st.text_input('Symptom 4',0)
        Sym_7 = st.text_input('Symptom 7',0)
        Sym_10 = st.text_input('Symptom 10',0)
        Sym_13 = st.text_input('Symptom 13',0)
        Sym_16 = st.text_input('Symptom 16',0)

    with col2:
        Sym_2 = st.text_input('Symptom 2',0) 
        Sym_5 = st.text_input('Symptom 5',0)
        Sym_8 = st.text_input('Symptom 8',0)
        Sym_11 = st.text_input('Symptom 11',0)
        Sym_14 = st.text_input('Symptom 14',0)
        Sym_17 = st.text_input('Symptom 17',0)

    with col3:
        Sym_3 = st.text_input('Symptom 3',0) 
        Sym_6 = st.text_input('Symptom 6',0)
        Sym_9 = st.text_input('Symptom 9',0)
        Sym_12 = st.text_input('Symptom 12',0)
        Sym_15 = st.text_input('Symptom 15',0)
        
    
    def add_data(label,Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17):

        #  Arrange the input in array form
        data = [label,Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17]

        # load the original data
        dataset = pd.read_csv('D:\Diseases Prediction Based on symtomps\Dataset\dataset.csv')

        # convert the input data into DataFrame
        df = pd.DataFrame([data], columns = dataset.columns)

        # Concatenate the data into Original Dataset
        dataset = pd.concat([dataset,df],ignore_index= True)
        
        # convert the dataframe into file
        dataset.to_csv('D:\Diseases Prediction Based on symtomps\Dataset\dataset.csv',mode = 'w', index= False)
        
        # dataset new data  
        df = pd.read_csv('D:\Diseases Prediction Based on symtomps\Dataset\dataset.csv')
        
    # submit the 
    if st.button("Submit"):
        ans = add_data(label,Sym_1,Sym_2,Sym_3,Sym_4,Sym_5,Sym_6,Sym_7,Sym_8,Sym_9,Sym_10,Sym_11,Sym_12,Sym_13,Sym_14,Sym_15,Sym_16,Sym_17)
        st.success('Data insertion procedure is Complete, Thank you!! 🤗')
        
# Train model   
elif selected_option == 'Train Model':
    
    # Header
    st.title('Model Training Page')
    
    # Header
    st.header("Train the model")
    
    # Instruction
    st.write("Click on the button to start traning the model")
    
    # training model
    def training_model():
        
        # dataset 
        dataset = pd.read_csv('D:\Diseases Prediction Based on symtomps\Dataset\dataset.csv')
        
        # Remove unwanted stuff
        for col in dataset.columns:
            dataset[col] = dataset[col].str.replace('_',' ')
        
        # Removing the white space in each cell of Dataframe
        cols = dataset.columns
        data = dataset[cols].values.flatten()

        s = pd.Series(data)
        s = s.str.strip()
        s = s.values.reshape(dataset.shape)

        dataset = pd.DataFrame(s,columns = cols)
        
        # handling the missing values
        dataset.fillna(0, inplace = True)
        
        # Encode the Categorical data
        ## Load the severity file
        severity = pd.read_csv('D:\Diseases Prediction Based on symtomps\Dataset\Symptom-severity.csv')
        
        ## remove the slashs from the dataset
        severity['Symptom'] = severity['Symptom'].str.replace('_',' ')
        
        ## assigned the weights
        vals= dataset.values
        symp = severity['Symptom'].unique()
        cols = dataset.columns
        
        for i in range(len(symp)):
            vals[vals == symp[i]] = severity[severity['Symptom'] == symp[i]]['weight'].values[0]
            
        ## Convert this array into DataFrame
        df = pd.DataFrame(vals,columns = cols)
        
        
        ## For non - encoded cell
        df = df.replace('spotting  urination',0)
        df = df.replace('dischromic  patches',0)
        dataset = df.replace('foul smell of urine',0)
        
        
        # Splits the DataFrame into Traning and label dataset
        X = dataset.iloc[:,1:].values
        y = dataset['Disease'].values
        
        # splits the data
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
       
        # fit the model
        model_RFC.fit(X_train, y_train)
        
        # make the prediction
        pred =  model_RFC.predict(X_test)
        
        # calculate the Accuracy
        Acc = accuracy_score(y_test,pred)
        
        # return the Accuracy
        return Acc
        
    if st.button("Start Training"):
        with st.spinner("Loading.."):
            Acc = training_model()
            
        st.success("Model Trained Successfully")
        st.success(f"The Accuracy of model is: { Acc*100 }")
        
        