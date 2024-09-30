import numpy as np
import pickle
import streamlit as st


# Loading the saved model
loaded_model = pickle.load(open("D:/ML PROJECTS/diabates prediction/trained_model (1).pkl",'rb'))
scaler = pickle.load(open("D:/ML PROJECTS/diabates prediction/standard_scaler.pkl",'rb'))

## Creating a function :
def diabetes_prediction(input_data):
    # changing normal data to numpy data
    input_data_as_numpy_array = np.asarray(input_data)

    # input reshpae.
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    # print(std_data)

    prediction = loaded_model.predict(std_data)

    # print(prediction)

    if (prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"



def main():
    st.title("Diabetes Prediction web app")

    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('BloodPressure Value')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI Level')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of Peron')

    #Code for Prediction
    diagnosis = ''

    #Creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)

if __name__ =='__main__':
    main()