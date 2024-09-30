import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/ML PROJECTS/cgpa prediction/cgpa_model.pkl','rb'))

# Creating a prediction function for CGPA
def predict_package(cgpa):
    # Reshape the CGPA for the model (since model expects 2D input)
    cgpa_reshaped = np.asarray(cgpa).reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(cgpa_reshaped)

    return prediction[0]

# Main function for the Streamlit app
def main():
    st.title("Package Prediction Based on CGPA")

    # Input field for CGPA
    cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.01)

    package = ""

    # On button click, make prediction
    if st.button("Predict Package"):
        predicted_package = predict_package(cgpa)
        package = f"CGPA of the student is: {cgpa}\nPackage offered: {predicted_package}"

    st.success(package)

if __name__ == '__main__':
    main()
