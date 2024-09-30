import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open("D:/ML PROJECTS/Breast Cancer prediction/trained_model.pkl",'rb'))

# Creating a prediction function
def breast_cancer_prediction(input_data):
    # Convert the tuple to numpy array
    input_data_as_array = np.asarray(input_data)

    # Reshape the data as we're predicting for one instance
    input_data_reshaped = input_data_as_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Return result
    if prediction[0] == 1:
        return "Malignant"
    else:
        return "Benign"

# Main function for the Streamlit app
def main():
    st.title("Breast Cancer Prediction")

    # Collect inputs from the user
    radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
    texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=40.0, value=20.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=100.0, step=0.1)
    area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=700.0, step=0.1)
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.2, value=0.1, step=0.001)
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.2, step=0.001)
    concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.3, step=0.001)
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=0.5, value=0.15, step=0.001)
    symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.2, step=0.001)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=0.1, value=0.06, step=0.001)

    radius_se = st.number_input("Radius SE", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    texture_se = st.number_input("Texture SE", min_value=0.0, max_value=5.0, value=1.5, step=0.01)
    perimeter_se = st.number_input("Perimeter SE", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
    area_se = st.number_input("Area SE", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
    smoothness_se = st.number_input("Smoothness SE", min_value=0.0, max_value=0.03, value=0.007, step=0.001)
    compactness_se = st.number_input("Compactness SE", min_value=0.0, max_value=0.1, value=0.025, step=0.001)
    concavity_se = st.number_input("Concavity SE", min_value=0.0, max_value=0.5, value=0.03, step=0.001)
    concave_points_se = st.number_input("Concave Points SE", min_value=0.0, max_value=0.05, value=0.02, step=0.001)
    symmetry_se = st.number_input("Symmetry SE", min_value=0.0, max_value=0.1, value=0.02, step=0.001)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, max_value=0.01, value=0.003, step=0.001)

    radius_worst = st.number_input("Radius Worst", min_value=0.0, max_value=40.0, value=25.0, step=0.1)
    texture_worst = st.number_input("Texture Worst", min_value=0.0, max_value=50.0, value=35.0, step=0.1)
    perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, max_value=300.0, value=150.0, step=1.0)
    area_worst = st.number_input("Area Worst", min_value=0.0, max_value=4000.0, value=1500.0, step=10.0)
    smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=0.3, value=0.15, step=0.001)
    compactness_worst = st.number_input("Compactness Worst", min_value=0.0, max_value=2.0, value=0.5, step=0.001)
    concavity_worst = st.number_input("Concavity Worst", min_value=0.0, max_value=2.0, value=0.7, step=0.001)
    concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=1.0, value=0.3, step=0.001)
    symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=1.0, value=0.3, step=0.001)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=0.2, value=0.08, step=0.001)

    result = ""

    # On button click, make prediction
    if st.button("Breast Cancer Prediction"):
        result = breast_cancer_prediction((radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                                           concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                                           radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se,
                                           concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                                           radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                                           compactness_worst, concavity_worst, concave_points_worst, symmetry_worst,
                                           fractal_dimension_worst))

    st.success(result)


if __name__ == '__main__':
    main()
