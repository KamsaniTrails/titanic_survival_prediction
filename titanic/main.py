import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open("titanic/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define a function to predict survival
def predict_survival(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit frontend
st.title('Titanic Survival Prediction')

st.write("Enter passenger details to predict survival:")

# Input fields for user input
passenger_id = st.number_input("Passenger ID", min_value=1, max_value=100000, value=1)
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0, max_value=500, value=50)
sex = st.selectbox("Sex", ['male', 'female'])
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=1)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
pclass = st.selectbox("Pclass", [1, 2, 3])
embarked = st.selectbox("Embarked (C, Q, S)", ['C', 'Q', 'S'])
familysize = sibsp + parch + 1  # Creating family size by adding siblings, parents, and the passenger

# Prepare the features for prediction
sex_map = {'male': 0, 'female': 1}
embarked_map = {'C': 0, 'Q': 1, 'S': 2}

features = [
    passenger_id,
    age,
    fare,
    sex_map[sex],  # Encoding 'male' as 0 and 'female' as 1
    sibsp,
    parch,
    pclass,
    embarked_map[embarked],  # Encoding Embarked C, Q, S as 0, 1, 2
    familysize
]

# Make the prediction when the user clicks the button
if st.button('Predict Survival'):
    prediction = predict_survival(features)
    
    if prediction == 1:
        st.success("The passenger is likely to survive!")
    else:
        st.error("The passenger is likely not to survive.")
