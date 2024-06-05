import streamlit as st
import json
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load JSON data for categorical variables
street_name = load_json('sample1.json')
town = load_json('sample.json')
flat_type = load_json('sample2.json')
flat_model = load_json('sample3.json')
block = load_json('sample4.json')

# Set page configuration
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="🏨",
    layout="wide"
)

# Sidebar navigation
with st.sidebar:
    selected = st.selectbox("Main Menu", ["About Project", "Predictions"])

# About Project Section
if selected == "About Project":
    st.markdown("# :red[Singapore Resale Flat Prices Prediction]")
    st.image("A-Singles-Guide-to-Buying-an-HDB-Flat-1.jpeg")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")
    
# Predictions Section
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 95.8%)]")
    
    with st.form('Regression'):
        month = st.number_input("Month")
        floor_area_sqm_input = st.number_input("Floor Area (Per Square Meter)")
        lease_commence_date_input = st.number_input("Lease Commence Date")
        remaining_lease_input = st.number_input("Lease Remaining Years")
        year = st.number_input("Year")
        current_remaining_lease_input = st.number_input("Current Remaining Lease")
        street_name_input = st.selectbox("Street Name", list(street_name.keys()))
        town_input = st.selectbox("Town", list(town.keys()))
        flat_type_input = st.selectbox("Flat Type", list(flat_type.keys()))
        flat_model_input = st.selectbox("Flat Model", list(flat_model.keys()))
        block_input = st.text_input("Block Number")
        lower_bound = st.number_input("Storey lower")
        upper_bound = st.number_input("Storey Upper")
        price_per_sqm = st.number_input("Price per Square Meter")
        yearsholding = st.number_input("Years Holding")
      
        submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

        if submit_button:
            try:
                with open("linearreglasso.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)

                # Encode categorical variables
                label_encoders = {
                    'street_name': LabelEncoder().fit(list(street_name.keys())),
                    'town': LabelEncoder().fit(list(town.keys())),
                    'flat_type': LabelEncoder().fit(list(flat_type.keys())),
                    'flat_model': LabelEncoder().fit(list(flat_model.keys())),
                    'block': LabelEncoder().fit(list(block.keys()))
                }

                # Transform categorical inputs
                street_name_encoded = label_encoders['street_name'].transform([street_name_input])[0]
                town_encoded = label_encoders['town'].transform([town_input])[0]
                flat_type_encoded = label_encoders['flat_type'].transform([flat_type_input])[0]
                flat_model_encoded = label_encoders['flat_model'].transform([flat_model_input])[0]
                block_encoded = label_encoders['block'].transform([block_input])[0]

                # Create input array
                user_data = np.array([
                    month, floor_area_sqm_input, lease_commence_date_input, remaining_lease_input, year,
                    current_remaining_lease_input, street_name_encoded, town_encoded, flat_type_encoded, flat_model_encoded,
                    block_encoded, lower_bound, upper_bound, price_per_sqm, yearsholding
                ]).reshape(1, -1)

                # Make prediction
                y_pred = loaded_model.predict(user_data)
                st.write('## :green[Predicted resale price:] ', y_pred[0])
                st.balloons()
            except Exception as e:
                st.write("An error occurred:", e)   
