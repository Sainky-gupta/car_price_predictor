import streamlit as st
import pandas as pd
import pickle
import base64

# =====================================================
# 🎯 Load Model and Data
# =====================================================
@st.cache_resource
def load_model():
    return pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_car.csv')

model = load_model()
car_data = load_data()

# =====================================================
# 🏷️ Page Configuration
# =====================================================
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# =====================================================
# 🎨 Background Image via Base64
# =====================================================
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}

        /* 🎨 Style input boxes */
        select, input {{
            background-color: rgba(0, 0, 0, 0.6);
            color: #00ffff;
            border: 2px solid #00ffff;
            border-radius: 8px;
            padding: 8px;
        }}
        select:focus, input:focus {{
            border-color: #ff00ff;
            box-shadow: 0 0 10px #ff00ff;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("car_img.jpg")  # your image file

# =====================================================
# 🖌️ Header Section
# =====================================================
st.title("🚗 Car Price Predictor")
st.write("Predict the price of a car based on its details using a trained Linear Regression model.")

# =====================================================
# 🧭 Input Section
# =====================================================
companies = sorted(car_data['company'].unique())
fuel_types = sorted(car_data['fuel_type'].unique())
years = sorted(car_data['year'].unique(), reverse=True)

st.header("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Select Car Company", companies)

    # ✅ Filter models dynamically based on selected company
    filtered_models = car_data[car_data['company'] == company]['name'].unique()
    model_name = st.selectbox("Select Car Model", sorted(filtered_models))

    fuel_type = st.selectbox("Fuel Type", fuel_types)

with col2:
    year = st.selectbox("Year of Purchase", years)
    kms = st.number_input("Kilometers Driven", min_value=0, step=1000)

# =====================================================
# 🧮 Predict Button
# =====================================================
if st.button("Predict Price"):
    input_df = pd.DataFrame([[model_name, company, year, kms, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price of the car is ₹ {round(prediction, 2):,}")

# =====================================================
# 📋 Optional Data Preview
# =====================================================
with st.expander("View sample of data used for training"):
    st.dataframe(car_data.head(10))

# =====================================================
# 📌 Footer
# =====================================================
st.markdown("---")
st.caption("Made with ❤️ by Sainky")
