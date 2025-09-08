import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Display car image right after the title
header_image = Image.open("images\logob.png")
st.image(header_image, width=900)
def set_page_styles():
    def set_page_styles():
        st.markdown(
        """
        <style>
        .stApp {
            background-color: black; 
            color: #4f4f4f;
        }
        .stTextInput, .stSelectbox, .stSlider, .stNumberInput {
            border-radius: 5px;
            border: 1px solid #9e9e9e;
        }
        .stButton>button {
            border-radius: 20px;
            border: 1px solid #c579ff;
            color: white;
            background-color: black;
        }
        .css-10trblm {
            font-family: 'Roboto', sans-serif;
        }
        h1 {
            color: #c579ff;
        }
        .st-bb {
            border-bottom: 2px solid #c579ff;
        }
        .st-bw {
            border-width: 2px !important;
        }
        .st-br {
            border-radius: 50% !important;
        }
        .css-145kmo2 {
            padding-top: 3.5rem;
            padding-bottom: 3.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_page_styles()


# Load the dataset into a pandas dataframe
df = pd.read_csv('cleaned_sbt_japan.csv')


# input dict for categorical labels
# car_brand dict
car_brand_list = df['car_brand'].unique()
car_brand_list.sort()

car_brand_dic = {model: index for index, model in enumerate(car_brand_list)}

# car_model dict
car_model_list = df['car_model'].unique()
car_model_list.sort()

car_model_dic = {model: index for index, model in enumerate(car_model_list)}

# car_transmission dict
car_transmission_list = df['transmission'].unique()
car_transmission_list.sort()

car_transmission_dic = {model: index for index, model in enumerate(car_transmission_list)}

# car_fuel_dic
car_fuel_list = df['fuel_type'].unique()
car_fuel_list.sort()

car_fuel_dic = {model: index for index, model in enumerate(car_fuel_list)}

# car_steer_dic
car_steer_list = df['steering_type'].unique()
car_steer_list.sort()

car_steer_dic = {model: index for index, model in enumerate(car_steer_list)}

# car_drive_train_dic
car_drive_train_list = df['drive_train'].unique()
car_drive_train_list.sort()

car_drive_train_dic = {model: index for index, model in enumerate(car_drive_train_list)}

# car_seats_dic
car_seats_list = df['no_of_seats'].unique()
car_seats_list.sort()

car_seats_dic = {model: index for index, model in enumerate(car_seats_list)}

# car_doors_dic
car_doors_list = df['no_of_doors'].unique()
car_doors_list.sort()

car_doors_dic = {model: index for index, model in enumerate(car_doors_list)}

# car_body_dic
car_body_list = df['body_type'].unique()
car_body_list.sort()

car_body_dic = {model: index for index, model in enumerate(car_body_list)}



# creating a function for filtering the model name correspond to it brand.
def find_model(brand):
    model = df[df['car_brand'] == brand]['car_model'] # return series of filter model name for specific brand.
    return list(model) # return list of filter model name for specific brand.



# # loding the model
@st.cache_data()
def model_loader(path):
    model = joblib.load(path)
    return model


# # loading both models
with st.spinner('🚕🛺🚙🚜🚚🚓🚗🚕 Hold on, the app is loading !! 🚕🛺🚙🚜🚚🚓🚗🚕'):
    model_cat_grid = model_loader("xgb_model.pkl")




# # writing header
# st.title('# Used Car Price Predition™  🚗')
# st.markdown("<h2 style='text-align: center;'>Second Wind Wheels Price Prediction™</h2>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; font-family: Arial, sans-serif; color: olive; margin-bottom: 2px;'>Second Wind Wheels Price Prediction™</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

# start taking inputs
# 1. brand integer
brand_inp = col1.selectbox('Enter the brand of the car', car_brand_list, help='The Brand The Vehicle belongs to') # passing the brand list
brand = car_brand_dic[brand_inp] # mapping the brand name to its index number

# 2. year integer
year = col1.slider('Enter the year when the car was manufactured', 2015, 2024, help='According to the `Kenya Revenue Authority`, a car imported into Kenya must be less than 8 years old from the year of first registration')

col1.subheader("Technical Specifications")
# 3. taking milage info integer
mileage = col1.number_input('Enter the mileage of the car in kmpl', help='What is distance has been covered by the vehicle?')

# 4. fuel_type
fuel_type = col1.selectbox('Enter the fuel type of the car', car_fuel_list, help='The fuel type of the vehicle')
fuel_type = car_fuel_dic[fuel_type] # mapping the fuel type to its index number

# 5.engine_size_cc
engine_size_cc = col1.number_input('Enter the engine size of the car in cc', help='The engine size of the vehicle in cc')

# 6. body_type
body_type = col1.selectbox('Enter the body type of the car', car_body_list, help='The body type of the vehicle')
body_type = car_body_dic[body_type] # mapping the body type to its index number

col1.subheader("DriveTrain Configuration")
# 7. transmission
transmission = col1.selectbox('Enter the transmission type of the car', car_transmission_list, help='The transmission type of the vehicle')
transmission = car_transmission_dic[transmission] # mapping the transmission type to its index number

# 8. steering_type
steering_type = col1.selectbox('Enter the steering type of the car', car_steer_list, help='The steering type of the vehicle')
steering_type = car_steer_dic[steering_type] # mapping the steering type to its index number

# 9. drive_train
drive_train = col1.selectbox('Enter the drive train of the car', car_drive_train_list, help='The drive train of the vehicle')
drive_train = car_drive_train_dic[drive_train] # mapping the drive train to its index number

# 10. no_of_seats
no_of_seats = col1.selectbox('Enter the no of seats in the car', car_seats_list, help='The no of seats in the vehicle')
no_of_seats = car_seats_dic[no_of_seats] # mapping the no of seats to its index number

# Switch to col2
# 11. car_model
car_model_list = find_model(brand_inp) # calling the function to filter the model name for specific brand
car_model_list = list(set(car_model_list)) # remove duplicates by converting to set and back to list
car_model_list.sort() # sorting the list
car_model_prompt = f'Enter the model for the {brand_inp}' if brand_inp != 'Select' else 'Enter the model of the car'
car_model = col2.selectbox(car_model_prompt, car_model_list, help='The model of the vehicle') # passing the model list
if car_model != 'Select': # if the model is not 'Select' then map the model name to its index number
    car_model = car_model_dic[car_model] # mapping the model name to its index number

# 12. no_of_doors
no_of_doors = col2.selectbox('Enter the no of doors in the car', car_doors_list, help='The no of doors in the vehicle')
no_of_doors = car_doors_dic[no_of_doors] # mapping the no of doors to its index number

# # 13. Power Steering
# power_steering = col2.selectbox('Would you like Power Steering with this unit?', ['Yes', 'No'], help='The power steering type of the vehicle')
# power_steering = 1 if power_steering == 'Yes' else 0 # mapping the power steering type to its index number

# # 14. Air Condition
# air_condition = col2.selectbox('Would you like Air Condition with this unit?', ['Yes', 'No'], help='The air condition type of the vehicle')
# air_condition = 1 if air_condition == 'Yes' else 0 # mapping the air condition type to its index number

# # 15. Navigation
# navigation = col2.selectbox('Would you like Navigation with this unit?', ['Yes', 'No'], help='The navigation type of the vehicle')
# navigation = 1 if navigation == 'Yes' else 0 # mapping the navigation type to its index number

# # 16. Air Bag
# air_bag = col2.selectbox('Would you like Air Bag with this unit?', ['Yes', 'No'], help='The air bag type of the vehicle')
# air_bag = 1 if air_bag == 'Yes' else 0 # mapping the air bag type to its index number

# # 17. Anti-Lock Brake
# anti_lock_brake = col2.selectbox('Would you like Anti-Lock Brake with this unit?', ['Yes', 'No'], help='The anti-lock brake type of the vehicle')
# anti_lock_brake = 1 if anti_lock_brake == 'Yes' else 0 # mapping the anti-lock brake type to its index number

# # 18. Fog Lights
# fog_lights = col2.selectbox('Would you like Fog Lights with this unit?', ['Yes', 'No'], help='The fog lights type of the vehicle')
# fog_lights = 1 if fog_lights == 'Yes' else 0 # mapping the fog lights type to its index number

# # 19. Power Windows
# power_windows = col2.selectbox('Would you like Power Windows with this unit?', ['Yes', 'No'], help='The power windows type of the vehicle')
# power_windows = 1 if power_windows == 'Yes' else 0 # mapping the power windows type to its index number

# # 20. Alloy Wheels
# alloy_wheels = col2.selectbox('Would you like Alloy Wheels with this unit?', ['Yes', 'No'], help='The alloy wheels type of the vehicle')
# alloy_wheels = 1 if alloy_wheels == 'Yes' else 0 # mapping the alloy wheels type to its index number

with st.container():      
    # 13. Power Steering
    col2.subheader("Car Features")
    power_steering = col2.selectbox('Would you like Power Steering with this unit?', ['Yes', 'No'], help='The power steering type of the vehicle')
    power_steering = 1 if power_steering == 'Yes' else 0 # mapping the power steering type to its index number

    # 14. Air Condition
    air_condition = col2.selectbox('Would you like Air Condition with this unit?', ['Yes', 'No'], help='The air condition type of the vehicle')
    air_condition = 1 if air_condition == 'Yes' else 0 # mapping the air condition type to its index number

    # 15. Navigation
    navigation = col2.selectbox('Would you like Navigation with this unit?', ['Yes', 'No'], help='The navigation type of the vehicle')
    navigation = 1 if navigation == 'Yes' else 0 # mapping the navigation type to its index number

    # 16. Air Bag
    air_bag = col2.selectbox('Would you like Air Bag with this unit?', ['Yes', 'No'], help='The air bag type of the vehicle')
    air_bag = 1 if air_bag == 'Yes' else 0 # mapping the air bag type to its index number

    # 17. Anti-Lock Brake
    anti_lock_brake = col2.selectbox('Would you like Anti-Lock Brake with this unit?', ['Yes', 'No'], help='The anti-lock brake type of the vehicle')
    anti_lock_brake = 1 if anti_lock_brake == 'Yes' else 0 # mapping the anti-lock brake type to its index number

    # 18. Fog Lights
    fog_lights = col2.selectbox('Would you like Fog Lights with this unit?', ['Yes', 'No'], help='The fog lights type of the vehicle')
    fog_lights = 1 if fog_lights == 'Yes' else 0 # mapping the fog lights type to its index number

    # 19. Power Windows
    power_windows = col2.selectbox('Would you like Power Windows with this unit?', ['Yes', 'No'], help='The power windows type of the vehicle')
    power_windows = 1 if power_windows == 'Yes' else 0 # mapping the power windows type to its index number

    # 20. Alloy Wheels
    alloy_wheels = col2.selectbox('Would you like Alloy Wheels with this unit?', ['Yes', 'No'], help='The alloy wheels type of the vehicle')
    alloy_wheels = 1 if alloy_wheels == 'Yes' else 0 # mapping the alloy wheels type to its index number


# creatng a input array for prediction
input_array = np.array([brand, 
                        year, 
                        mileage, 
                        fuel_type, 
                        engine_size_cc, 
                        body_type, 
                        transmission, 
                        steering_type, 
                        drive_train, 
                        no_of_seats, 
                        car_model, 
                        no_of_doors, 
                        power_steering, 
                        air_condition, 
                        navigation, 
                        air_bag, 
                        anti_lock_brake, 
                        fog_lights, 
                        power_windows, 
                        alloy_wheels]).reshape(1, -1)

predict = col1.button('Predict')

if predict:
    with st.spinner('🚕🛺🚙🚜🚚🚓🚗🚕 Hold on, the app is loading !! 🚕🛺🚙🚜🚚🚓🚗🚕'):
        pred = model_cat_grid.predict(input_array)
        pred = pred[0]
        st.success(f'Predicted Price of the car is {pred} Kshs')






