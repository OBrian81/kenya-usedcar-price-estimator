import json
import requests
import streamlit as st
import scipy.stats as sp
import joblib
# import pygwalker as pyg
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split as train_test_split
import streamlit.components.v1 as html
from  PIL import Image
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
from textblob import TextBlob

# Import EDA Packages
import pandas as pd
import numpy as np

# Import Visualisation Packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
from matplotlib import style

# Hide Warnings
import warnings
warnings.filterwarnings('ignore')

# Enable displaying all columns and preventing truncation
pd.set_option('display.max.columns', None)
pd.set_option('display.max_colwidth', None)

st.set_page_config(
    layout="wide",
    page_title="Second Wond Wheels - Used Car Price Prediction",
    page_icon="./assets/images/car_icon.jpg",
    initial_sidebar_state="expanded",
)

# Hide the Streamlit Footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#st.set_option('deprecation.showPyplotGlobalUse', False)

# Define the wrangle function to read an excel file and return a dataframe
@st.cache_data
def wrangle(filepath):
    # Read the file into a dataframe
    df = pd.read_csv(filepath)
    
    return df


# Load the data
df = wrangle("clean_dataset.csv")

# Load a modfied version of the dataframe
# @st.cache_data
# def modify_df():
#     global df
#     df = df.drop(columns = ['id', 'conversation_id'], axis = 1)
#     return df

# Load the Lottie File
@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath) as f:
        return json.load(f)

# Load lottie files
from streamlit_lottie import st_lottie # pip install streamlit-lottie
# # Define the paths of the lottie files
# path = "./assets/lottie files/car3.json"

# # Load the Lottie File
# lottie_twitter = load_lottiefile(path)

# # Display the Lottie File in the sidebar
# with st.sidebar:
#     st_lottie(lottie_twitter, quality='high', speed=1, height=250, key="initial")

# logo_image = Image.open("assets\images\logo2.jpg")
# with st.sidebar:
#     st.image(logo_image, width=300)



# Start of the sidebar option Menu

with st.sidebar:choose = option_menu("Second Wind Wheels", ["Background", "Data Collection & Preparation", "Data Analysis", "Data Modelling"],
                         icons=['house', 'bi bi-journal-bookmark', 'bar-chart-line', 'bi bi-gear' ,'file-earmark-text'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "lightblue"},
        "icon": {"color": "teal", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "Grey"},
        "nav-link-selected": {"background": "linear-gradient(144deg,#808080, #A9A9A9 50%,#D3D3D3)", "border-radius": "5px"},
    }
    )


# Background Page
if choose == "Background":

    # create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        header_image = Image.open("images\logob.png")
        st.image(header_image, width=900)

    # with col2:
    #     st.write("")
    #     st.write("")
    #     st.write("")
    #     st.write("")
    #     st.markdown("<hr style='border:2px solid orange'>", unsafe_allow_html=True)
    #     # Link to the Roboto font
    #     st.markdown('<link href="https://fonts.googleapis.com/css2?family=Arial&display=swap" rel="stylesheet">', unsafe_allow_html=True)

        # st.markdown("<hr style='border:2px solid orange'>", unsafe_allow_html=True)
        # st.markdown("<hr style='border:2px solid white'>", unsafe_allow_html=True)
        # st.markdown("<hr style='border:2px solid orange'>", unsafe_allow_html=True)


    # Executive Summary
    
    st.markdown("""
    <h2 style='font-family: Times New Roman; color: olive;'> Executive Summary:
    </p>
    """
    , unsafe_allow_html=True)

    st.markdown("""
    <h5 style='font-family: Times New Roman'>As Data Scientists at `Second Wind Wheels`, a leading online used car retailer, we propose a project to develop a machine learning model for predicting the accurate market value of used cars. This model will revolutionize our pricing strategy, boosting profits and enhancing customer satisfaction.
    </p>
    """
    , unsafe_allow_html=True)

    # Problem Statement
    
    # Use the Roboto font
    st.markdown("""
    <h2 style='font-family: Times New Roman; color: olive;'> Problem Statement:
    </p>
    """
    , unsafe_allow_html=True) 

    st.markdown("""
    <h5 style='font-family: Times New Roman'>Currently, used car pricing relies heavily on manual valuations by human appraisers. This method is prone to subjectivity, inconsistencies, and delays, leading to:
    </p>
    """
    , unsafe_allow_html=True)

    # st.markdown('<link href="https://fonts.googleapis.com/css2?family=Arial&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown("""
    <h5 style='font-family: Times New Roman'>

    - **Overpricing: Cars priced too high stagnate on the lot, incurring storage costs and lost sales.**

    - **Underpricing: We miss out on maximizing profits by selling cars for less than their true market value.**

    - **Customer dissatisfaction: Unfairly priced cars may discourage buyers and damage brand reputation.**
    </p>
    """
    , unsafe_allow_html=True)

    # Proposed Solution
    
    st.markdown("""
    <h2 style='font-family: Jua; color: olive;' Proposed Solution:
    </p>
    """
    , unsafe_allow_html=True) 

    st.markdown("""
    <h5 style='font-family: Times New Roman'>To address this business challenge, we will leverage Second Wind Wheels existing used car sales data along with additional data scraped from various automotive websites to develop a machine learning model that can accurately predict used car prices.
    </p>
    """
    , unsafe_allow_html=True)

    # Data Pertinence & Attribution
    
    st.markdown("""
    <h2 style='font-family: Jua; color: olive;'> Data Pertinence & Attribution:
    </p>
    """
    , unsafe_allow_html=True) 

    
    st.markdown("""
    <h5 style='font-family: Times New Roman'>The model will be developed using up-to-date used car listing data scraped from major automotive sales sites to incorporate broader market pricing trends. It includes details on used car sales across the Japan.
    </p>
    """
    , unsafe_allow_html=True)


    # Objectives
    
    st.markdown("""
    <h2 style='font-family: Jua; color: olive;'> Objectives:
    </p>
    """
    , unsafe_allow_html=True) 

    
    st.markdown("""
    <h5 style='font-family: Times New Roman'>Defining clear goals and success metrics early on helps guide the analysis in an effective direction.
    </p>
    """
    , unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # First container in the first column
    with col1.container(border=True):
            st.subheader("Main Objectives")
            with st.container(border=True):
                    st.markdown("""
                        <h5 style='font-family: Times New Roman'>

                        - **Collect and preprocess a comprehensive dataset of used cars, including various attributes such as make, model, year, mileage, and historical prices.**

                        - **Explore and apply various data mining and machine learning techniques, such as regression models, to identify the most accurate price prediction model.**

                        - **Evaluate the performance of the developed model and ensure it meets the predefined accuracy and reliability standards.**
                        </p>
                        """
                        , unsafe_allow_html=True)



    # Second container in the second column
    with col2.container(border=True):
        st.subheader("Specific Objectives")
        with st.container(border=True):
            st.markdown("""
                <h5 style='font-family: Times New Roman'>

                - **Collect and preprocess a comprehensive dataset of used cars, including various attributes such as make, model, year, mileage, and historical prices.**

                - **Explore and apply various data mining and machine learning techniques, such as regression models, to identify the most accurate price prediction model.**

                - **Evaluate the performance of the developed model and ensure it meets the predefined accuracy and reliability standards.**
                </p>
                """
                , unsafe_allow_html=True)


# Data Preparation Page
elif choose == "Data Collection & Preparation":
    
    html_temp = """
    <h1 style="background-color: teal; color: white; text-align: center; text-shadow: 2px 2px 4px #000000; font-weight: bold; padding: 10px;">Prepare the Data</h1>
    </div>


    """
    
    st.markdown(html_temp, unsafe_allow_html=True)

    st.markdown("""
    <h5 style='font-family: Times New Roman'>This involves transforming raw data into a clean and structured format that can be easily analyzed and modeled. The goal of data preparation is to ensure that the data is accurate, complete, consistent, and relevant.
    Overall, data preparation is a critical step in a data science project, as the quality and accuracy of the data will directly impact the accuracy and effectiveness of the machine learning models.
    </p>
    """
    , unsafe_allow_html=True)


    # Create two columns
    col1, col2 = st.columns([1, 1.2], gap="large")

    # Use the first column for the Data Dictionary
    with col1:
        st.markdown("""
        <h2 style="font-family: Jua;color:olive;text-align:center;">Data Dictionary</h2>
        """
        , unsafe_allow_html=True)

        st.markdown("""
        <h5 style='font-family: Times New Roman'>The data dictionary describes the variables (columns) in the dataset and their corresponding descriptions.
        </p>
        """
        , unsafe_allow_html=True)

        with st.container(border=True, height=400):
            st.markdown("""
                <style>
                body {
                    font-family: 'Times New Roman', sans-serif;
                }
                </style>
                | Column Name | Description |
                | --- | --- |
                | inventory_location | The location where the used car is available for sale or purchase. |
                | mileage_km | The distance in kilometers that the used car has been driven. |
                | engine_size_cc | The engine capacity in cubic centimeters of the used car. |
                | transmission | The type of transmission system used in the used car (e.g., manual, automatic, CVT, etc.). |
                | fuel_type | The type of fuel used by the used car (e.g., petrol, diesel, electric, etc.). |
                | steering_type | The type of steering system used in the used car (e.g., hydraulic, electronic, etc.). |
                | drive_train | The type of drive train system used in the used car (e.g., front-wheel drive, rear-wheel drive, all-wheel drive, etc.). |
                | no_of_seats | The number of seats available in the used car. |
                | no_of_doors | The number of doors in the used car. |
                | body_type | The type of body style used in the used car (e.g., sedan, hatchback, SUV, etc.). |
                | Power Steering | Indicator for whether the used car has power steering or not. |
                | Air Conditioner | Indicator for whether the used car has air conditioning or not. |
                | Navigation | Indicator for whether the used car has a navigation system or not. |
                | Air Bag | Indicator for whether the used car has air bags or not. |
                | Anti-Lock Brake System | Indicator for whether the used car has an anti-lock brake system or not. |
                | Fog Lights | Indicator for whether the used car has fog lights or not. |
                | Power Windows | Indicator for whether the used car has power windows or not. |
                | Alloy Wheels | Indicator for whether the used car has alloy wheels or not. |
                | year | The year of production of the used car. |
                | month | The month of production of the used car. |
                | car_brand | The brand of the used car. |
                | car_model | The model of the used car. |
                | price(Ksh) | The selling price of the used car in Kenyan Shillings. |
                """, unsafe_allow_html=True)

    # Use the second column for the Dataset Preview
    with col2:
        st.markdown("""
        <h2 style="font-family: Jua;color:olive;text-align:center;">Preview The Dataset</h2>
        """
        , unsafe_allow_html=True)

        st.markdown("""
        <h5 style='font-family: Times New Roman'>The dataset contains 28,000 rows and 21 columns. The columns are a mix of categorical and numerical data types.
        </p>
        """
        , unsafe_allow_html=True)

        st.write(df)

    st.markdown("""
        <h2 style="font-family: Jua;color:olive;">Data Preparation Steps:</h2>
        """
        , unsafe_allow_html=True)

    st.markdown("""
                <h5 style='font-family:Times New Roman '>

    -  **Dealing with rare categories. If they are too many to list we either group them into a single category or drop the categories.**
                
    - **Dealing with outliers greater than a certain upper bound or a certain lower bound. We either capped them or removed them.**
                
    - **Dropping redundant columns.**
                
    - **Feature Engineering new coulns form the already existing ones.**
                
    - **Dealing with missing values.**
                
    - **Dealing with duplicates and ensuring that they are indeed duplicate values.**
                
    - **Dealing with typos and mislabeled classes.**
                
    - **Dealing with inconsistent capitalization.**
                
    - **Dealing with inconsistent units of measure.**
                
    - **Dealing with inconsistent date and time formats.**
                
    - **Dealing with inconsistent currency formats.**

    
    </p>
    """
    , unsafe_allow_html=True)



# Data Analysis Page
elif choose == "Data Analysis":
    html_temp = """
    <h1 style="background-color: teal; color: white; text-align: center; text-shadow: 2px 2px 4px #000000; font-weight: bold; padding: 10px;">Explaratory Data Analysis</h1>
    </div>

    """
    st.markdown(html_temp, unsafe_allow_html=True)


    st.markdown("""
        <h2 style="font-family: Jua;color:olive;">Data Analysis Steps:</h2>
        """
        , unsafe_allow_html=True)

    st.markdown("""
        <h5 style='font-family:  Times New Roman'>This involves understanding the data, getting a sense of its structure, checking for anomalies and outliers, testing a hypothesis, or checking assumptions with the help of summary statistics and graphical representations. It's a crucial step before further data processing and modeling as it provides context and important insights about the underlying structure of the data.
        </p>
        """
        , unsafe_allow_html=True)

    st.markdown("""
    ---
    """)


    st.markdown("""
                <h5 style='font-family:  Times New Roman'>

    -  **Summarizing the data: This includes generating descriptive statistics such as mean, median, mode, standard deviation, and variance for numerical data, and frequency tables and bar charts for categorical data.**
                
    - **Visualizing the data: This involves creating visual representations of the data such as scatterplots, boxplots, histograms, heatmaps, or time series plots, to identify patterns and relationships between variables.**
                
    - **Identifying patterns and relationships: This involves exploring potential relationships and correlations between variables, and identifying any patterns, trends, or seasonality in the data.**
                
    - **Detecting outliers: This involves identifying any values that are significantly different from the rest of the data, and investigating whether they represent valid or erroneous data.**
                
    - **Hypothesis testing: This involves testing hypotheses about potential relationships or differences in the data using statistical methods such as t-tests or ANOVA.**
                
    - **Exploring data subsets: This involves exploring subsets of the data, such as groups or clusters, to identify any differences or similarities within the data.**

    </p>
    """
    , unsafe_allow_html=True)

    st.markdown("""
        <h2 style="font-family: Jua;color:olive;">Summary Statistics:</h2>
        """
        , unsafe_allow_html=True)

    st.markdown("""
        <h5 style='font-family: Times New Roman'>The summary statistics provide a quick overview of the data, including the number of rows and columns, the data types of the columns, and the descriptive statistics for the numerical columns.
        </p>
        """
        , unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # Use the first column
    with col1:
        with st.container(border = True):
            st.subheader("Statistical Summary Table")
            st.write(df.describe(include=[np.number]))

    # Use the second column
    with col2:
        st.markdown("<hr style='border:2px solid olive'>", unsafe_allow_html=True)
        st.markdown("""
                    <h5 style='font-family: Times New Roman'>

        -  **The average mileage of the cars is 39,792 km, with a standard deviation of 35,513 km. This means that the mileage values are quite spread out, ranging from 1,000 km to over 100,000 km.**
                    
        - **The average engine size of the cars is 2,361 cc, with a standard deviation of 864 cc. This means that the engine sizes are also quite varied, ranging from 900 cc to over 4,000 cc.**
                    
        - **Most of the cars have power steering (89.5%), air conditioner (97.2%), air bag (97.3%), power windows (89.7%), and alloy wheels (87.2%). Only a few cars have navigation (3.7%), anti-lock brake system (4.6%), and fog lights (4.9%).**
                    
        - **The average price of the cars is 5,630,688 Ksh, with a standard deviation of 2,710,810 Ksh. This means that the prices are also quite dispersed, ranging from a few hundred thousand to over 10 million Ksh.**
        
        </p>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='border:2px solid olive'>", unsafe_allow_html=True)


    st.markdown("""
        <h2 style="font-family: Jua;color:olive;">Data Visualisation:</h2>
        """
        , unsafe_allow_html=True)

    st.markdown("""
        <h5 style='font-family: Times New Roman'>The data visualizations provide a graphical representation of the data, including the distribution of the numerical columns, the relationship between the numerical columns, and the distribution of the categorical columns.
        </p>
        """
        , unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # Use the first column
    with col1:
        

        with st.container(border=True):
            # Create a selectbox for the user to select an option
            selection = st.selectbox('Choose an a Data Analysis Question:', ('What is the top 3 manufacturer with most cars?',
             'Top 5 Toyota models',
            'Top 5 mercedes models',
            'Top 5 BMW models'))

            # Change the subheader based on the selection
            if selection == 'What is the top 3 manufacturer with most cars?':
                st.subheader("Top 3 Manufucters with Most Cars")
                top_manufacturers = df['car_brand'].value_counts().nlargest(10)
                fig = px.bar(x=top_manufacturers.index, y=top_manufacturers.values, labels={'x':'Manufacturer', 'y':'Number of Cars'}, color=top_manufacturers.values, color_continuous_scale='Viridis')
                fig.update_layout(title_text='Top 3 Manufacturers with Most Cars')
                st.plotly_chart(fig)
            elif selection == 'Top 5 Toyota models':
                st.subheader("Top 5 Toyota models")

                # Extracting top 5 Toyota models
                top_toyota_models = df[df['car_brand'] == 'TOYOTA']['car_model'].value_counts().nlargest(5)

                fig = px.bar(x=top_toyota_models.index, y=top_toyota_models.values, labels={'x':'Model', 'y':'Number of Cars'}, color=top_toyota_models.values, color_continuous_scale='Viridis')
                fig.update_layout(title_text='Top 5 Toyota Models')
                st.plotly_chart(fig)
            
            elif selection == 'Top 5 mercedes models':
                st.subheader("Top 5 mercedes models")
                #Extracting top 5 merceded model
                top_mercedes_models = df[df['car_brand'] == 'MERCEDES']['car_model'].value_counts().nlargest(5)
                fig = px.bar(x=top_mercedes_models.index, y=top_mercedes_models.values, labels={'x':'Model', 'y':'Number of Cars'}, color=top_mercedes_models.values, color_continuous_scale='Viridis')
                fig.update_layout(title_text='Top 5 Mercedes Models')
                st.plotly_chart(fig)

            else:
                st.subheader("Top 5 BMW models")
                # Extract the top 5 BMW car models from the DataFrame based on frequency
                top_bmw_models = df[df['car_brand'] == 'BMW']['car_model'].value_counts().nlargest(5)
                fig = px.bar(x=top_bmw_models.index, y=top_bmw_models.values, labels={'x':'Model', 'y':'Number of Cars'}, color=top_bmw_models.values, color_continuous_scale='Viridis')
                fig.update_layout(title_text='Top 5 BMW Models')
                st.plotly_chart(fig)
                

            # # Rest of your code
            # top_manufacturers = df['car_brand'].value_counts().nlargest(10)
            # fig = px.bar(x=top_manufacturers.index, y=top_manufacturers.values, labels={'x':'Manufacturer', 'y':'Number of Cars'}, color=top_manufacturers.values, color_continuous_scale='Viridis')
            # fig.update_layout(title_text='Top 3 Manufacturers with Most Cars')
            # st.plotly_chart(fig)

    
    # pyg.walk(df)



# Data Modelling Page
elif choose == "Data Modelling":
    html_temp = """
    <h1 style="background-color: teal; color: white; text-align: center; text-shadow: 2px 2px 4px #000000; font-weight: bold; padding: 10px;">Modelling</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.markdown("""
        <h2 style="font-family: Jua;color:olive;">Modelling:</h2>
        """
        , unsafe_allow_html=True)

    st.markdown("""
        <h5 style='font-family: Times New Roman'>This involved the selection, training, and tuning of machine learning models. Our methodology involved leveraging both Random Forest Regressor and XGBoost Regressor, 
                accompanied by grid searches to optimize on their respective parameters, thereby enhancing their performance.
                 
    
        <h5 style='font-family: Times New Roman'>  Initially, the Random Forest Regressor Model yielded an r2 score of 0.8514. However, after hyperparameter tuning
                the score improved to  0.8535, indicating a slight improvement in predictive accuracy.

        <h5 style='font-family: Times New Roman'>Due to the improved scores we proceeded to evaluate our XGBoost model using the refined parameters obtained from the hyperparameter tuning process.
                Impressively, the XGBoost model demonstrated a superior performance with an r2 score of 0.8616.

        <h5 style='font-family: Times New Roman'>Due to the improved scores we designated the XGBOOST model as our final model for subsequent analyses and applications.

        
        </p>
        """
        , unsafe_allow_html=True)

    st.markdown("""
    ---
    """)
