# How to Run Streamlit Dashboard
This is a brief guide to running the Streamlit dashboard. Make sure you have installed all the required libraries before running the application. You can ignore the ipynb file because I only used it for comprehensive data cleaning and performing exploratory data analysis (EDA) before creating the data dashboard.

# Install Dependencies
conda create --name main-ds python=3.9
conda activate main-ds
pip install numpy pandas scipy matplotlib seaborn jupyter streamlit babel
# dashboard.py

import streamlit as st
import numpy as np
import pandas as pd

# Your Streamlit app code goes here

if __name__ == "__main__":
    st.title("")
streamlit run dashboard.py
    # Add your Streamlit app content here
# View Dashboard Directly
In the data dashboard, you can directly observe the teren rent the bicycle on 2011-2012, starting from dteday, season, day, yr, month, hr, holiday, weekday, workingday, weathersit temp, atemp, hum, windspeed, casual, register, cnt fro the data. Many factors influence the use of the bicycle. In addition, the dashboard also provides information on how to read the data using daily and hourly data.

Here's the Link : http://localhost:8504/