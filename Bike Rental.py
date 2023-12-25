import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df1 = pd.read_csv('E:\INDOSAT - DCODING\Bike-sharing-dataset\hour.csv')
df2 = pd.read_csv('E:\INDOSAT - DCODING\Bike-sharing-dataset\day.csv')
df = pd.concat([df1,df2], ignore_index=True)
df
