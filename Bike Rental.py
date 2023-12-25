import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
day_data_path = r'E:\INDOSAT - DCODING\Bike-sharing-dataset\day.csv'
hour_data_path = r'E:\INDOSAT - DCODING\Bike-sharing-dataset\hour.csv'

day_data = pd.read_csv(day_data_path)
hour_data = pd.read_csv(hour_data_path)

# Sidebar
st.sidebar.title('Bike Rental Dashboard')
selected_dataset = st.sidebar.radio('Select Dataset', ('Day', 'Hour'))

# Main content
st.title('Bike Rental Data Exploration')

if selected_dataset == 'Day':
    st.write(day_data.head())
else:
    st.write(hour_data.head())

# Exploratory Data Analysis (EDA)
st.title('Exploratory Data Analysis')

# Plot histogram untuk variabel target 'cnt'
plt.figure(figsize=(10, 6))
sns.histplot(day_data['cnt'], bins=30, kde=True)
plt.title('Distribusi Jumlah Sepeda (cnt)')
plt.xlabel('Jumlah Sepeda')
plt.ylabel('Frekuensi')
st.pyplot()

# Korelasi antar variabel numerik
correlation_matrix = day_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi antar Variabel')
st.pyplot()

# Boxplot untuk variabel kategorikal 'season' terhadap 'cnt'
plt.figure(figsize=(12,6))
ax = sns.boxplot(x='season', y='cnt', data=day_data)
plt.title('Distribution of bike rentals per season')
for i in ax.containers:
    ax.bar_label(i)
st.pyplot()

# Data Visualization
st.title('Data Visualization')

# Ubah kolom 'dteday' menjadi tipe data datetime
hour_data['dteday'] = pd.to_datetime(hour_data['dteday'])

# Insight 1: Tren Penggunaan Sepeda (2011-2012)
st.subheader('Tren Penggunaan Sepeda (2011-2012)')
daily_counts = hour_data.groupby(hour_data['dteday'].dt.date)['cnt'].sum()
plt.figure(figsize=(14, 6))
sns.lineplot(x=daily_counts.index, y=daily_counts.values, marker='o', linestyle='-', color='b')
plt.title('Tren Penggunaan Sepeda (2011-2012)')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Sepeda')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot()
st.write(
    "Analisis tren penggunaan sepeda per jam dari tahun 2011 hingga 2012 menunjukkan perubahan visual menggunakan "
    "line plot. Dengan mengelompokkan data per tanggal, kita dapat melihat bagaimana penggunaan sepeda berfluktuasi "
    "sepanjang waktu, memberikan gambaran visual yang kuat."
)

# Insight 2: Hubungan dengan Kondisi Cuaca
st.subheader('Hubungan dengan Kondisi Cuaca')
# Scatter plot untuk suhu dan jumlah sepeda disewa
plt.figure(figsize=(12, 6))
sns.scatterplot(x='temp', y='cnt', data=hour_data, alpha=0.5)
plt.title('Hubungan antara Suhu dan Jumlah Sepeda Disewa')
plt.xlabel('Suhu (Celsius)')
plt.ylabel('Jumlah Sepeda')
st.pyplot()

# Scatter plot untuk kelembapan dan jumlah sepeda disewa
plt.figure(figsize=(12, 6))
sns.scatterplot(x='hum', y='cnt', data=hour_data, alpha=0.5)
plt.title('Hubungan antara Kelembapan dan Jumlah Sepeda Disewa')
plt.xlabel('Kelembapan')
plt.ylabel('Jumlah Sepeda')
st.pyplot()

# Korelasi antar variabel numerik
correlation_matrix = hour_data[['temp', 'hum', 'windspeed', 'cnt']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi antar Variabel Cuaca dan Jumlah Sepeda Disewa')
st.pyplot()

st.write(
    "Evaluasi hubungan antara suhu, kelembapan, dan kecepatan angin dengan penggunaan sepeda per jam menggunakan scatter "
    "plot dan heatmap korelasi memperlihatkan pola hubungan antara variabel cuaca dan jumlah sepeda yang disewa."
)

# Insight 3: Membangun Model Prediksi
st.subheader('Membangun Model Prediksi')
# Memilih fitur-fitur yang akan digunakan untuk prediksi
features = ['temp', 'hum', 'windspeed']

# Memisahkan variabel independen (X) dan dependen (y)
X = hour_data[features]
y = hour_data['cnt']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Regresi Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi data uji
y_pred = model.predict(X_test)

# Mengukur akurasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualisasi prediksi vs. aktual
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test.index, y=y_test, label='Actual', color='blue')
sns.scatterplot(x=y_test.index, y=y_pred, label='Predicted', color='red')
plt.title('Prediksi vs. Aktual Jumlah Sepeda Disewa')
plt.xlabel('Indeks Data Uji')
plt.ylabel('Jumlah Sepeda')
plt.legend()
st.pyplot()

st.write(
    "Membangun model prediksi menggunakan Regresi Linear dengan fitur-fitur seperti suhu, kelembapan, kecepatan angin, "
    "dan musim. Evaluasi model dilakukan dengan mengukur Mean Squared Error (MSE) dan R-squared (R2), serta disajikan "
    "visualisasi prediksi vs. aktual menggunakan scatter plot. Dengan langkah-langkah ini, dapat memahami faktor-faktor "
    "yang memengaruhi penggunaan sepeda per jam dan melakukan estimasi berdasarkan kondisi cuaca dan musim dengan akurasi yang baik."
)

# Insight Tambahan
st.subheader('Insight Tambahan')
st.write(
    "Dalam melihat data, juga perlu diperhatikan faktor-faktor lain yang berhubungan dengan keaaan pada saat kita melakukan rental sepeda pada tahun tersebut"
)


# Footer
st.sidebar.markdown('Created with ❤️ by Siti Alamiah')


