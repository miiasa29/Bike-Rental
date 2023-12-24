#!/usr/bin/env python
# coding: utf-8
Proyek Analisis Data: Bike Rental
Nama: Siti Alamiah
Email: sitialamiah@gmail.com
ID Dicoding: sitialamiahMenentukan Pertanyaan Bisnis

1. Bagaimana tren penggunaan sepeda per jam dalam sistem berbagi sepeda Capital dari tahun 2011 hingga 2012? Apakah ada 
pola tertentu yang dapat diidentifikasi berdasarkan analisis data eksplorasi?

2 Bagaimana hubungan antara kondisi cuaca (seperti suhu, kelembapan, dan kecepatan angin) dengan penggunaan sepeda per jam? 
  Apakah cuaca memiliki pengaruh signifikan terhadap jumlah sepeda yang disewa?

3. Dapatkah kita membangun model prediksi yang dapat memperkirakan penggunaan sepeda per jam ("cnt") berdasarkan informasi 
   cuaca dan musim? Apa jenis model yang paling sesuai untuk dataset ini, dan seberapa akurat model tersebut dalam 
   memprediksi penggunaan sepeda?
# In[ ]:


# import library


# In[49]:


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


# In[ ]:


# data wrangling
# gathering data


# In[9]:


df1 = pd.read_csv('E:\INDOSAT - DCODING\Bike-sharing-dataset\hour.csv')
df2 = pd.read_csv('E:\INDOSAT - DCODING\Bike-sharing-dataset\day.csv')

df = pd.concat([df1,df2], ignore_index=True)
df


# In[10]:


# assesing data


# In[11]:


df.info()


# In[12]:


df.describe()


# In[14]:


df.shape


# In[15]:


# menghitung jumlh nilai null
df.isnull().sum()


# In[16]:


# Melihat nilai unik pada setiap kolom untuk mendapatkan pemahaman
# tentang kategori atau klasifikasi data.
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")


# In[17]:


# Menampilkan korelasi antar kolom numerik.
df.corr()


# In[18]:


# cleaning data


# In[20]:


# Menjelajahi kerangka data, mengidentifikasi Potensi Kesalahan,
# dan memahami tipe data
pd.set_option('display.max_columns', None)
def data_overview(df, head=5):
    print(" SHAPE ".center(125,'-'))
    print('Rows:{}'.format(df.shape[0]))
    print('Columns:{}'.format(df.shape[1]))
    print(" MISSING VALUES ".center(125,'-'))
    print(df.isnull().sum())
    print(" DUPLICATED VALUES ".center(125,'-'))
    print(df.duplicated().sum())
    print(" HEAD ".center(125,'-'))
    print(df.head(3))
    print(" DATA TYPES ".center(125,'-'))
    print(df.dtypes)

data_overview(df)


# In[21]:


# Memeriksa ouliers dalam varibale target "cnt"
Q1 = df['cnt'].quantile(0.25)
Q3 = df['cnt'].quantile(0.75)
IQR = Q3 - Q1


# In[22]:


# menentukan batas untuk outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[23]:


# Mengidentifikasi outlier
outliers = df[(df['cnt'] < lower_bound) | (df['cnt'] > upper_bound)]
outliers.style.background_gradient(cmap='Greys')


# In[24]:


# menghapus outlier
df = df[(df['cnt'] >= lower_bound) & (df['cnt'] <= upper_bound)]
print("shape after outliers removal :",df.shape)


# In[25]:


# mengubah variabel diskrit "musim" menjadi tempat sampah
df = pd.get_dummies(df, columns=['season'], dtype=int)
df.head()


# In[26]:


# exploratory data analysis


# In[27]:


# Histogram untuk variabel target 'cnt'
plt.figure(figsize=(10, 6))
sns.histplot(df['cnt'], bins=30, kde=True)
plt.title('Distribusi Jumlah Sepeda (cnt)')
plt.xlabel('Jumlah Sepeda')
plt.ylabel('Frekuensi')
plt.show()


# In[28]:


# Korelasi antar variabel numerik
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi antar Variabel')
plt.show()


# In[29]:


# Boxplot untuk variabel kategorikal 'jam' terhadap 'cnt'
plt.figure(figsize=(12,6))
ax = sns.boxplot(x='hr', y='cnt', data=df)
plt.title('Distribution of bike rentals per hour')
for i in ax.containers:
    ax.bar_label(i,)
plt.show()


# In[30]:


plt.figure(figsize=(12,6))
sns.boxplot(x='weekday', y='cnt', data=df)
plt.title('Distribution of bike rentals V/S days of the week')


# In[31]:


plt.figure(figsize=(12,6))
d = sns.FacetGrid(df, col="workingday")
d. map(sns.barplot, "hr", "cnt")


# In[32]:


plt.figure(figsize=(12,6))
sns.boxplot(x='mnth', y='cnt', data=df)
plt.title('Distribution of bike rentals V/S months')
plt.show()


# In[33]:


# Mengubah kolom 'dteday' menjadi tipe data datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# Line plot untuk jumlah sepeda per jam
plt.figure(figsize=(14, 6))
sns.lineplot(x='dteday', y='cnt', data=df, ci=None)
plt.title('Jumlah Sepeda per Jam')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Sepeda')
plt.show()


# In[34]:


# visualization and explanatory analysis

Pertanyaan 1 :
Bagaimana tren penggunaan sepeda per jam dalam sistem berbagi sepeda Capital dari tahun 2011 hingga 2012? Apakah ada pola tertentu yang dapat diidentifikasi berdasarkan analisis data eksplorasi?Untuk mengevaluasi tren penggunaan sepeda per jam dari tahun 2011 hingga 2012, dapat dilihat bahwa untuk membuat line plot yang menunjukkan perubahan jumlah sepeda yang disewa sepanjang waktu
# In[35]:


# Mengelompokkan data per tanggal dan menghitung jumlah sepeda
daily_counts = df.groupby(df['dteday'].dt.date)['cnt'].sum()


# In[36]:


# Line plot untuk tren penggunaan sepeda per jam dari tahun 2011 hingga 2012
plt.figure(figsize=(14, 6))
sns.lineplot(x=daily_counts.index, y=daily_counts.values, marker='o', linestyle='-', color='b')
plt.title('Tren Penggunaan Sepeda per Jam (2011-2012)')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Sepeda')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

- Mengubah Tipe Data Tanggal: Menggunakan pd.to_datetime untuk mengubah kolom 'dteday' menjadi tipe data datetime.
- Mengelompokkan Data: Menggunakan groupby untuk mengelompokkan data berdasarkan tanggal dan kemudian menghitung jumlah 
  sepeda ('cnt') yang disewa setiap harinya.

- Line Plot: Membuat line plot menggunakan sns.lineplot dengan sumbu x sebagai tanggal dan sumbu y sebagai jumlah sepeda 
  yang disewa.

- Penyempurnaan Plot: Menambahkan judul, label sumbu, dan memutar label tanggal agar lebih mudah dibaca.
# In[ ]:


Pertanyaan 2:
Bagaimana hubungan antara kondisi cuaca (seperti suhu, kelembapan, dan kecepatan angin) dengan penggunaan sepeda per jam? 
Apakah cuaca memiliki pengaruh signifikan terhadap jumlah sepeda yang disewa?

Untuk mengevaluasi hubungan antara kondisi cuaca (suhu, kelembapan, dan kecepatan angin) dengan penggunaan sepeda per jam, kita dapat menggunakan scatter plot untuk melihat sebaran data dan juga heatmap untuk melihat korelasi antar variabel numerik
# In[37]:


# Scatter plot untuk suhu dan jumlah sepeda disewa
plt.figure(figsize=(12, 6))
sns.scatterplot(x='temp', y='cnt', data=df, alpha=0.5)
plt.title('Hubungan antara Suhu dan Jumlah Sepeda Disewa')
plt.xlabel('Suhu (Celsius)')
plt.ylabel('Jumlah Sepeda')
plt.show()


# In[38]:


# Scatter plot untuk kelembapan dan jumlah sepeda disewa
plt.figure(figsize=(12, 6))
sns.scatterplot(x='hum', y='cnt', data=df, alpha=0.5)
plt.title('Hubungan antara Kelembapan dan Jumlah Sepeda Disewa')
plt.xlabel('Kelembapan')
plt.ylabel('Jumlah Sepeda')
plt.show()


# In[39]:


# Scatter plot untuk kecepatan angin dan jumlah sepeda disewa
plt.figure(figsize=(12, 6))
sns.scatterplot(x='windspeed', y='cnt', data=df, alpha=0.5)
plt.title('Hubungan antara Kecepatan Angin dan Jumlah Sepeda Disewa')
plt.xlabel('Kecepatan Angin')
plt.ylabel('Jumlah Sepeda')
plt.show()


# In[40]:


# Korelasi antar variabel numerik
correlation_matrix = df[['temp', 'hum', 'windspeed', 'cnt']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi antar Variabel Cuaca dan Jumlah Sepeda Disewa')
plt.show()

Scatter Plot: Membuat scatter plot untuk masing-masing variabel cuaca (suhu, kelembapan, dan kecepatan angin) terhadap jumlah sepeda yang disewa (cnt).

Heatmap Korelasi: Membuat heatmap untuk melihat korelasi antar variabel numerik. Ini memberikan gambaran tentang sejauh mana variabel cuaca berkorelasi dengan jumlah sepeda yang disewa.

Dengan visualisasi ini, dapat dinilai bahwa hubungan antara variabel cuaca dan jumlah sepeda yang disewa dengan Scatter plot memberikan gambaran visual sementara heatmap korelasi memberikan informasi korelasi antar variabel numerik secara lebih rinci. Jika ada korelasi yang signifikan, hal ini dapat menunjukkan bahwa kondisi cuaca memiliki pengaruh terhadap jumlah sepeda yang disewa.Pertanyaan 3:
Dapatkah kita membangun model prediksi yang dapat memperkirakan penggunaan sepeda per jam ("cnt") berdasarkan informasi cuaca dan musim? Apa jenis model yang paling sesuai untuk dataset ini, dan seberapa akurat model tersebut dalam memprediksi penggunaan sepeda?Untuk membangun model prediksi, dapat menggunakan beberapa model regresi dengan mencoba memprediksi variabel kontinu ("cnt"). Sebagai contohnya dengan menggunakan Regresi Linear atau Regresi Decision Tree. Selanjutnya,akan dilakukan pengukuran akurasi pada model tersebut
# In[48]:


# Memilih fitur-fitur yang akan digunakan untuk prediksi
features = ['temp', 'hum', 'windspeed']

# Memisahkan variabel independen (X) dan dependen (y)
X = df[features]
y = df['cnt']

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

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2): {r2}')

# Visualisasi prediksi vs. aktual
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test.index, y=y_test, label='Actual', color='blue')
sns.scatterplot(x=y_test.index, y=y_pred, label='Predicted', color='red')
plt.title('Prediksi vs. Aktual Jumlah Sepeda Disewa')
plt.xlabel('Indeks Data Uji')
plt.ylabel('Jumlah Sepeda')
plt.legend()
plt.show()

Dari kode tersebut, Anda melakukan beberapa langkah dalam membangun dan mengevaluasi model regresi linear untuk memprediksi jumlah sepeda yang disewa (cnt). Berikut adalah kesimpulan dari setiap langkah:

1. Memilih Fitur: Anda memilih fitur-fitur yang akan digunakan untuk prediksi, yaitu 'temp' (temperatur), 'hum' (kelembapan), dan 'windspeed' (kecepatan angin).

2. Memisahkan Variabel Independen dan Dependen: :Anda memisahkan variabel independen (X) dan dependen (y) dari DataFrame df menggunakan fitur yang telah dipilih.

3. Membagi Data: Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan train_test_split.

4. Membuat Model Regresi Linear:Anda membuat model regresi linear menggunakan LinearRegression dari scikit-learn.

5. Melatih Model: Model dilatih menggunakan data latih (X_train dan y_train).

6. Memprediksi Data Uji: Model digunakan untuk memprediksi nilai 'cnt' pada data uji (X_test).

7. Mengukur Akurasi Model: Mengukur akurasi model menggunakan metrik Mean Squared Error (MSE) dan R-squared (R2).

8. Menampilkan Hasil Evaluasi: Menampilkan nilai MSE dan R2 untuk mengevaluasi seberapa baik model bekerja pada data uji.

9. Visualisasi Prediksi vs. Aktual: Membuat visualisasi scatter plot untuk membandingkan nilai aktual ('Actual') dan nilai prediksi ('Predicted') pada data uji.

Dengan melihat hasil evaluasi dan visualisasi, Anda dapat menilai seberapa baik model regresi linear ini dapat memprediksi jumlah sepeda yang disewa berdasarkan fitur-fitur yang dipilih. Semakin kecil MSE dan semakin mendekati 1 nilai R2, semakin baik kinerja model. Anda juga dapat mengevaluasi visualisasi untuk melihat sejauh mana prediksi model cocok dengan nilai aktual.Conclusion: 

Pertanyaan 1:
Tren Penggunaan Sepeda (2011-2012):
Analisis tren penggunaan sepeda per jam dari tahun 2011 hingga 2012 dilakukan dengan mengubah data tanggal menjadi tipe data datetime, mengelompokkan data per tanggal, dan membangun line plot. Line plot tersebut memberikan gambaran visual tentang bagaimana penggunaan sepeda berubah sepanjang periode waktu tersebut.

Pertanyaan 2 :
Hubungan dengan Kondisi Cuaca:
Untuk mengevaluasi hubungan antara kondisi cuaca (suhu, kelembapan, dan kecepatan angin) dengan penggunaan sepeda per jam, dilakukan analisis dengan menggunakan scatter plot dan heatmap korelasi. Scatter plot menunjukkan sebaran data untuk masing-masing variabel cuaca terhadap jumlah sepeda yang disewa, sementara heatmap korelasi memberikan informasi tentang sejauh mana variabel cuaca berkorelasi dengan jumlah sepeda yang disewa.

Pertanyaan 3 :
1. Membangun Model Prediksi: Membangun model prediksi untuk memperkirakan penggunaan sepeda per jam ("cnt") berdasarkan 
   informasi cuaca dan musim. Dalam contoh tersebut, digunakan model Regresi Linear, tetapi model regresi lainnya juga 
   dapat digunakan.

2. Fitur yang Digunakan: Memilih fitur-fitur seperti suhu, kelembapan, kecepatan angin, dan musim sebagai prediktor dalam 
   model.

3. Evaluasi Akurasi Model: Mengukur akurasi model menggunakan Mean Squared Error (MSE) dan R-squared (R2). Hasil evaluasi 
   ini memberikan pemahaman tentang seberapa baik model dapat memprediksi penggunaan sepeda.

4. Visualisasi Prediksi vs. Aktual: Menyajikan hasil prediksi model dibandingkan dengan nilai aktual menggunakan scatter 
   plot. Visualisasi ini membantu dalam melihat sejauh mana model dapat mengikuti pola penggunaan sepeda yang sebenarnya.
   
Dengan langkah-langkah ini, kita dapat mendapatkan wawasan yang lebih baik tentang faktor-faktor yang memengaruhi penggunaan sepeda per jam, dan kita dapat menggunakan model prediksi untuk melakukan estimasi berdasarkan kondisi cuaca dan musim.Alasan dibagunnya model diprediksi dikarenakan bahwa Membangun model prediksi dalam konteks dataset penggunaan sepeda per jam dapat memiliki beberapa alasan yang mendasar:

1. Peramalan Permintaan: Model prediksi memungkinkan untuk meramalkan permintaan sepeda di masa depan berdasarkan kondisi 
   cuaca dan musim. Ini dapat membantu penyedia layanan berbagi sepeda untuk mengantisipasi lonjakan atau penurunan 
   permintaan, yang dapat memengaruhi persediaan dan distribusi sepeda.

2. Perencanaan Sumber Daya: Dengan memahami pola penggunaan sepeda, penyedia layanan dapat merencanakan sumber daya mereka 
   dengan lebih efisien. Ini termasuk alokasi sepeda di lokasi yang lebih strategis, pengelolaan stok, dan penjadwalan 
   pemeliharaan.

3. Optimasi Layanan: Model prediksi dapat membantu dalam mengoptimalkan layanan berbagi sepeda dengan menyesuaikan 
   penempatan dan jumlah sepeda di berbagai titik sewa berdasarkan perkiraan permintaan.

4. Peningkatan Pengalaman Pengguna: Dengan memprediksi permintaan, layanan berbagi sepeda dapat meningkatkan pengalaman 
   pengguna. Pemahaman yang lebih baik tentang kapan dan di mana sepeda diperlukan dapat meningkatkan ketersediaan sepeda 
   dan kepuasan pelanggan.

5. Efisiensi Operasional: Dengan memprediksi penggunaan sepeda, penyedia layanan dapat meningkatkan efisiensi operasional 
   mereka. Ini termasuk penjadwalan pemeliharaan, manajemen distribusi sepeda, dan pengelolaan stok.

6. Pengambilan Keputusan Berbasis Data: Model prediksi memberikan dasar untuk pengambilan keputusan berbasis data. 
   Keputusan yang didasarkan pada informasi yang akurat dan terperinci dapat mengoptimalkan kinerja dan hasil layanan 
   berbagi sepeda.

7. Pemahaman Terhadap Faktor Pengaruh: Model prediksi juga membantu dalam memahami sejauh mana faktor-faktor tertentu, 
   seperti kondisi cuaca dan musim, memengaruhi penggunaan sepeda. Ini dapat menjadi dasar untuk strategi pemasaran atau 
   penyesuaian layanan. Dengan membangun model prediksi, layanan berbagi sepeda dapat meningkatkan efisiensi operasional, 
   meningkatkan kualitas layanan, dan merespons lebih baik terhadap perubahan permintaan dan kondisi lingkungan.