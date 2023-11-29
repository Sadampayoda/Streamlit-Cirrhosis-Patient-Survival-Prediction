import streamlit as st


from sklearn.ensemble import RandomForestClassifier

import pickle
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import numpy as np

# import streamlit as st

# Instal matplotlib

st.title("Aplikasi Prediksi")



st.write("""
  # Cirrhosis Patient Survival Prediction
*
     """)

Age = st.number_input("Masukkan Umur anda:")

options = ["Iyaa", "Tidak"]
Ascites = st.selectbox("Apakah terkena Arcites ? :", options)

st.write("Ascites adalah suatu kondisi di mana cairan berlebihan menumpuk dalam rongga perut, antara lapisan organ dalam rongga perut (seperti hati dan usus) dan dinding perut.")

if Ascites == 'Iyaa':
    Ascites = 1
else :
    Ascites = 0

Hepatomegaly = st.selectbox("Apakah terkena Hepatomegaly ? ", options)
st.write('Hepatomegaly adalah istilah medis yang digunakan untuk menggambarkan pembesaran hati. Hati yang sehat memiliki ukuran tertentu, tetapi berbagai kondisi dapat menyebabkan hati menjadi lebih besar dari ukuran normal')
if Hepatomegaly == 'Iyaa':
    Hepatomegaly = 1
else :
    Hepatomegaly = 0

Edema = st.selectbox("Status pada edama anda ? ", 
                     ['tidak ada edema dan tanpa terapi diuretik',
                      'edema hadir tanpa diuretik, atau edema yang teratasi oleh diuretik',
                      'edema meskipun terapi diuretik'
                      ])

if Edema == 'edema meskipun terapi diuretik':
    Edema = 2
elif Edema == 'edema hadir tanpa diuretik, atau edema yang teratasi oleh diuretik':
    Edema = 1
else : 
    Edema = 0

st.write('- Edema adalah suatu kondisi medis yang ditandai oleh penumpukan cairan yang berlebihan di dalam jaringan tubuh, biasanya di ruang interstisial antara sel-sel. Ini dapat menyebabkan pembengkakan atau pembesaran area yang terkena.')
st.write('- Diuretik adalah jenis obat yang meningkatkan produksi urine dan membantu tubuh mengeluarkan kelebihan cairan dan garam.')

Bilirubin = st.number_input("Kadar serum bilirubin dalam mg/dl pada indikator kerusakan hati : ")
st.write('Bilirubin adalah suatu pigmen kuning yang dihasilkan dari pemecahan hemoglobin, protein yang mengandung zat besi yang terdapat dalam sel darah merah.')

Albumin = st.number_input('Kadar albumin dalam gm/dl, protein yang diproduksi oleh hati : ')
st.write('Albumin adalah salah satu jenis protein yang dihasilkan oleh hati dan ditemukan dalam darah manusia. Ini adalah salah satu protein plasma utama yang memiliki beberapa fungsi penting dalam tubuh')

Copper = st.number_input('Kadar tembaga dalam urine (µg/day), indikator gangguan metabolisme tembaga : ')


Alk_phos = st.number_input('Kadar fosfatase alkali dalam U/liter, indikator kerusakan hati atau masalah tulang')

st.write('Alkaline phosphatase adalah enzim yang ditemukan dalam berbagai jaringan tubuh, terutama dalam hati, tulang, usus halus, dan plasenta pada wanita hamil. Ini memiliki peran penting dalam metabolisme fosfat dan dapat diukur dalam tes darah untuk mengevaluasi kesehatan dan fungsi berbagai organ.')

Sgot = st.number_input('Kadar serum glutamat oksalat transaminase (SGOT) dalam U/ml, indikator kerusakan hati : ') 
st.write('Tes darah SGOT sering dilakukan sebagai bagian dari panel fungsi hati untuk mengevaluasi kesehatan hati dan organ-organ lain yang dapat mengandung enzim ini')

Prothrombin = st.number_input('Waktu protrombin dalam detik, indikator fungsi pembekuan darah')
st.write('Prothrombin adalah sebuah protein yang terlibat dalam proses pembekuan darah. Ini adalah salah satu faktor pembekuan darah yang penting dan berperan dalam mengubah fibrinogen menjadi fibrin, suatu langkah kunci dalam pembentukan bekuan darah')

Stage = st.selectbox('stadium histologis penyakit (1, 2, 3, atau 4) : ',[
    1,
    2,
    3,
    4
])



st.write('Tahap atau tingkat keparahan sirosis hati pada saat pengamatan awal (1, 2, 3, atau 4) semakin mendekati 4 maka semakin parah')



if st.button("Submit"):
    
    url = "cirrhosis.csv"
    df = pd.read_csv(url)
    df.interpolate(method='linear', inplace=True)
    df['Drug'	].fillna(df['Drug'].mode()[0], inplace=True)
    df['Ascites'	].fillna(df['Ascites'].mode()[0], inplace=True)
    df['Hepatomegaly'	].fillna(df['Hepatomegaly'].mode()[0], inplace=True)
    df['Spiders'	].fillna(df['Stage'].mode()[0], inplace=True)
    df['Stage'	].fillna(df['Stage'].mode()[0], inplace=True)
    df['Age'] = df['Age'] // 1000
    x = df.drop(['Status','ID'], axis=1)
    y = df["Status"]
    x['Drug'] = x['Drug'].astype('category').cat.codes
    x['Ascites'] = x['Ascites'].astype('category').cat.codes
    x['Hepatomegaly'] = x['Hepatomegaly'].astype('category').cat.codes
    x['Spiders'] = x['Spiders'].astype('category').cat.codes
    x['Stage'] = x['Stage'].astype('category').cat.codes
    x['Sex'] = x['Sex'].astype('category').cat.codes
    x['Edema'] = x['Edema'].astype('category').cat.codes

    x = df.drop(['N_Days','ID','Status','Drug','Sex','Spiders','Cholesterol','Tryglicerides','Platelets'], axis=1)
    x['Ascites'] = x['Ascites'].astype('category').cat.codes
    x['Hepatomegaly'] = x['Hepatomegaly'].astype('category').cat.codes
    x['Edema'] = x['Edema'].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.transform(X_test)
    x = pd.DataFrame(X_train,columns=x.columns)
    # Buat model Random Forest
    random_forest_model = RandomForestClassifier(n_estimators=100)
    # Latih model
    random_forest_model.fit(X_train, y_train)
    # Prediksi dengan model

    # with open('saved.pkl', 'rb') as file:
    #     loaded_model = pickle.load(file)
    random_forest_predictions = random_forest_model.predict([[Age,Ascites,Hepatomegaly,Edema,Bilirubin,Albumin,Copper,Alk_phos,Sgot,Prothrombin,Stage]])


    # load = joblib.load('saved_data.joblib');
    # prediction = load.predict([[Age,Ascites,Hepatomegaly,Edema,Bilirubin,Albumin,Copper,Alk_phos,Sgot,Prothrombin,Stage]])
    
    st.write("Hasil Prediksi:", random_forest_predictions)


    
    # new_data = [Age,Ascites,Hepatomegaly,Edema,Bilirubin,Albumin,Copper,Alk_phos,Sgot,Stage]

    # input_features = [20,0,1,0,1.8,3.64,186.0,2115.0,136.00,10.0,3.0]
    # prediction = predict(input_features,loaded_model)
    # # Proses input dan tampilkan hasilnya
    # st.write("Hasil Prediksi:", prediction)

    












# if choice == "Penjelasan dataset":
#     st.write("""
#     # CDC Diabetes Health Indicators*
#     """)
#     st.write("Kumpulan data ini dibuat dengan tujuan lebih memahami hubungan antara gaya hidup dan diabetes di Amerika Serikat dan didanai oleh Centers for Disease Control and Prevention (CDC). Setiap baris dalam dataset mewakili individu yang berpartisipasi, mencakup informasi sensitif seperti jenis kelamin, penghasilan, dan tingkat pendidikan. Pemisahan data direkomendasikan, dan prapemrosesan data melibatkan normalisasi usia. Analisis kumpulan data ini, termasuk CDC Diabetes Indicator, menjadi kunci dalam mengidentifikasi pola, tren, dan faktor risiko terkait diabetes, serta mengembangkan model prediktif. Hasil analisis ini membantu dalam merumuskan strategi efektif untuk pencegahan diabetes, memberikan dasar bagi kebijakan kesehatan masyarakat, dan meningkatkan pemahaman terhadap dampak kesehatan masyarakat di Amerika Serikat.")

#     # URL dataset Abalone di UCI Repository
#     url = "C:\kuliah\semester5\psd\Tugas-streamlit\diabetes_indikator\diabetes_012_health_indicators_BRFSS2015.csv"

#     # Mengimpor data ke dalam pandas DataFrame
#     df = pd.read_csv(url)

#     # Menampilkan DataFrame
#     df

#     st.write("""
#     ## Di Normalisasi dataset*
#     """)

#     # Pisahkan fitur (X) dan target (y)
#     X = df.drop("Diabetes_012", axis=1)
#     y = df["Diabetes_012"]

#     # Bagi dataset menjadi data pelatihan dan data pengujian
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     x = pd.DataFrame(X_train,columns=X.columns)
#     x

# elif choice == 'Akurasi berbagai metode':
#     url = "C:\kuliah\semester5\psd\Tugas-streamlit\diabetes_indikator\diabetes_012_health_indicators_BRFSS2015.csv"

#     # Mengimpor data ke dalam pandas DataFrame
#     df = pd.read_csv(url)

    

#     # Pisahkan fitur (X) dan target (y)
#     X = df.drop("Diabetes_012", axis=1)
#     y = df["Diabetes_012"]

#     # Bagi dataset menjadi data pelatihan dan data pengujian
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     st.write("""
#     ## Akurasi dari setiap metode*
#     """)

#     st.write("""
#     #### 1 Decision Trees
#     """)
#     st.write("""
#     Decision Trees (Pohon Keputusan) adalah model prediktif dalam machine learning yang digunakan untuk pengambilan keputusan berdasarkan serangkaian aturan dan kondisi. Model ini mengambil bentuk struktur pohon dengan setiap simpul (node) yang merepresentasikan keputusan atau pengujian terhadap suatu fitur, cabang (branch) yang mengarah ke simpul lainnya, dan daun (leaf) yang memberikan hasil atau prediksi.
#     """)

#     decision_tree_model = DecisionTreeClassifier()
#     # Latih model
#     decision_tree_model.fit(X_train, y_train)
#     # Prediksi dengan model
#     decision_tree_predictions = decision_tree_model.predict(X_test)
#     # Evaluasi kinerja model
#     decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
#     st.write("Akurasi decision_tree:", decision_tree_accuracy)

#     st.write("""
#     #### 2 Random Forest
#     """)
#     st.write("""
#     Random Forest adalah suatu teknik machine learning yang menggunakan konsep ansambel (ensemble learning) dengan membangun sejumlah besar Decision Trees selama proses pelatihan dan menggabungkan hasil prediksi dari semua pohon untuk menghasilkan prediksi yang lebih akurat dan stabil. Dalam Random Forest, setiap Decision Tree dibangun dengan dataset yang diambil secara acak dari dataset pelatihan, dan variasi tambahan diintroduksi melalui pengambilan sampel acak fitur pada setiap node pemisahan dalam pohon.
#     """)

#     # Buat model Random Forest
#     random_forest_model = RandomForestClassifier(n_estimators=100)
#     # Latih model
#     random_forest_model.fit(X_train, y_train)
#     # Prediksi dengan model
#     random_forest_predictions = random_forest_model.predict(X_test)
#     # Evaluasi kinerja model
#     random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)

#     st.write("Akurasi decision_tree:", decision_tree_accuracy)
    
#     st.write("""
#     #### 3 logistic regression
#     """)

#     st.write("""
#     Logistic Regression adalah suatu metode dalam statistik dan machine learning yang digunakan untuk melakukan klasifikasi biner, yaitu memprediksi kategori keluaran yang terdiri dari dua kelas atau label. Meskipun namanya mengandung kata "regression," Logistic Regression sebenarnya digunakan untuk tugas klasifikasi, bukan regresi.
#     """)

#     logistic_regression_model = LogisticRegression()

#     # Latih model
#     logistic_regression_model.fit(X_train, y_train)

#     # Prediksi dengan model
#     logistic_regression_predictions = logistic_regression_model.predict(X_test)

#     # Evaluasi kinerja model
#     logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)


#     st.write("Akurasi Regresi Logistik:", logistic_regression_accuracy)

#     st.write("""
#     #### 4 Neural_network
#     """)

#     st.write("""
#     Neural Network, atau jaringan saraf tiruan, adalah suatu model komputasi yang terinspirasi oleh struktur dan fungsi sistem saraf manusia. Ini adalah bagian dari bidang machine learning dan artificial intelligence (AI) yang bertujuan untuk mengembangkan model komputasi yang dapat belajar dari data dan melakukan tugas tanpa harus secara eksplisit diprogram.
#     """)

#     # Buat model Jaringan Saraf Tiruan
#     neural_network_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

#     # Latih model
#     neural_network_model.fit(X_train, y_train)

#     # Prediksi dengan model
#     neural_network_predictions = neural_network_model.predict(X_test)

#     # Evaluasi kinerja model
#     neural_network_accuracy = accuracy_score(y_test, neural_network_predictions)

#     st.write("Akurasi neural_network:", neural_network_accuracy)

#     st.write("""
#     #### 5 Percepton
#     """)

#     st.write("""
#     Perceptron adalah model dasar dalam dunia neural network yang mewakili unit pemrosesan sederhana yang digunakan untuk tugas klasifikasi biner. Ini adalah jenis dasar dari neuron dalam jaringan saraf tiruan. Perceptron ditemukan oleh Frank Rosenblatt pada tahun 1957 dan menjadi dasar pengembangan lebih lanjut dalam bidang machine learning.
#     """)

#     # Buat model Perceptron
#     perceptron_model = Perceptron(max_iter=1000, random_state=42)

#     # Latih model Perceptron
#     perceptron_model.fit(X_train, y_train)

#     # Prediksi dengan model Perceptron
#     perceptron_predictions = perceptron_model.predict(X_test)

#     # Evaluasi kinerja model Perceptron
#     perceptron_accuracy = accuracy_score(y_test, perceptron_predictions)

#     st.write("Akurasi Perceptron:", perceptron_accuracy)

# elif choice == 'Prediksi data':
#     st.write(""" # Prediksi data anda """)

#     options = ["Iyaa", "Tidak", ]
#     selected_option = st.selectbox("Pilih opsi HighBP:", options)