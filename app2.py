import streamlit as st
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import scipy
import numpy as np
 
st.write("""
# Upload file audio*
""")


uploaded_audio = st.file_uploader("Unggah file audio (MP3, WAV, dll.)", type=["mp3", "wav", "ogg"])

# Direktori penyimpanan
upload_dir = "uploads"

# Membuat direktori 'uploads' jika belum ada
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Jika file telah diunggah
if uploaded_audio is not None:
    with open(os.path.join(upload_dir, uploaded_audio.name), "wb") as f:
        f.write(uploaded_audio.read())
    st.success(f"File '{uploaded_audio.name}' berhasil diunggah dan disimpan di direktori 'uploads'")


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

data = {
    'Std' : [],
    'Mean': [],
    'Median': [],
    'Min': [],
    'Max': [],
    'Skewness': [],
    'Kurtosis': [],
    'Mode': [],
    'Q1': [],
    'Q3': [],
    'IQR': [],
    'ZCR Mean': [],
    'ZCR Median': [],
    'ZCR Std': [],
    'ZCR Kurtosis': [],
    'ZCR Skewness': [],
    'RMSE Mean': [],
    'RMSE Median': [],
    'RMSE Std': [],
    'RMSE Kurtosis': [],
    'RMSE Skewness': [],
    
}


upload_dir = "uploads"
audio_files = os.listdir(upload_dir)

# Komponen untuk memilih file audio
selected_audio = st.selectbox("Pilih file audio yang ingin diproses:", audio_files)

# Jika file audio dipilih
if selected_audio:
    # Baca file audio yang dipilih
    selected_audio_path = os.path.join(upload_dir, selected_audio)
    st.audio(selected_audio_path)
    x, sr = librosa.load(selected_audio_path)

    # Tampilkan grafik gelombang audio
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(x, sr=sr)
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Amplitudo")
    plt.title("Grafik Gelombang Audio")
    st.pyplot(plt)

    zcr = librosa.feature.zero_crossing_rate(x)

    # Menghitung statistik dari file audio
    mean = np.mean(x)
    std = np.std(x)
    median = np.median(x)
    min_value = np.min(x)
    max_value = np.max(x)
    skewness = scipy.stats.skew(x)
    kurtosis = scipy.stats.kurtosis(x)
    mode = scipy.stats.mode(x)[0]
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    iqr = scipy.stats.iqr(x)

    zcr_mean = zcr.mean()
    zcr_median = np.median(zcr)
    zcr_std = np.std(zcr)
    zcr_kurtosis = scipy.stats.kurtosis(zcr.ravel())
    zcr_skewness = scipy.stats.skew(zcr.ravel())

    # x_normalized = (x - np.mean(x)) / np.std(x)
    # rmse = np.sqrt(np.mean(np.square(x_normalized)))
    rmse = np.sqrt(np.mean(np.square(x)))


    rms_mean = rmse.mean()
    rms_median = np.median(rmse)
    rms_std = np.std(rmse)
    rms_kurtosis = scipy.stats.kurtosis(rmse, nan_policy='omit')
    rms_skewness = scipy.stats.skew(rmse, nan_policy='omit')





    # Menambahkan hasil statistik dan label ke dalam list
    # data['File Name'].append(audio_file)
    data['Mean'].append(mean)
    data['Std'].append(std)
    data['Median'].append(median)
    data['Min'].append(min_value)
    data['Max'].append(max_value)
    data['Skewness'].append(skewness)
    data['Kurtosis'].append(kurtosis)
    data['Mode'].append(mode)
    data['Q1'].append(q1)
    data['Q3'].append(q3)
    data['IQR'].append(iqr)
    data['ZCR Mean'].append(zcr_mean)
    data['ZCR Median'].append(zcr_median)
    data['ZCR Std'].append(zcr_std)
    data['ZCR Kurtosis'].append(zcr_kurtosis)
    data['ZCR Skewness'].append(zcr_skewness)
    data['RMSE Mean'].append(rms_mean)
    data['RMSE Median'].append(rms_median)
    data['RMSE Std'].append(rms_std)
    data['RMSE Kurtosis'].append(rms_kurtosis)
    data['RMSE Skewness'].append(rms_skewness)
    df = pd.DataFrame(data)
    st.dataframe(df)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("""
    # Data audio*
    """)
    file_path = 'result.csv'
    audio_data = pd.read_csv(file_path)

    dataAudio = pd.DataFrame(audio_data)
    st.dataframe(audio_data)


    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    

    # Mengambil fitur-fitur dari DataFrame (kecuali 'File Name' dan 'Label')
    features = dataAudio.drop(columns=['File Name', 'Label','RMSE Kurtosis',	'RMSE Skewness'	 ])

    # Menggunakan StandardScaler untuk mentransformasi fitur-fitur
    scaler_audio = StandardScaler()
    scaled_features = scaler_audio.fit_transform(features)
    # dump(scaler, open('wb/scaler.pkl', 'wb'))

    # Split data menjadi data train dan data test
    # Split data menjadi X dan y
    X = scaled_features
    y = dataAudio['Label']
    # X_train dan y_train adalah data pelatihan dan labelnya
    # X_test adalah data uji yang ingin di normalisasi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)



    
    from sklearn.preprocessing import MinMaxScaler
    


    # Menggunakan StandardScaler untuk mentransformasi fitur-fitur
    scaler_audio = StandardScaler()
    scaled_features = scaler_audio.fit_transform(features)

    # Normalisasi data pelatihan
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # dummy_data = data
    dummy_data = df.drop(columns=['RMSE Kurtosis',	'RMSE Skewness'	 ])

    dummy_data_df = pd.DataFrame(dummy_data)
    # Normalisasi data dummy dengan metode Min-Max
    dummy_data_scaled = scaler_audio.transform(dummy_data_df)

    # Menggunakan StandardScaler yang sama untuk normalisasi dummy_data
    dummy_data_scaled = scaler.transform(dummy_data_df)

    # Menggunakan MinMaxScaler untuk dataset make_blobs
    scaler_blobs = MinMaxScaler()

    # Fitscaler train dataset make_blobs
    scaler_blobs.fit(X_train)

    # Transformasi dataset make_blobs
    X_train_blobs_scaled = scaler_blobs.transform(X_train)
    X_test_blobs_scaled = scaler_blobs.transform(X_test)

    # Mengecek rentang nilai setiap fitur setelah penskalaan
    # for i in range(X_test_blobs_scaled.shape[1]):
    #     st.text('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
    #         (i, X_train_blobs_scaled[:, i].min(), X_train_blobs_scaled[:, i].max(),
    #         X_test_blobs_scaled[:, i].min(), X_test_blobs_scaled[:, i].max()))
    # st.text("Dimensi X_train:", X_train.shape)




    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.impute import SimpleImputer
    import joblib
    from sklearn.datasets import make_blobs
    import numpy as np
    

    # 1. Split fitur (X) dan labels (y) menjadi 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # X_synthetic, y_synthetic = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
    # ukuranLatih = f"Ukuran Data Latih (X_train, y_train): {X_train_shape}, {y_train_shape}"
    # ukuranData = f"Ukuran Data Latih (X_train, y_train): {X_test.shape}, {y_test.shape}"
    # st.text(ukuranLatih)
    # st.text(ukuranData)

    # 2. Normalisasi dengan metode Min-Max
    minmax_scaler = MinMaxScaler()
    X_train_minmax_scaled = minmax_scaler.fit_transform(X_train)
    X_test_minmax_scaled = minmax_scaler.transform(X_test)

    # 3. Penanganan NaN
    imputer = SimpleImputer(strategy='mean')
    X_train_minmax_scaled = imputer.fit_transform(X_train_minmax_scaled)
    X_test_minmax_scaled = imputer.transform(X_test_minmax_scaled)

    # 4. Membuat dan melatih model K-Nearest Neighbors (KNN)
    knn_model = KNeighborsClassifier(n_neighbors=35)
    # 0.7178571428571429
    knn_model.fit(X_train_minmax_scaled, y_train)

    # 5. Membuat dan melatih model Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=1)
    nn_model.fit(X_train_minmax_scaled, y_train)

    y_pred_knn = knn_model.predict(X_test_minmax_scaled)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("Akurasi Model KNN:", accuracy_knn)

    y_pred_nn = nn_model.predict(X_test_minmax_scaled)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    print("Akurasi Model Neural Network:", accuracy_nn)


    # 6. Menyimpan kedua model
    joblib.dump(knn_model, 'knn/knn_model.pkl')
    joblib.dump(nn_model, 'knn/nn_model.pkl')

    # 7. Melakukan prediksi dengan kedua model dengan data dummy

    # Mengatasi NaN di dummy_data_scaled dengan SimpleImputer
    imputer_dummy = SimpleImputer(strategy='mean')
    dummy_data_scaled_imputed = imputer_dummy.fit_transform(dummy_data_scaled)


    # Prediksi dengan model KNN
    dummy_pred_knn = knn_model.predict(dummy_data_scaled_imputed)

    # Prediksi dengan model Neural Network
    dummy_pred_nn = nn_model.predict(dummy_data_scaled_imputed)

    hasilKnn = "Prediksi Model KNN untuk Data Dummy:", dummy_pred_knn
    ModelKnn = "Prediksi Model Neural Network untuk Data Dummy:", dummy_pred_nn
    st.text(hasilKnn)
    st.text(ModelKnn)


    from sklearn.decomposition import PCA
    pca = PCA(n_components=18)
    pca.fit(X)
    pca = pca.transform(X)
    st.text(pca)
    

