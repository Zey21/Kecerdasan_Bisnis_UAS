#Import Modul
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Preprocessing function
def preprocess_data(data):
    # Drop any missing values
    data.dropna(inplace=True)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data


st.title("Kelompok 2 - Data Garam_AHC Multiple Linkage")

# inisialisasi data
tab1, tab2 = st.tabs(["Description data", "Processing"])

with tab1:
    st.subheader("Deskripsi")
    st.write(
        "Analisis Klastering Menggunakan AHC Multiple Linkage pada Data Garam")
    st.caption(""" Dalam analisis ini, kami menggunakan metode AHC multiple linkage untuk menjelajahi struktur 
    klaster dalam data garam. Metode ini memungkinkan kami mengelompokkan data garam ke dalam klaster-klaster 
    berdasarkan kesamaan mereka, membantu dalam memahami pola dan hubungan di antara mereka.""")

with tab2:
    st.subheader("Processing Data \nAplikasi Upload Dataset")
    
    # Tambahkan komponen untuk mengunggah file
    uploaded_file = st.file_uploader("Unggah file dataset (format: CSV)", type="csv")
    
    if uploaded_file is not None:
        # Baca file dataset
        df = pd.read_csv(uploaded_file,  encoding='utf-8', sep=';')
        data = df
        
        # Tampilkan preview dataset
        st.subheader("Preview Dataset")
        st.write(df.head())

        # Memisahkan fitur (X) dan label (y)
        X = data.drop('Grade', axis=1)
        y = data['Grade']

        # Memisahkan data menjadi data latihan dan data pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data_scaler = preprocess_data(X_train)
        X_train = data_scaler
        st.write("Data setelah diproses dengan Scaller")
        st.write(data_scaler)
        
        result_ahc = []
        result_kmeans = []

        def AHC(cluster):
            # Menerapkan AHC
            clustering = AgglomerativeClustering(n_clusters= cluster) #2-9 mencari siluet data yang bagus dari perbandingan uji coba ncluster dari nc 2- 9
            clustering.fit(X.values)
            
            labels = clustering.fit_predict(X.values)
            st.subheader("AHC - ncluster : {}".format(cluster))
            
            #Penambahan Siluet dan Koefisien 
            # Menghitung nilai siluet untuk setiap titik data
            silhouette_avg = silhouette_score(X, labels)
            sample_silhouette_values = silhouette_samples(X, labels)

            # Menggambar siluet
            plt.figure()
            fig, ax = plt.subplots()
            y_lower = 10
            for i in range(cluster):
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = plt.cm.get_cmap("Spectral")(i / 3)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10

            ax.set_title("Silhouette plot")
            ax.set_xlabel("Silhouette coefficient values")
            ax.set_ylabel("Cluster label")

            # Batas sumbu x
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")
            result_ahc.append(silhouette_avg)
            st.pyplot(plt)
        
        def K_means(cluster):
            # Membuat model K-Means
            kmeans = KMeans(n_clusters=cluster, random_state=42)
            kmeans.fit(X)

            # Mendapatkan label klaster
            labels = kmeans.labels_
            
            st.subheader("K_means - ncluster : {}".format(cluster))

            # Menghitung nilai siluet untuk setiap titik data
            silhouette_avg = silhouette_score(X, labels)
            sample_silhouette_values = silhouette_samples(X, labels)

            st.write("Output :")
            # Menggambar siluet
            plt.figure()
            fig, ax = plt.subplots()
            y_lower = 10
            for i in range(cluster):
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = plt.cm.get_cmap("Spectral")(i / 4)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10

            ax.set_title("Silhouette plot")
            ax.set_xlabel("Silhouette coefficient values")
            ax.set_ylabel("Cluster label")

            # Batas sumbu x
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")
            result_kmeans.append(silhouette_avg)
            
            st.pyplot(plt)
        
        ################################################################
        #Call :
        AHC(int(2))
        AHC(int(3))
        AHC(int(4))
        AHC(int(5))
        AHC(int(6))
        AHC(int(7))
        AHC(int(8))
        AHC(int(9))
        
        # Mencari nilai terbesar dalam list
        max_value = max(result_ahc)

        # Mencari indeks nilai terbesar dalam list
        max_index = result_ahc.index(max_value)

        st.subheader("Kesimpulan :\n Dari beberapa model AHC n_cluster(2-9) yang memiliki Siluet terbaik adalah n_cluster {}, yakni {}".format(max_index+2,max_value))
        
        #Call :
        
        K_means(int(2))
        K_means(int(3))
        K_means(int(4))
        K_means(int(5))
        K_means(int(6))
        K_means(int(7))
        K_means(int(8))
        K_means(int(9))
        
        
        # Mencari nilai terbesar dalam list
        max_value2 = max(result_kmeans)

        # Mencari indeks nilai terbesar dalam list
        max_index2 = result_kmeans.index(max_value2)

        st.subheader("Kesimpulan :\n Dari beberapa model AHC n_cluster(2-9) yang memiliki Siluet terbaik adalah n_cluster {}, yakni {}".format(max_index2+2,max_value2))
