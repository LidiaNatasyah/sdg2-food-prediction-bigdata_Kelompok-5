# Prediksi Harga Komoditas Pangan Berbasis PySpark untuk Mendukung SDG 2 (Zero Hunger)

Proyek ini merupakan implementasi pipeline data end-to-end menggunakan **Apache Spark (PySpark)** dan **Medallion Architecture** untuk memprediksi harga komoditas pangan strategis di Indonesia. Sistem mengintegrasikan data harga pangan, data cuaca, dan data kurs mata uang untuk menghasilkan model prediksi yang dapat mendukung pemantauan ketahanan pangan nasional, sejalan dengan tujuan **Sustainable Development Goal (SDG) 2: Zero Hunger**.

---

## 📌 Latar Belakang

Fluktuasi harga pangan merupakan salah satu tantangan utama dalam menjaga stabilitas ketahanan pangan. Faktor eksternal seperti kondisi cuaca dan perubahan nilai tukar mata uang dapat memengaruhi harga komoditas di berbagai wilayah Indonesia.

Melalui pemanfaatan teknologi Big Data dan Machine Learning, proyek ini bertujuan membangun sistem prediksi harga pangan yang mampu mengolah data dari berbagai sumber secara terintegrasi dan efisien menggunakan Apache Spark.

---

## 🏗️ Arsitektur Sistem

## 📊 Dataset dan Sumber Data

Proyek ini mengintegrasikan tiga sumber data utama untuk membangun model prediksi harga pangan:

### 1. Data Harga Pangan
- Sumber: Open Data Badan Pangan Nasional (Bapanas)
- Cakupan: Harga komoditas pangan strategis per provinsi
- Format: CSV
- Peran: Target prediksi (label)

### 2. Data Cuaca Historis
- Sumber: Open-Meteo API
- Variabel:
  - Curah hujan (mm)
  - Suhu rata-rata (°C)
- Cakupan: 38 provinsi di Indonesia
- Format: CSV
- Peran: Fitur prediktor

### 3. Data Kurs USD/IDR
- Sumber: Yahoo Finance
- Variabel:
  - Nilai tukar USD terhadap IDR
- Rentang waktu: 2021–2026
- Format: CSV
- Peran: Fitur makroekonomi

### Hasil Integrasi Data
Setelah proses ETL pada Silver Layer, seluruh sumber data digabungkan menjadi satu dataset terpadu yang berisi:

- Provinsi
- Komoditas
- Tahun
- Bulan
- Harga
- Curah hujan rata-rata bulanan
- Suhu rata-rata bulanan
- Kurs rata-rata bulanan

Proyek ini menerapkan konsep **Medallion Architecture** yang terdiri atas tiga lapisan utama:

### 🥉 Bronze Layer

Lapisan penyimpanan data mentah (*raw data*) yang berasal dari beberapa sumber:

* Data Harga Pangan Nasional (Bapanas)
* Data Cuaca Harian Indonesia
* Data Kurs Harian USD terhadap Rupiah

Seluruh data mentah disimpan dalam object storage menggunakan **MinIO**.

### 🥈 Silver Layer (`transform.py`)

Tahap pemrosesan dan integrasi data yang meliputi:

* Pembersihan data (*data cleaning*)
* Transformasi tipe data
* Agregasi data harian menjadi rata-rata bulanan
* Penggabungan (*join*) data harga, cuaca, dan kurs
* Penanganan *missing values*

Output tahap ini berupa dataset historis terintegrasi yang siap digunakan untuk analisis lebih lanjut.

### 🥇 Gold Layer (`silver_to_gold.py`)

Tahap *feature engineering* menggunakan PySpark ML:

* Encoding variabel kategorik menggunakan **StringIndexer**
* Transformasi fitur kategorik menggunakan **One-Hot Encoding**
* Pembentukan *feature vector* menggunakan **VectorAssembler**

Fitur yang digunakan meliputi:

* Provinsi
* Komoditas
* Bulan
* Tahun
* Curah hujan rata-rata
* Suhu rata-rata
* Kurs rata-rata USD/IDR

Output tahap ini berupa dataset siap latih dalam format yang kompatibel dengan Spark MLlib.

---

## 🤖 Machine Learning (`train_model.py`)

Tahap ini melakukan pelatihan dan evaluasi model menggunakan **Spark MLlib**.

### Algoritma yang Digunakan

1. Random Forest Regressor
2. Decision Tree Regressor

### Metrik Evaluasi

* Root Mean Square Error (RMSE)
* Coefficient of Determination (R²)

### Pengujian Caching Spark

Selain evaluasi model, dilakukan juga pengujian performa komputasi menggunakan fitur **Spark Cache**.

| Kondisi      | Waktu Training |
| ------------ | -------------: |
| Tanpa Cache  |     6.86 detik |
| Dengan Cache |     3.34 detik |

Penggunaan cache berhasil mempercepat proses pelatihan sekitar **51%** dibandingkan pemrosesan langsung dari disk.

---

## 📊 Hasil Eksperimen

| Model         |      RMSE |     R² |
| ------------- | --------: | -----: |
| Random Forest | 11,527.38 | 88.23% |
| Decision Tree | 11,344.47 | 88.60% |

Berdasarkan hasil evaluasi, **Decision Tree Regressor** memberikan performa terbaik dengan nilai:

* RMSE = 11,344.47
* R² = 88.60%

Hasil tersebut menunjukkan bahwa model mampu menjelaskan lebih dari **88% variasi harga pangan** pada data pengujian.

---

## 💾 Penyimpanan Model

Model yang telah dilatih disimpan menggunakan Spark MLlib sehingga dapat digunakan kembali tanpa proses pelatihan ulang:

* `model_random_forest`
* `model_decision_tree`

---

## 🚀 Cara Menjalankan Program

Pastikan Docker Desktop dan MinIO telah berjalan.

### 1. Menjalankan layanan MinIO

```bash
docker-compose up -d
```

### 2. Membuat Silver Layer

```bash
python transform.py
```

### 3. Membuat Gold Layer

```bash
python silver_to_gold.py
```

### 4. Melatih dan Mengevaluasi Model

```bash
python train_model.py
```

---

## 🛠️ Teknologi yang Digunakan

* Apache Spark (PySpark)
* Spark MLlib
* MinIO Object Storage
* Docker
* Python
* Pandas
* Boto3

---

## 👥 Tim Pengembang

| Nama                    | NIM       | GitHub           |
| ----------------------- | --------- | ---------------- |
| Ahmad Rizqi             | 122450138 | @ikii-sd         |
| Lidia Natasyah Marpaung | 123450015 | @LidiaNatasyah   |
| Benget Sidabutar        | 123450047 | @BengetSidabutar |
| Aisyah Musfirah         | 123450084 | @aisyahmusfirah  |

---

## 🎯 Keterkaitan dengan SDGs

Proyek ini mendukung **Sustainable Development Goal (SDG) 2: Zero Hunger** melalui pemanfaatan teknologi Big Data dan Machine Learning untuk membantu pemantauan serta prediksi harga komoditas pangan, sehingga dapat menjadi dasar pengambilan keputusan yang lebih baik dalam menjaga stabilitas ketahanan pangan nasional.

---

> Proyek ini dikembangkan sebagai Tugas Besar Mata Kuliah Analisis Big Data, Program Studi Sains Data, Institut Teknologi Sumatera (ITERA), 2026.
