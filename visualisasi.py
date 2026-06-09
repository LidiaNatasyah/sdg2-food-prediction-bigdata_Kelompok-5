from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    print("1. Menyalakan Mesin PySpark...")
    spark = SparkSession.builder \
        .appName("Bikin_Grafik_Dan_Cek_Fitur") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("2. Membaca Data dan Model yang sudah disimpan...")
    # Baca data yang sudah siap
    df_gold = spark.read.parquet("dataset_ml_ready.parquet")
    
    # Baca model yang sudah dilatih
    model_rf = RandomForestRegressionModel.load("model_random_forest")

    print("3. Mengekstrak Nama Asli Fitur dari Metadata...")
    # Mengambil "kamus" tersembunyi dari dalam data
    metadata = df_gold.schema["features"].metadata
    kamus_fitur = {}
    
    if "ml_attr" in metadata and "attrs" in metadata["ml_attr"]:
        attrs = metadata["ml_attr"]["attrs"]
        for tipe_atribut in attrs.keys():
            for attr in attrs[tipe_atribut]:
                kamus_fitur[attr["idx"]] = attr["name"]
    
    print("4. Membuat Prediksi...")
    prediksi_rf = model_rf.transform(df_gold)

    print("5. Mengubah ke format Pandas untuk digambar...")
    pd_hasil = prediksi_rf.select("harga", "prediction").sample(fraction=0.1, seed=42).limit(100).toPandas()

    print("6. Menggambar Grafik Aktual vs Prediksi...")
    plt.figure(figsize=(12, 6))
    plt.plot(pd_hasil['harga'], label='Harga Asli', color='blue', marker='o', linestyle='dashed', linewidth=1, markersize=4)
    plt.plot(pd_hasil['prediction'], label='Prediksi Random Forest', color='red', marker='x', linewidth=2, markersize=4)
    plt.title('Perbandingan Harga Asli vs Prediksi (100 Sampel Acak)')
    plt.xlabel('Indeks Data')
    plt.ylabel('Harga')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafik_harga_vs_prediksi.png')
    plt.close()

    print("7. Menggambar Grafik Feature Importance dengan Nama Asli...")
    importances = model_rf.featureImportances.toArray()
    
    # Menerjemahkan indeks angka menjadi nama asli menggunakan kamus
    nama_fitur_list = []
    for i in range(len(importances)):
        if i in kamus_fitur:
            nama_fitur_list.append(kamus_fitur[i])
        else:
            # Jika karena suatu hal namanya tidak ada, biarkan tetap Fitur_X
            nama_fitur_list.append(f"Fitur_{i}")
    
    # Membuat tabel untuk 10 fitur teratas
    df_importance = pd.DataFrame({
        'Nama_Fitur': nama_fitur_list,
        'Kepentingan': importances
    }).sort_values(by='Kepentingan', ascending=False).head(10)

    # Cetak langsung fitur paling penting di terminal agar kamu langsung tahu
    fitur_tertinggi = df_importance.iloc[0]['Nama_Fitur']
    nilai_tertinggi = df_importance.iloc[0]['Kepentingan']
    print(f"\n[INFO PENTING] Fitur Paling Berpengaruh adalah: {fitur_tertinggi} ({nilai_tertinggi*100:.2f}%)\n")

    # Gambar grafiknya
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Kepentingan', y='Nama_Fitur', data=df_importance, palette='viridis')
    plt.title('10 Fitur Paling Berpengaruh dalam Menentukan Harga')
    plt.xlabel('Tingkat Kepentingan')
    plt.ylabel('Nama Fitur')
    plt.tight_layout()
    plt.savefig('grafik_feature_importance.png')
    plt.close()

    print("=== SELESAI! Cek grafik barumu, namanya sudah otomatis berubah ===")
    spark.stop()