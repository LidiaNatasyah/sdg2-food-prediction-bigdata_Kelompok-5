import boto3
import pandas as pd
from io import BytesIO

print("Mengambil data dari MinIO...\n")
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='admin',
    aws_secret_access_key='admin123'
)

# Pilih file yang mau dilihat, misalnya dataset_terpadu.parquet dari Silver Layer
response = s3.get_object(Bucket='silver', Key='dataset_terpadu.parquet')
df = pd.read_parquet(BytesIO(response['Body'].read()))

print("=== 5 BARIS PERTAMA DATA ===")
# Menampilkan semua kolom agar tidak terpotong (opsional tapi disarankan)
pd.set_option('display.max_columns', None) 
print(df.head())

print("\n=== INFORMASI KOLOM DAN JUMLAH DATA ===")
print(df.info())


# Hapus baris 'from pyspark.sql import functions as F' jika ada

komoditas_utama = ["Daging Sapi Murni", "Cabai Rawit Merah", "Beras Premium"]

# CATATAN: Jika nama variabel penampung datamu bukan 'df', 
# silakan ganti kata 'df' di bawah ini dengan nama variabelmu yang sebenarnya.
df_filter = df[df["komoditas"].isin(komoditas_utama)]

# Menghitung Nilai Terendah, Tertinggi, dan Rata-rata menggunakan Pandas
df_statistik = df_filter.groupby("komoditas")["harga"].agg(
    Nilai_Terendah="min",
    Nilai_Tertinggi="max",
    Rata_rata="mean"
).reset_index()

# Membulatkan nilai rata-rata agar rapi (tidak ada angka desimal panjang)
df_statistik["Rata_rata"] = df_statistik["Rata_rata"].round(0)

# Menampilkan hasil berupa tabel di terminal
print("\n=== STATISTIK HARGA KOMODITAS ===")
print(df_statistik.to_string(index=False))