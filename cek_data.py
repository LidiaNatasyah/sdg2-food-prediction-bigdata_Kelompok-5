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