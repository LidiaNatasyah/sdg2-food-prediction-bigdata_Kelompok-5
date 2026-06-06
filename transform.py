from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    current_timestamp,
    regexp_replace,
    col
)
import boto3
from io import BytesIO
import pandas as pd

if __name__ == "__main__":

    print("1. Menyalakan Mesin PySpark...")

    spark = SparkSession.builder \
        .appName("BronzeToSilver_PySpark") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("2. Menghubungkan ke MinIO...")

    s3 = boto3.client(
        's3',
        endpoint_url='http://localhost:9000',
        aws_access_key_id='admin',
        aws_secret_access_key='admin123'
    )

    def baca_csv_minio(nama_file, sep):
        obj = s3.get_object(Bucket='bronze', Key=nama_file)

        pdf = pd.read_csv(
            BytesIO(obj['Body'].read()),
            sep=sep
        )

        pdf.columns = pdf.columns.str.lower().str.strip()

        return spark.createDataFrame(pdf)

    print("3. Mengambil data Bronze...")

    df_harga = baca_csv_minio(
        'harga_pangan.csv',
        ';'
    )

    df_cuaca = baca_csv_minio(
        'Data_Cuaca_Harian_Indonesia_2021_2026.csv',
        ','
    )

    df_kurs = baca_csv_minio(
        'Data_Kurs_Harian_2021_2026.csv',
        ','
    )

    print("4. Membersihkan kolom harga...")

    df_harga = df_harga.withColumn(
        "harga",
        regexp_replace(
            col("harga").cast("string"),
            r"[Rp,\s\-]",
            ""
        )
    )

    df_harga = df_harga.withColumn(
        "harga",
        col("harga").cast("double")
    )

    print("5. Agregasi bulanan menggunakan Spark...")

    df_cuaca_bln = (
        df_cuaca
        .groupBy(
            "nama provinsi",
            "tahun",
            "bulan"
        )
        .agg(
            avg("curah_hujan_mm").alias("avg_curah_hujan"),
            avg("suhu_rata_c").alias("avg_suhu")
        )
    )

    df_kurs_bln = (
        df_kurs
        .groupBy(
            "tahun",
            "bulan"
        )
        .agg(
            avg("kurs_usd_idr").alias("avg_kurs")
        )
    )

    print("6. Menyamakan tipe data untuk Join...")

    for c in ["tahun", "bulan"]:

        df_harga = df_harga.withColumn(
            c,
            col(c).cast("string")
        )

        df_cuaca_bln = df_cuaca_bln.withColumn(
            c,
            col(c).cast("string")
        )

        df_kurs_bln = df_kurs_bln.withColumn(
            c,
            col(c).cast("string")
        )

    print("7. Menjalankan Left Join...")

    df_join_1 = (
        df_harga
        .join(
            df_cuaca_bln,
            ["nama provinsi", "tahun", "bulan"],
            "left"
        )
    )

    df_terpadu = (
        df_join_1
        .join(
            df_kurs_bln,
            ["tahun", "bulan"],
            "left"
        )
    )

    print("8. Menangani Missing Value...")

    df_silver = df_terpadu.na.drop()

    df_silver = df_silver.withColumn(
        "processed_at",
        current_timestamp()
    )

    jumlah_data = df_silver.count()

    print(f"Jumlah data Silver: {jumlah_data}")

    print("9. Menyimpan Silver Layer...")

    df_silver.write \
        .mode("overwrite") \
        .parquet("dataset_terpadu.parquet")

    print("=== SILVER LAYER BERHASIL DIBUAT ===")

    spark.stop()