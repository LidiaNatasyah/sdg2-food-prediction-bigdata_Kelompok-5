from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml import Pipeline
import boto3
import os

if __name__ == "__main__":

    print("1. Menyalakan Mesin PySpark...")

    spark = SparkSession.builder \
        .appName("SilverToGold_PureSpark") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("2. Membaca data Silver...")

    df_spark = spark.read.parquet(
        "dataset_terpadu.parquet"
    )

    print("3. Mencari nama kolom otomatis...")

    cols = df_spark.columns

    col_prov = next(
        (c for c in cols if "provinsi" in c.lower()),
        None
    )

    col_kom = next(
        (
            c for c in cols
            if "komoditas" in c.lower()
            or "bahan" in c.lower()
        ),
        None
    )

    col_harga = next(
        (c for c in cols if "harga" in c.lower()),
        None
    )

    print(
        f"[INFO] Provinsi='{col_prov}', "
        f"Komoditas='{col_kom}', "
        f"Harga='{col_harga}'"
    )

    print("4. Mengubah tahun menjadi numerik...")

    df_spark = df_spark.withColumn(
        "tahun",
        col("tahun").cast("double")
    )

    print("5. Feature Engineering...")

    indexer = StringIndexer(
        inputCols=[
            col_prov,
            col_kom,
            "bulan"
        ],
        outputCols=[
            "provinsi_index",
            "komoditas_index",
            "bulan_index"
        ],
        handleInvalid="keep"
    )

    encoder = OneHotEncoder(
        inputCols=[
            "provinsi_index",
            "komoditas_index",
            "bulan_index"
        ],
        outputCols=[
            "provinsi_vec",
            "komoditas_vec",
            "bulan_vec"
        ]
    )

    fitur_numerik = [
        "tahun",
        "avg_curah_hujan",
        "avg_suhu",
        "avg_kurs"
    ]

    fitur_final = [
        "provinsi_vec",
        "komoditas_vec",
        "bulan_vec"
    ] + fitur_numerik

    assembler = VectorAssembler(
        inputCols=fitur_final,
        outputCol="features",
        handleInvalid="keep"
    )

    pipeline = Pipeline(
        stages=[
            indexer,
            encoder,
            assembler
        ]
    )

    model = pipeline.fit(df_spark)

    df_gold = model.transform(df_spark)

    jumlah_data = df_gold.count()

    print(
        f"Jumlah data Gold: {jumlah_data}"
    )



    print("6. Menyimpan Gold Layer...")

    (
        df_gold
        .selectExpr(
            "features",
            f"`{col_harga}` as harga"
        )
        .write
        .mode("overwrite")
        .parquet("dataset_ml_ready.parquet")
    )

    print(
        "7. Mengirim Gold Layer ke MinIO..."
    )

    s3 = boto3.client(
        's3',
        endpoint_url='http://localhost:9000',
        aws_access_key_id='admin',
        aws_secret_access_key='admin123'
    )

    folder_lokal = "dataset_ml_ready.parquet"
    bucket_tujuan = "gold"

    for root, dirs, files in os.walk(folder_lokal):

        for file in files:

            lokasi_asli = os.path.join(
                root,
                file
            )

            nama_di_minio = (
                lokasi_asli
                .replace("\\", "/")
            )

            s3.upload_file(
                lokasi_asli,
                bucket_tujuan,
                nama_di_minio
            )

    print(
        "SUKSES! Gold Layer berhasil "
        "disimpan ke MinIO."
    )

    spark.stop()

    print(
        "=== FEATURE ENGINEERING SELESAI ==="
    )