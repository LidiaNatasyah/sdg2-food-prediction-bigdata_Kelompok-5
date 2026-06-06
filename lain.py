from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("dataset_terpadu.parquet")

df.select("bulan").distinct().show(20, False)