from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.ml.regression import (
    RandomForestRegressor,
    DecisionTreeRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator
import time

if __name__ == "__main__":

    print("1. Menyalakan Mesin PySpark...")

    spark = SparkSession.builder \
        .appName("ML_Final_Sesuai_Proposal") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("2. Membaca data dari lapisan Gold...")

    df_gold = spark.read.parquet(
        "dataset_ml_ready.parquet"
    )

    print(f"Jumlah data Gold: {df_gold.count()}")

    train_data, test_data = df_gold.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    evaluator_rmse = RegressionEvaluator(
        labelCol="harga",
        predictionCol="prediction",
        metricName="rmse"
    )

    evaluator_r2 = RegressionEvaluator(
        labelCol="harga",
        predictionCol="prediction",
        metricName="r2"
    )

    print("\n=============================================")
    print(" TAHAP 1: PEMBUKTIAN KECEPATAN CACHE")
    print("=============================================")

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="harga",
        numTrees=30,
        maxDepth=10,
        maxBins=100
    )

    # TANPA CACHE
    start_rf_nocache = time.time()

    rf.fit(train_data)

    waktu_rf_nocache = (
        time.time() - start_rf_nocache
    )

    print(
        f"Waktu belajar RF TANPA cache : "
        f"{waktu_rf_nocache:.2f} detik"
    )

    # DENGAN CACHE
    train_data.cache()
    train_data.count()

    start_rf_cache = time.time()

    model_rf = rf.fit(train_data)

    waktu_rf_cache = (
        time.time() - start_rf_cache
    )

    print(
        f"Waktu belajar RF DENGAN cache: "
        f"{waktu_rf_cache:.2f} detik"
    )

    print("\n=============================================")
    print(" TAHAP 2: ADU AKURASI MODEL")
    print("=============================================")

    prediksi_rf = model_rf.transform(
        test_data
    )

    rmse_rf = evaluator_rmse.evaluate(
        prediksi_rf
    )

    r2_rf = (
        evaluator_r2.evaluate(
            prediksi_rf
        ) * 100
    )

    print("\n--- RANDOM FOREST ---")

    print(
        f"RMSE : {rmse_rf:.2f}"
    )

    print(
        f"R²   : {r2_rf:.2f}%"
    )

    dt = DecisionTreeRegressor(
        featuresCol="features",
        labelCol="harga",
        maxDepth=10,
        maxBins=100
    )

    start_dt = time.time()

    model_dt = dt.fit(train_data)

    waktu_dt = (
        time.time() - start_dt
    )

    prediksi_dt = model_dt.transform(
        test_data
    )

    rmse_dt = evaluator_rmse.evaluate(
        prediksi_dt
    )

    r2_dt = (
        evaluator_r2.evaluate(
            prediksi_dt
        ) * 100
    )

    print("\n--- DECISION TREE ---")

    print(
        f"Waktu Training : "
        f"{waktu_dt:.2f} detik"
    )

    print(
        f"RMSE : {rmse_dt:.2f}"
    )

    print(
        f"R²   : {r2_dt:.2f}%"
    )

    print("\n=============================================")
    print(" FEATURE IMPORTANCE RANDOM FOREST")
    print("=============================================")

    print(
        model_rf.featureImportances
    )

    print("\n=============================================")
    print(" CONTOH HASIL PREDIKSI")
    print("=============================================")

    print("\n--- RANDOM FOREST ---")

    prediksi_rf.selectExpr(
        "harga as Harga_Asli",
        "prediction as Prediksi_RF"
    ).show(5, False)

    print("\n--- DECISION TREE ---")

    prediksi_dt.selectExpr(
        "harga as Harga_Asli",
        "prediction as Prediksi_DT"
    ).show(5, False)

    print("\n=============================================")
    print(" MENYIMPAN MODEL")
    print("=============================================")

    model_rf.write() \
        .overwrite() \
        .save("model_random_forest")

    model_dt.write() \
        .overwrite() \
        .save("model_decision_tree")

    print(
        "Model Random Forest tersimpan."
    )

    print(
        "Model Decision Tree tersimpan."
    )

    print("\n=============================================")
    print(" KESIMPULAN")
    print("=============================================")

    if r2_rf > r2_dt:
        print(
            "Random Forest memiliki performa terbaik."
        )
    else:
        print(
            "Decision Tree memiliki performa terbaik."
        )

    print(
        f"Selisih Akurasi : "
        f"{abs(r2_rf-r2_dt):.2f}%"
    )

    train_data.unpersist()

    print(
        "\n=== PROSES SELESAI 100% ==="
    )

    spark.stop()