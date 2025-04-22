# ml_pipeline_spark.py

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
from neo4j import GraphDatabase

def main():
    # 1) Start SparkSession (clean each run)
    spark = SparkSession.builder \
        .appName("OrdersReturnsPrediction") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

    # 2) Load cleaned CSV
    df = spark.read.csv("orders_clean.csv", header=True, inferSchema=True)

    # 3) Create returned_flag label
    df = df.withColumn("returned_flag", (col("returned_count") > 0).cast("integer"))

    # 4) Prepare categorical features
    cats = ["ship_mode","customer_segment","region","category","sub_category"]
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in cats]
    encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in cats]

    # 5) Assemble all features
    numeric = ["sales","quantity","discount","profit"]
    assembler = VectorAssembler(
        inputCols=numeric + [c+"_vec" for c in cats],
        outputCol="features"
    )

    # 6) Random Forest classifier
    rf = RandomForestClassifier(labelCol="returned_flag", featuresCol="features", numTrees=20)

    # 7) Build pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

    # 8) Train/test split & fit
    train, test = df.randomSplit([0.8,0.2], seed=42)
    model = pipeline.fit(train)
    pred = model.transform(test)

    # 9) Evaluate
    evaluator = BinaryClassificationEvaluator(
        labelCol="returned_flag", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(pred)
    print(f"Test ROC AUC = {auc:.3f}")

    # 10) Export predictions
    pred_df = pred.select("order_id","returned_flag","prediction").toPandas()
    pred_df.to_csv("order_return_predictions.csv", index=False)

    # 11) Optionally write back to Neo4j
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","your_password"))
    with driver.session() as session:
        for row in pred_df.itertuples():
            session.run(
                "MATCH (o:Order {id:$order_id}) SET o.predicted_return = $prediction",
                order_id=row.order_id, prediction=int(row.prediction)
            )
    driver.close()

    # 12) Stop Spark
    spark.stop()

if __name__ == "__main__":
    main()
