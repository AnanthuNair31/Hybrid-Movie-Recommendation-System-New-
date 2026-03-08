from pyspark.sql import SparkSession
import xgboost as xgb
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

spark = SparkSession.builder \
    .appName("RankingModelTraining") \
    .master("local[*]") \
    .getOrCreate()

ranking_data = spark.read.parquet("model/ranking_dataset")

features = ranking_data.select(
    "userId",
    "movieId",
    "ALS_score",
    "avg_rating",
    "rating_count",
    "label"
)

features = features.fillna({
    "ALS_score": 0,
    "avg_rating": 0,
    "rating_count": 0,
    "label": 0
})



window = Window.partitionBy("userId") \
               .orderBy(col("ALS_score").desc())

features = features.withColumn(
    "rank",
    row_number().over(window)
).filter(col("rank") <= 20).drop("rank")

features = features.coalesce(4)



pandas_df = features.toPandas()

pandas_df = pandas_df.sort_values("userId")

X = pandas_df[[
    "ALS_score",
    "avg_rating",
    "rating_count"
]]

y = pandas_df["label"]

group = pandas_df.groupby("userId").size().to_list()



ranker = xgb.XGBRanker(
    objective="rank:pairwise",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8
)

ranker.fit(
    X,
    y,
    group=group
)

ranker.save_model("model/xgb_ranker.json")
