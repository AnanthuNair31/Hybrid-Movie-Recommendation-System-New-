from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from pyspark.sql.functions import avg, count

spark = SparkSession.builder \
    .appName("HybridMovieRecommender") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()





"""movies = spark.read.csv("data/movies.csv", header=True, inferSchema=True)
ratings = spark.read.csv("data/ratings.csv", header=True, inferSchema=True)
tags = spark.read.csv("data/tags.csv", header=True, inferSchema=True)

print("Movies:",movies.count())
print("Ratings:",ratings.count())
print("Tags:",tags.count())


ratings = ratings.dropna(subset=["rating"])
tags = tags.dropDuplicates(["userId", "movieId","tag"])
ratings =ratings.drop("timestamp")
tags =tags.drop("timestamp")

ratings = ratings \
    .withColumn("userId", ratings.userId.cast("int")) \
    .withColumn("movieId", ratings.movieId.cast("int")) \
    .withColumn("rating", ratings.rating.cast("float"))

dataset = ratings \
    .join(movies, on = "movieId", how = "left") \
    .join(tags, on = ["userId","movieId"], how = "left")

dataset = dataset.fillna({"tag": "unknown"})
print(dataset.show(20))

als_data = dataset.select("userId", "movieId", "rating")

hybrid_data = dataset.select(
    "userId",
    "movieId",
    "rating",
    "genres",
    "tag")

train,validation,test = als_data.randomSplit([0.8,0.1,0.1], seed = 42)
hybrid_data.write \
    .mode("overwrite") \
    .parquet("model/hybrid_dataset")

hybrid_data = spark.read.parquet("model/hybrid_dataset")
hybrid_data.printSchema()
als_data = hybrid_data.select(
    "userId",
    "movieId",
    "rating"
)
als_data = als_data.dropna()
als_data = als_data.repartition(200).cache()
als_data.count()


als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    rank=20,
    maxIter=10,
    regParam=0.1
)
als_model = als.fit(als_data)
user_candidates = als_model.recommendForAllUsers(50)

candidates = user_candidates.select(
    "userId",
    explode("recommendations").alias("rec")
).select(
    "userId",
    "rec.movieId",
    "rec.rating"
).withColumnRenamed("rating", "ALS_score")

candidates.write \
    .mode("overwrite") \
    .parquet("model/als_candidates")"""

hybrid_data = spark.read.parquet("model/hybrid_dataset")
als_candidates = spark.read.parquet("model/als_candidates")



movie_stats = hybrid_data.groupBy("movieId").agg(
    avg("rating").alias("avg_rating"),
    count("rating").alias("rating_count")
)

ranking_data = als_candidates.join(
    movie_stats,
    on="movieId",
    how="left"
)

movie_genres = hybrid_data.select(
    "movieId",
    "genres"
).dropDuplicates()

ranking_data = ranking_data.join(
    movie_genres,
    on="movieId",
    how="left"
)
movie_tags = hybrid_data.select(
    "movieId",
    "tag"
).dropDuplicates()

ranking_data = ranking_data.join(
    movie_tags,
    on="movieId",
    how="left"
)
labels = hybrid_data.select(
    "userId",
    "movieId",
    "rating"
).withColumnRenamed("rating", "label")

ranking_data = ranking_data.join(
    labels,
    ["userId", "movieId"],
    "left"
)
ranking_data = ranking_data.fillna({"label": 0})

ranking_data.write \
    .mode("overwrite") \
    .parquet("model/ranking_dataset")

