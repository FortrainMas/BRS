from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("MyApp") \
        .master("spark://localhost:7077") \
        .getOrCreate()
        
    spark.read.parquet("s3a://brs/data/raw/kaggle-books/Books.parquet").head(10)
