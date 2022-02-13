#import module
from pyspark.sql import SparkSession
from pyspark.sql.types import *
# create session in order to be capable of accessing all Spark API
spark = SparkSession \
    .builder \
    .appName("Introdution to Spark DataFrame") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
# define data schema for file we want to read
purchaseSchema = StructType([
    StructField("Date", DateType(), True),
    StructField("Time", StringType(), True),
    StructField("City", StringType(), True),
    StructField("Item", StringType(), True),
    StructField("Total", FloatType(), True),
    StructField("Payment", StringType(), True), ])
# read csv file with our defined schema into Spark DataFrame, and use "tab" delimiter
purchaseDataframe = spark.read.csv(
    "dataset/purchases.csv",
    header=True, schema=purchaseSchema, sep="\t")
# show 3 rows of our DataFrame
purchaseDataframe.show(3)
