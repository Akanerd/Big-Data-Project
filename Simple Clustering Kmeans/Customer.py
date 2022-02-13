# Import Modules
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from sympy import vectorize

# Create Session
appName = "Clustering Customer in Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Membaca file csv 
customers = spark.read.csv('dataset/customer.csv',inferSchema=True, header=True)
customers.show(5)

#Penggabungan data
assembler = VectorAssembler(inputCols= 
["Sex","Marital status","Age","Education",
"Income","Occupation","Settlement size"], outputCol="features")

data = assembler.transform(customers).select('ID','features')
data.show(truncate = False, n=5)

#define kMeans clustering algorithm
kmeans = KMeans(
    featuresCol=assembler.getOutputCol(), 
    predictionCol="cluster", k=5)
model = kmeans.fit(data)
print ("Model is successfully trained!")

# Print centroid for each cluster
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Cluster the data
prediction = model.transform(data)#cluster given data
prediction.groupBy("cluster").count().orderBy("cluster").show()#count members in each cluster
prediction.select('ID', 'cluster').show(5)#show several clustered data
