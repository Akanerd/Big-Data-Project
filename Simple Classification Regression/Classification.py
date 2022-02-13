# Import Modules
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
# import Modules Classification dengan metode Regresi Logistic
from pyspark.ml.classification import LogisticRegression
# import Modules penggabungan atritbut
from pyspark.ml.feature import VectorAssembler

# Membuat Session
appName = "Classification Starbuck Customers with Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Load and read data ke dalam format dataframe
csv = spark.read.csv('dataset/customers.csv',inferSchema=True, header=True)
csv.show(5)

# Data Preprocessing - Drop Columns
dropColumns = ['itemPurchaseCoffee','itempurchaseCold','itemPurchasePastries','itemPurchaseJuices','itemPurchaseSandwiches',
               'itemPurchaseOthers','spendPurchase','productRate','priceRate','promoRate','ambianceRate','wifiRate','serviceRate',
               'chooseRate','promoMethodApp','promoMethodSoc','promoMethodEmail','promoMethodDeal','promoMethodFriend','promoMethodDisplay',
               'promoMethodBillboard','promoMethodOthers','loyal' ]
# Drop kolom yang tidak dibutuhkan               
csv = csv.drop(*dropColumns)
# Hasil Data setelah drop kolom
#csv.show(5)
# Hasil Atribut untuk diklasifikasi
csv.printSchema()

# Data Preprocessing - Drop Duplicates
csv.dropDuplicates().show(5)
df = csv.count()
print("Data yang sudah didrop duplicates :",df)

# Data Preprocessing - Handle missing data
csv.fillna('0')
csv.na.drop().show(5)
df = csv.count()
print("Data yang sudah dihandle missing data :",df)

# Membagi data 70% untuk data training, 30 % untuk data testing
dividedData = csv.randomSplit([0.7, 0.3]) 
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing
train_rows = trainingData.count()
test_rows = testingData.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)

#Inisialisasi Penggabungan data
assembler = VectorAssembler(inputCols = [
    "gender", "age", "status", "income","Id",
    "timeSpend","location","membershipCard"], outputCol="features")
trainingDataFinal = assembler.transform(
    trainingData).select(col("features"), col("visitNo").alias("label"))
trainingDataFinal.show(truncate=False, n=2)

# 4 Inisialisasi klasifikasi dengan metode regresi
classifier = LogisticRegression(
    labelCol="label",featuresCol="features",maxIter=10,regParam=0.3)
# uji data klasifikasi
model = classifier.fit(trainingDataFinal)
print ("Classifier model is trained!")

#  Menyiapkan data testing
testingDataFinal = assembler.transform(
    testingData).select(col("features"), col("visitNo").alias("trueLabel"))
testingDataFinal.show(3)

# 5 Prediksi data testing
prediction = model.transform(testingDataFinal)
predictionFinal = prediction.select(
    "features", "prediction", "probability", "trueLabel")
predictionFinal.show(truncate=False, n=3)
prediction.show(truncate=False, n=3)

# 6 Hitung performa model
correctPrediction = predictionFinal.filter(
    predictionFinal['prediction'] == predictionFinal['trueLabel']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)



