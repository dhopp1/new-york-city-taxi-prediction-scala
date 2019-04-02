// Databricks notebook source
val test = spark.read.format("csv").option("header", "true").load("FileStore/tables/test.csv")
val train = spark.read.format("csv").option("header", "true").load("FileStore/tables/train.csv")

// COMMAND ----------

//feature engineering
def create_features(df: org.apache.spark.sql.DataFrame, train_set: Boolean): org.apache.spark.sql.DataFrame = {
   //converting strings to double
  var columns = Array("fare_amount", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count") 
  var temp = df
  for(column <- columns){
    if(train_set || column != "fare_amount"){
       temp = temp.withColumn(column, temp(column).cast(org.apache.spark.sql.types.DoubleType))
    }
  }
  return temp
}
