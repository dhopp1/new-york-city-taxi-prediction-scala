// Databricks notebook source
val test = spark.read.format("csv").option("header", "true").load("FileStore/tables/test.csv")
val train = spark.read.format("csv").option("header", "true").load("FileStore/tables/train.csv")

// COMMAND ----------

//feature engineering
def create_features(df: org.apache.spark.sql.DataFrame, train_set: Boolean): org.apache.spark.sql.DataFrame = {
   //converting strings to double
  var columns = Array("fare_amount", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count") 
  var temp = df
  //robust to a train or test set, without the fare amount column
  for(column <- columns){
    if(train_set || column != "fare_amount"){
       temp = temp.withColumn(column, temp(column).cast(org.apache.spark.sql.types.DoubleType))
    }
  }
  //changing datetimes from strings to datetimes
  temp = temp.withColumn("pickup_datetime", org.apache.spark.sql.functions.to_timestamp(temp("pickup_datetime"), "yyyy-MM-dd HH:mm:ss"))
  return temp
}

// COMMAND ----------

//making function to get day of week from datetime
def get_dow(input: String): java.time.DayOfWeek = {
    java.time.LocalDate.parse(input,java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")).getDayOfWeek
}
//to apply to everything in a column
val round_tenths_place_udf = udf(round_tenths_place _)
bid_results.withColumn(
  "bid_price_bucket", val round_tenths_place_udf($"bid_price"))

// COMMAND ----------

train.withColumn("test", get_dow(train("pickup_datetime")))
