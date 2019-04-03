// Databricks notebook source
import org.apache.spark.sql.{DataFrame, types, functions}
import org.apache.spark.ml.{Pipeline, PipelineModel, feature, evaluation, regression, tuning}

val test = spark.read.format("csv").option("header", "true").load("FileStore/tables/test.csv")
val train = spark.read.format("csv").option("header", "true").load("FileStore/tables/train.csv")

// COMMAND ----------

//feature engineering
def create_features(df: DataFrame, train_set: Boolean = true): DataFrame = {
   //converting strings to double
  var columns = Array("fare_amount", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count") 
  var temp = df
  //robust to a train or test set, without the fare amount column
  for(column <- columns){
    if(train_set || column != "fare_amount"){
       temp = temp.withColumn(column, temp(column).cast(types.DoubleType))
    }
  }
  
  //getting dow
  temp.createOrReplaceTempView("train")
  var dow = "SELECT *, DATE_FORMAT(TO_TIMESTAMP(SUBSTRING(pickup_datetime, 1, 19)), 'EEEE') AS dow FROM train"
  temp = spark.sql(dow)
  
  //converting day of week to one hot encoding
  val features = temp.columns.filter(_.contains("dow"))
  val encodedFeatures = features.flatMap{ name =>
    val stringIndexer = new feature.StringIndexer()
      .setInputCol(name)
      .setOutputCol(name + "_index")

    val oneHotEncoder = new feature.OneHotEncoderEstimator()
      .setInputCols(Array(name + "_index"))
      .setOutputCols(Array(name + "_vec"))
      .setDropLast(false)

    Array(stringIndexer, oneHotEncoder)
  }
  val pipeline = new Pipeline().setStages(encodedFeatures)
  val indexer_model = pipeline.fit(temp)
  temp = indexer_model.transform(temp)
  
  //changing datetimes from strings to datetimes
  temp = temp.withColumn("pickup_datetime", functions.to_timestamp(temp("pickup_datetime"), "yyyy-MM-dd HH:mm:ss"))
  
  //create columns for year, month, day, hour, minute
  temp = temp.withColumn("year", functions.year(temp("pickup_datetime")))
  temp = temp.withColumn("month", functions.month(temp("pickup_datetime")))
  temp = temp.withColumn("day", functions.dayofmonth(temp("pickup_datetime")))
  temp = temp.withColumn("hour", functions.hour(temp("pickup_datetime")))
  temp = temp.withColumn("minute", functions.minute(temp("pickup_datetime")))
  
  //drop fares <= 0
  temp = temp.filter(temp("fare_amount") > 0)
  //drop nas
  temp = temp.na.drop()
  //renaming fare_amount to labels
  temp = temp.withColumn("labels", temp("fare_amount"))
  //dropping unecessary columns
  val drop_columns = Array("key", "fare_amount", "pickup_datetime", "dow", "dow_index")
  for(drop <- drop_columns){
    temp = temp.drop(drop)
  }
  
  return temp
}

//creating engineered training data
val training_data = create_features(train)

// COMMAND ----------

//model: PipelineModel
def test_algorithm(data: DataFrame, evaluator_string: String, folds: Int = 2): (DataFrame, Double, PipeLineModel) = {
  //vec assembler
  val vec_assembler = new feature.VectorAssembler()
  .setInputCols(Array("pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count", "year", "month", "day", "hour", "minute", "dow_vec"))
  .setOutputCol("features")
  
  //pipeline
  val pipeline = new Pipeline()
    .setStages(Array(vec_assembler))
  val piped_data = pipeline.fit(data).transform(data)
  
  //train test split
  val Array(trainingData, testData) = piped_data.randomSplit(Array(0.8, 0.2))
  
  //evaluator
  val evaluator = new evaluation.RegressionEvaluator()
    .setMetricName(evaluator_string)

  //specify model
  val dt = new regression.DecisionTreeRegressor()
    .setLabelCol("labels")
    .setFeaturesCol("features")
  
  //param grid
  val paramGrid = new tuning.ParamGridBuilder()
  .addGrid(dt.maxDepth, Array(5, 10))
  .build()
  
  //cross validator
  val cv = new tuning.CrossValidator()
  .setEstimator(dt)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(folds)

  //train model
  val models = cv.fit(trainingData)
  val best_model = models.bestModel

  // evaluate on the test set
  val predictions = best_model.transform(testData)
  val result = evaluator.evaluate(predictions)

  return (predictions, result, best_model)
}
