import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SparkSession, DataFrame, types, functions}
import org.apache.spark.ml.{Pipeline, PipelineModel, feature, evaluation, regression, tuning}

object main extends App {

  val retrain = false

  // setting up spark contexts and sessions
  val conf = new SparkConf()
  conf.setMaster("local")
  conf.setAppName("nyc_taxi")
  val sc = new SparkContext(conf)
  val spark = SparkSession.builder().appName("nyc_taxi").master("local").getOrCreate()

  // reading test/train data
  val test = spark.read.format("csv").option("header", "true").load("data/test.csv")
  val train = spark.read.format("csv").option("header", "true").load("data/train.csv")

  // feature engineering
  def create_features(df: DataFrame, train_set: Boolean = true): DataFrame = {
     // converting strings to double
    var columns = Array("trip_duration", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count") 
    var temp = df
    // robust to a train or test set, without the fare amount column
    for(column <- columns){
      if(train_set || column != "trip_duration"){
         temp = temp.withColumn(column, temp(column).cast(types.DoubleType))
      }
    }
    
    // getting dow
    temp.createOrReplaceTempView("train")
    var dow = "SELECT *, DATE_FORMAT(TO_TIMESTAMP(SUBSTRING(pickup_datetime, 1, 19)), 'EEEE') AS dow FROM train"
    temp = spark.sql(dow)
    
    // converting day of week to one hot encoding
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
    
    // changing datetimes from strings to datetimes
    temp = temp.withColumn("pickup_datetime", functions.to_timestamp(temp("pickup_datetime"), "yyyy-MM-dd HH:mm:ss"))
    
    // create columns for year, month, day, hour, minute
    temp = temp.withColumn("year", functions.year(temp("pickup_datetime")))
    temp = temp.withColumn("month", functions.month(temp("pickup_datetime")))
    temp = temp.withColumn("day", functions.dayofmonth(temp("pickup_datetime")))
    temp = temp.withColumn("hour", functions.hour(temp("pickup_datetime")))
    temp = temp.withColumn("minute", functions.minute(temp("pickup_datetime")))
    
    // drop fares <= 0
    if(train_set){
      temp = temp.filter(temp("trip_duration") > 0)
    }
    // drop nas
    temp = temp.na.drop()
    // renaming trip_duration to labels
    if(train_set){
      temp = temp.withColumn("label", temp("trip_duration"))
    }
    // dropping unecessary columns
    val drop_columns = Array("key", "trip_duration", "pickup_datetime", "dow", "dow_index")
    for(drop <- drop_columns){
      temp = temp.drop(drop)
    }
    
    return temp
  }
  // creating engineered training data
  val training_data = create_features(train)
  val test_data = create_features(test, false)

  // vector assembler
  val vec_assembler = new feature.VectorAssembler()
    .setInputCols(Array("pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count", "year", "month", "day", "hour", "minute", "dow_vec"))
    .setOutputCol("features")

  // pipeline
  val pipeline = new Pipeline()
    .setStages(Array(vec_assembler))
  val piped_data = pipeline.fit(training_data).transform(training_data)

  // train test split
  val Array(trainingData, testData) = piped_data.randomSplit(Array(0.8, 0.2))

  // evaluator
  val evaluator = new evaluation.RegressionEvaluator()
    .setMetricName("rmse")

  // specify model
  val dt = new regression.DecisionTreeRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")

  // grid search
  val paramGrid = new tuning.ParamGridBuilder()
    .addGrid(dt.maxDepth, Array(10))
    .build()

  // cross validation
  val cv = new tuning.CrossValidator()
    .setEstimator(dt)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    //.setNumFolds(1)

  // train model
  val models = if (retrain) cv.fit(trainingData) else tuning.CrossValidatorModel.load("models/dt")
  val best_model = models.bestModel

  // evaluate on the test set
  val predictions = best_model.transform(testData)
  val result = evaluator.evaluate(predictions)

  // test predictions
  val test_piped_data = pipeline.fit(test_data).transform(test_data)
  val test_predictions = best_model.transform(test_piped_data) 
  test_predictions.select("id", "prediction").write.parquet("data/predictions")
}