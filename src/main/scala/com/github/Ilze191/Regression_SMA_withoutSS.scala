package com.github.Ilze191
import com.github.Ilze191.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, RFormula, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{avg, col, desc, lag}
import org.apache.spark.sql.types.DoubleType
import scala.collection.mutable.ArrayBuffer

object Regression_SMA_withoutSS extends App{
  val spark = getSpark("Trying regression")

  val filePath = "src/resources/stocks/stock_prices_.csv"
  val df = readDataWithView(spark, filePath)

  df.show()
  df.describe().show()
  println("Total number of rows: " + df.count())

  // check for existing nulls in dataframe
  val rowsWithNulls = df
    .filter(col("open").isNull
      && col("high").isNull
      && col("low").isNull
      && col("close").isNull
      && col("volume").isNull
      && col("ticker").isNull)
  println("Total number of nulls: " + rowsWithNulls.count())

  //vectorizing potential features in order to calculate correlation matrix
  val va = new VectorAssembler()
    .setInputCols(Array("open","high", "low", "volume"))
    .setOutputCol("vectorized features")
  val dfVectorized = va.transform(df)

  val corr = Correlation.corr(dfVectorized, "vectorized features").head

  println(s"Pearson correlation matrix:\n $corr")

  //setting frame window function
  var window = Window.partitionBy("ticker").orderBy("date")

  //creating new columns with previous days close prices (b_1 - close price one day ago, b_2 - close price two days ago)

  val df_withPreviousPrices = df
    .select("date", "ticker", "close")
    .withColumn("b_1", lag(col("close"), offset = 1).over(window))
    .withColumn("b_2", lag(col("close"), offset = 2).over(window))
    .withColumn("b_3", lag(col("close"), offset = 3).over(window))
    .withColumn("b_4", lag(col("close"), offset = 4).over(window))
    .withColumn("b_5", lag(col("close"), offset = 5).over(window))
    .withColumn("b_6", lag(col("close"), offset = 6).over(window))
    .withColumn("b_7", lag(col("close"), offset = 7).over(window))
    .withColumn("b_8", lag(col("close"), offset = 8).over(window))
    .withColumn("b_9", lag(col("close"), offset = 9).over(window))
    .withColumn("b_10", lag(col("close"), offset = 10).over(window))
    .na.drop()
  df_withPreviousPrices.show(10)
  println("Total number of rows after dropping nulls for SMA: " + df_withPreviousPrices.count()) // the number of rows after dropping rows with nulls

  //calculating simple moving average for 10 previous days
  var df_SMA = df_withPreviousPrices
    .withColumn("SMA_10",
      (col("b_1") + col("b_2") + col("b_3") +
        col("b_4") + col("b_5") + col("b_6") +
        col("b_7") + col("b_8") + col("b_9") + col("b_10")) / 10)

  //df_SMA.show()

  df_SMA.createOrReplaceTempView("df_SMA_view")


  //indexing ticker
  val tickerIdx = new StringIndexer().setInputCol("ticker").setOutputCol("tickerInd")


  //using One Hot Encoder for categorical value tickerInd
  val ohe = new OneHotEncoder().setInputCol("tickerInd").setOutputCol("ticker_encoded")

  val stages = Array(tickerIdx, ohe)
  val pipeline = new Pipeline().setStages(stages)

  val fittedPipeline = pipeline.fit(df_SMA).transform(df_SMA)

  fittedPipeline.show(false)

  // creating RFormula

  val model_RFormula = new RFormula()
    .setFormula("close ~ SMA_10 + ticker_encoded")

  val fittedRF = model_RFormula.fit(fittedPipeline)
  val preparedDF = fittedRF.transform(fittedPipeline)
  preparedDF.show(10, false)

  val Array(train, test) = preparedDF.randomSplit(Array(0.75, 0.25))

  // building linear regression
  val lr = new LinearRegression()

  println(lr.explainParams())

  val lrModel = lr.fit(train)

  val fittedDF = lrModel.transform(test)

  println()
  println(s"INTERCEPT: ${lrModel.intercept}")
  println(s"COEFFICIENTS: ${lrModel.coefficients.toArray.mkString(",")}")

  // printing some model summary:
  println()
  println("MODEL EVALUATION")
  val trainingSummary = lrModel.summary
  trainingSummary.residuals.show(10)
  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")

  //preparing to check model on test data
  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")

  println()
  println("MODEL EVALUATION ON TEST DATA")
  val rmse = evaluator.setMetricName("rmse").evaluate(fittedDF)
  println(s"RMSE: $rmse")


  val r2 = evaluator.setMetricName("r2").evaluate(fittedDF)
  println(s"r2: $r2")


  // Preparing for predicting price for next day
  //different stocks:

  preparedDF.createOrReplaceTempView("preparedDF_view")

  val stocks = spark.sql(
    """
      |SELECT DISTINCT(ticker), ticker_encoded
      |FROM preparedDF_view
      |""".stripMargin
  )
  //stocks.show()

  val stockArray = stocks.select("ticker").rdd.map(r => r(0)).collect().toList

  //calculating Simple Moving Average for every stock for 10 days (starting from current day)
  val SMA_buffer = new ArrayBuffer[Double]()

  for (i <- stockArray)  {
    val SMA_current = df_SMA
      .orderBy(desc("date"))
      .limit(10)
      .where(s"ticker = '$i'")
      .agg(avg(col("close")))
      .first.getDouble(0)

    SMA_buffer += SMA_current
  }
  val SMA_list = SMA_buffer.toArray

  //calculating predictions for every stock
  val predictions = Array(
    lrModel.predict(Vectors.dense(SMA_buffer(0), 1, 0, 0, 0)),
    lrModel.predict(Vectors.dense(SMA_buffer(1), 0, 1, 0, 0)),
    lrModel.predict(Vectors.dense(SMA_buffer(2), 0, 0, 1, 0)),
    lrModel.predict(Vectors.dense(SMA_buffer(3), 0, 0, 0, 1)),
    lrModel.predict(Vectors.dense(SMA_buffer(4), 0, 0, 0, 0))
  )

  //putting results (SMA and predictions) into stock dataframe

  val rdd1 = spark.sparkContext.parallelize(SMA_list)
  val rdd_new1 = stocks.rdd.zip(rdd1).map(r => Row.fromSeq(r._1.toSeq ++ Seq(r._2)))
  val stocksNew =spark.createDataFrame(rdd_new1, stocks.schema.add("SMA_currentDay", DoubleType))

  val rddPred = spark.sparkContext.parallelize(predictions)
  val rdd_newPred = stocksNew.rdd.zip(rddPred).map(r => Row.fromSeq(r._1.toSeq ++ Seq(r._2)))
  val stocksPredicted = spark.createDataFrame(rdd_newPred, stocksNew.schema.add("Stock price prediction", DoubleType))
  stocksPredicted.show
}
