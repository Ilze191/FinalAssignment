package com.github.Ilze191

import com.github.Ilze191.Utilities.getSpark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.functions.{col, expr, udf}

object Classification extends App{
  val spark = getSpark("StockMarketAnalysis")

  val filePath = "src/resources/stocks/stock_prices_.csv"

  val df = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(filePath)

  df.show(5,false)
  df.printSchema()

  // Add a new column with previous day's close price
  val prevCloseDF = df
    .withColumn("prevClose", expr("" +
      "LAG (close,1,0) " +
      "OVER (PARTITION BY ticker " +
      "ORDER BY date )"))
    .where("prevClose != 0.0")

  prevCloseDF.show(5, false)

 // Function which will be applied in making a new column
  val priceChange = udf((close: Double, prevClose: Double) => {
    val difference = close - prevClose
    if (difference < 0) "DOWN" else if (difference > 0) "UP" else "SAME"
  }
  )
  // Add a new column with categorized close price change
  val categorizedDF = prevCloseDF
    .withColumn("label", priceChange(col("close"), col("prevClose")))

  categorizedDF.show(5)

  val assembler = new VectorAssembler()
    .setInputCols(Array("open", "high", "low", "close"))
    .setOutputCol("features")

  val output = assembler.transform(categorizedDF)
    .select("date", "features", "label")

  output.show(10)

  // Indexing categorical label values
  val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(output)

  // Split data set into training and test data set - 70% and 30%
  val Array(train, test) = output.randomSplit(Array(0.7, 0.3))

  //  train.show(5, false)
  //  test.show(5, false)

  val lr = new LogisticRegression()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("features")

  // Convert indexed labels back to original labels
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labelsArray(0))

  // Chain indexers and LogisticRegression in a Pipeline
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, lr,labelConverter))

  // Train model. This also runs the indexer
  val model = pipeline.fit(train)

  // Run model with test data set to get predictions
  // This will add new columns - rawPrediction, probability and prediction
  val predictions = model.transform(test)

  predictions.show()

  // Select example rows to display
  predictions.select("features","label", "predictedLabel").show(5)
  // Will compare prediction and label if there are any mismatches
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  // Error is going to be 1.0 - accuracy
  val accuracy = evaluator.evaluate(predictions)
  println(s"Accuracy $accuracy Test Error = ${(1.0 - accuracy)}")


}
