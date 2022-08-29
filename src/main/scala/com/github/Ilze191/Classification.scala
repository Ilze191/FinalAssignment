package com.github.Ilze191

import com.github.Ilze191.SparkUtil.getSpark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.functions.{col, expr, udf}

object Classification extends App{
  val spark = getSpark("Classification")

  val filePath = "src/resources/stocks/stock_prices_.csv"

  val df = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(filePath)

  println("** ORIGINAL DATAFRAME **")
  df.show(5,false)
  df.printSchema()

  //Add a new column with previous day's close price
  val prevCloseDF = df
    .withColumn("prevClose", expr("" +
      "LAG (close,1,0) " +
      "OVER (PARTITION BY ticker " +
      "ORDER BY date )"))
    .where("prevClose != 0.0")

  println("** ADDED COLUMN WITH PREVIOUS DAY'S CLOSE PRICE **")
  prevCloseDF.show(5, false)

 //Function which will be applied in making a new column
  val priceChange = udf((close: Double, prevClose: Double) => {
    val difference = close - prevClose
    if (difference < 0) "DOWN" else if (difference > 0) "UP" else "UNCHANGED"
  }
  )
  //Add a new column with close price change values as 3 categories
  val categorizedDF = prevCloseDF
    .withColumn("label", priceChange(col("close"), col("prevClose")))
  println("** ADDED LABEL COLUMN WITH CATEGORIZED PRICE CHANGE VALUES **")
  categorizedDF.show(5)

  val assembler = new VectorAssembler()
    .setInputCols(Array("open", "high", "low", "close"))
    .setOutputCol("features")

  val outputDF = assembler.transform(categorizedDF)
    .select("date", "features", "label")
  println("** DATAFRAME WHICH WILL BE USED FOR ML **")
  outputDF.show(5)

  //Indexing categorical label values
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel").fit(outputDF)

  //Split data set into training and test data sets - 70% and 30%
  val Array(train, test) = outputDF.randomSplit(Array(0.7, 0.3))

  //train.show(5, false)
  //test.show(5, false)

  val lr = new LogisticRegression()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("features")

  //Convert indexed labels back to original labels
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labelsArray(0))

  //Chain transformers in a pipeline
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, lr,labelConverter))

  //Train model
  val model = pipeline.fit(train)

  //Run model with test data set to get predictions
  //This will add new columns - rawPrediction, probability and prediction
  val predictionsDF = model.transform(test)
  println("** DATAFRAME WITH PREDICTIONS **")
  predictionsDF.show(10)

  println("** DATA WITH PREDICTED LABEL **")
  //Select rows to display
  predictionsDF.select("features","label", "predictedLabel").show(5)

  //Will compare prediction and label if there are any mismatches
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  //Error is going to be 1.0 - accuracy
  val accuracy = evaluator.evaluate(predictionsDF)
  println(s"Accuracy $accuracy Test Error = ${(1.0 - accuracy)}")


}
