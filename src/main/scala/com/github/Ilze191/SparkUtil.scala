package com.github.Ilze191

import org.apache.spark.sql.{DataFrame, SparkSession}

object SparkUtil extends App {
  /**
   * Returns a new or an existing Spark session
   * @param appName sets a name for the application
   * @param partitionCount specifies into how many partitions Spark will divide the data
   * @param master sets the Spark master URL to connect to, default "local"
   * @param verbose provides additional details on version session will run
   * @return sparkSession
   */
  def getSpark(appName: String, partitionCount: Int = 1, master : String = "local", verbose : Boolean = true): SparkSession = {
    if (verbose) println(s"$appName with Scala version: ${util.Properties.versionNumberString}")
    val sparkSession = SparkSession.builder().appName(appName).master(master).getOrCreate
    sparkSession.conf.set("spark.sql.shuffle.partitions", partitionCount)
    if (verbose) println(s"Session started on Spark version ${sparkSession.version} with ${partitionCount} partitions")
    sparkSession
  }

  /**
   * @param spark SparkSession
   * @param filePath location of the source file
   * @param source specifies the file format
   * @param viewName creates a local temporary view using the given name
   * @param header specifies if the first file row is a header
   * @param inferSchema specifies whether Spark should infer column types when reading the file
   * @param printSchema specifies whether to print the schema
   * @param cacheOn determines if to cache the data
   * @return DataFrame
   */
  def readDataWithView(spark: SparkSession,
                       filePath: String,
                       source: String = "csv",
                       viewName: String = "dfTable",
                       header: Boolean = true,
                       inferSchema: Boolean = true,
                       printSchema:Boolean = true,
                       cacheOn: Boolean = true): DataFrame = {
    val df = spark.read.format(source)
      .option("header", header.toString)
      .option("inferSchema", inferSchema.toString)
      .load(filePath)
    //so if you pass only white space or nothing to view we will not create it
    //so if viewName is not blank
    if (viewName.nonEmpty){
      df.createOrReplaceTempView(viewName)
      println(s"Created Temporary View for SQL queries called: $viewName")
    }
    if (printSchema) df.printSchema()
    if (cacheOn) df.cache()
    df
  }
}
