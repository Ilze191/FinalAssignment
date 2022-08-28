package com.github.Ilze191

import org.apache.spark.sql.SparkSession

object Utilities extends App {
  /**
   * Returns a new or an existing Spark session
   * @param appName - sets a name for the application
   * @param partitionCount - default 5 - starting default is 200
   * @param master - sets the Spark master URL to connect to, default "local"
   * @param verbose - provides additional details on version session will run
   * @return sparkSession
   */
  def getSpark(appName:String, partitionCount:Int = 1,
               master:String = "local",
               verbose:Boolean = true): SparkSession = {
    if (verbose) println(s"$appName with Scala version: ${util.Properties.versionNumberString}")
    val sparkSession = SparkSession.builder().appName(appName).master(master).getOrCreate()
    sparkSession.conf.set("spark.sql.shuffle.partitions", partitionCount)
    if (verbose) println(s"Session started on Spark version ${sparkSession.version} with ${partitionCount} partitions")
    sparkSession
  }
}
