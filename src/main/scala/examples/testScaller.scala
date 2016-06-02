/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
package libble.examples

import libble.features.Scaller
import org.apache.spark.{SparkConf, SparkContext}

/**
  * This is the example of using SVD.
  */
object testScaller {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "D:\\Program Files\\hadoop-2.6.0")

    val conf = new SparkConf()
      .setAppName("myTest")
    val sc = new SparkContext(conf)


    import libble.context.implicits.sc2LibContext
    val training = sc.loadLIBBLEFile("sparse.data")

    val scaller = new Scaller(true, true)
    val features = training.map(_.features)
    scaller.computeFactor(features)



    println("center:" + scaller.getCenter.get)
    println("std:" + scaller.getStd.get)


    val result = scaller.transform(features).collect()
    println(result.mkString(", "))


  }

}
