/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
package libble.examples

import libble.collaborativeFiltering.{MatrixFactorization, Rating}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/***
  * Here is the example of using Matrix Factorization.
  */
object testMF {
  def main(args: Array[String]) {

    if (args.length < 1) {
      System.err.println("Usage: ~ path:String --numIters=Int --numParts=Int --rank=Int --regParam_u=Double --regParam_v=Double --stepsize=Double")
      System.exit(1)
    }

    val optionsList = args.drop(1).map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }
    val options = mutable.Map(optionsList: _*)
    System.setProperty("hadoop.home.dir", "D:\\Program Files\\hadoop-2.6.0")
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf()
      .setAppName("testMF")
    val sc = new SparkContext(conf)


    val stepsize = options.remove("stepsize").map(_.toDouble).getOrElse(0.01)
    val regParam_u = options.remove("regParam_u").map(_.toDouble).getOrElse(0.05)
    val regParam_v = options.remove("regParam_u").map(_.toDouble).getOrElse(0.05)
    val numIters = options.remove("numIters").map(_.toInt).getOrElse(200)
    val numParts = options.remove("numParts").map(_.toInt).getOrElse(16)
    val rank = options.remove("rank").map(_.toInt).getOrElse(40)

    val trainSet = sc.textFile(args(0), numParts)
      .map(_.split(',') match { case Array(user, item, rate) =>
        Rating(rate.toDouble, user.toInt, item.toInt)
      })

    val model = new MatrixFactorization()
      .train(trainSet,
        numIters,
        numParts,
        rank,
        regParam_u,
        regParam_v,
        stepsize)
  }
}
