/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
package libble.examples

import libble.dimReduction.PCA
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable


object testPCA {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    System.setProperty("spark.ui.port", "4042")
    System.setProperty("spark.akka.frameSize", "100")

    val conf = new SparkConf().setAppName("testPCA")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryoserializer.buffer.max", "2000m")
    val sc = new SparkContext(conf)

    if (args.length < 5) {
      System.err.println("Usage: ~ path:String --elasticF=Double --numIters=Int --stepSize=Double --regParam=Double --nuPart=Int  --numClasses=Int")
      System.exit(1)
    }
    val optionsList = args.drop(1).map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }
    val options = mutable.Map(optionsList: _*)

    val stepSize = options.remove("stepSize").map(_.toDouble).getOrElse(0.1)
    val numIters = options.remove("numIters").map(_.toInt).getOrElse(10)
    val numPart = options.remove("numPart").map(_.toInt).getOrElse(2)
    val K = options.remove("k").map(_.toInt).getOrElse(1)
    val bound = options.remove("bound").map(_.toDouble).getOrElse(1e-6)
    val batchSize = options.remove("batchSize").map(_.toInt).getOrElse(100)

    /*
     * Scope PCA
     */
    import libble.context.implicits._
    val training = sc.loadlibbleFile(args(0))

    val mypca = new PCA(K, bound, stepSize, numIters, numPart, batchSize) //matrix, altogether update eigens

    val PCAModel = mypca.train(training)

    val lambda = PCAModel._1
    val v = PCAModel._2
    lambda.foreach(x => print(x + ","))
    val projected = mypca.transform(v)
    projected.collect().foreach(x => println(x.features))
  }

}
