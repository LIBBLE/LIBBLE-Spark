package libble.examples

import libble.classification.LogisticRegression
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

object testLR {
  def main(args: Array[String]) {

    if (args.length < 1) {
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
    System.setProperty("hadoop.home.dir", "D:\\Program Files\\hadoop-2.6.0")
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf()
      .setAppName("myTest")
    val sc = new SparkContext(conf)





    val stepSize = options.remove("stepSize").map(_.toDouble).getOrElse(1.0)
    val regParam = options.remove("regParam").map(_.toDouble).getOrElse(0.00001)
    val numIter = options.remove("numIters").map(_.toInt).getOrElse(5)
    val elasticF = options.remove("elasticF").map(_.toDouble).getOrElse(0.00001)
    val numPart = options.remove("numPart").map(_.toInt).getOrElse(20)
    val numClasses = options.remove("numClasses").map(_.toInt).getOrElse(2)
    import libble.context.implicits.sc2LibContext
    val training = sc.loadlibbleFile(args(0), numPart)

    val m = new LogisticRegression(stepSize, regParam, elasticF, numIter, numPart).setClassNum(10)

    m.train(training)


  }
}
