/**
Copyright 2016 LAMDA-09. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
package libble.examples

import libble.dimReduction.PCA
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable


object testPCA {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    System.setProperty("spark.ui.port", "4042")
    System.setProperty("spark.akka.frameSize", "100")

    val conf = new SparkConf().setAppName("testSVD")
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
    val training = sc.loadLIBBLEFile(args(0))

    val mypca = new PCA(K, bound, stepSize, numIters, numPart, batchSize)       //matrix, altogether update eigens
    val PCAModel = mypca.train(training)

    val pc = PCAModel._2
    val projected = mypca.transform(training, pc)
    projected.collect().foreach(x=>println(x.features))

  }
}
