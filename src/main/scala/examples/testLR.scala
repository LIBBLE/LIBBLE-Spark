/**
  * Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
  * All Rights Reserved.
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License. */
package libble.examples

import libble.classification.LogisticRegression
import libble.generalizedLinear.L1Updater
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/** *
  * Here is the example of using LogisticRegression.
  */
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
    val training = sc.loadLIBBLEFile(args(0), numPart)
    val m = new LogisticRegression(stepSize, regParam, elasticF, numIter, numPart)
    m.train(training)


  }
}
