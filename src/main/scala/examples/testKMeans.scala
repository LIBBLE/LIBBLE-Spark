/*
 * Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
 * All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package libble.examples

import libble.clustering.KMeans
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/**
  * Created by Aplysia_x on 2016/12/9.
  */
object testKMeans {
  def main(args: Array[String]) {

    if (args.length < 1) {
      System.err.println("Usage: ~ path:String --k=Int --maxIters=Int --stopBound=Double")
      System.exit(1)
    }
    //    System.setProperty("hadoop.home.dir", "D:\\Program Files\\hadoop-2.6.0")

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
      .setAppName("My Test Kmeans")
    val sc = new SparkContext(conf)

    val k = options.remove("k").map(_.toInt).getOrElse(10)
    val maxIters = options.remove("maxIters").map(_.toInt).getOrElse(10)
    val stopBound = options.remove("stopBound").map(_.toDouble).getOrElse(0.0001)

    import libble.context.implicits.sc2LibContext
    val training = sc.loadLIBBLEFile(args(0))
    val m = new KMeans(k, maxIters, stopBound)
    val data = training.map(e => (e.label, e.features))
    m.train(data)
  }
}