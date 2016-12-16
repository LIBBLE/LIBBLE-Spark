/*
 *
 *  Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
 *  All Rights Reserved.
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  You may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
package libble.examples

import libble.collaborativeFiltering.{MatrixFactorizationByScope, MatrixFactorization, Rating}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable


/***
  * Here is the example of using Matrix Factorization.
  */
object testCF {
  def main(args: Array[String]) {
    val optionsList = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }
    val options = mutable.Map(optionsList: _*)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf()
      .setAppName("testMF")
    val sc = new SparkContext(conf)

    val trainsetPath = options.remove("trainset").map(_.toString).getOrElse("data\\testMF.txt")
    val stepsize = options.remove("stepsize").map(_.toDouble).getOrElse(0.1)
    val regParam_u = options.remove("regParam_u").map(_.toDouble).getOrElse(0.1)
    val regParam_v = options.remove("regParam_u").map(_.toDouble).getOrElse(0.1)
    val numIters = options.remove("numIters").map(_.toInt).getOrElse(50)
    val numParts = options.remove("numParts").map(_.toInt).getOrElse(2)
    val rank = options.remove("rank").map(_.toInt).getOrElse(10)
    val testsetPath = options.remove("testset").map(_.toString).getOrElse("data\\testMF.txt")

    val trainSet = sc.textFile(trainsetPath, numParts)
      .map(_.split(',') match { case Array(user, item, rate) =>
        Rating(rate.toDouble, user.toInt, item.toInt)
      })
    val testSet = sc.textFile(testsetPath, numParts)
      .map(_.split(',') match { case Array(user, item, rate) =>
        Rating(rate.toDouble, user.toInt, item.toInt)
      })

    val model = new MatrixFactorizationByScope()
      .train(trainSet,
        numIters,
        numParts,
        rank,
        regParam_u,
        regParam_v,
        stepsize)

    val result = model.predict(testSet.map(r=>(r.index_x,r.index_y)))
    val rmse = result.map(r=>((r.index_x,r.index_y), r.rating))
      .join(testSet.map(r=>((r.index_x,r.index_y), r.rating)))
      .values
      .map(i => math.pow(i._1 - i._2, 2))
      .sum() / testSet.count()
    println(s"rmse of test set: $rmse")
  }
}
