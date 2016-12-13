
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

package libble.clustering

import java.util

import libble.linalg.implicits.vectorAdOps
import libble.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD

/**
  * KMeans Algorithm.
  */
class KMeans(
              private var k: Int,
              private var maxIters: Int,
              private var stopBound: Double = 0) extends Serializable {

  @transient
  private var initCenters: Option[Array[(Vector, Double)]] = None

  def this(k: Int, stopBound: Double) = this(k, 100, stopBound)

  /**
    * set the number of clusters
    *
    * @param k
    * @return this
    */
  def setK(k: Int): this.type = {
    this.k = k
    this
  }

  /**
    * set the Max Iter
    *
    * @param maxIters
    * @return this
    */
  def setMaxIters(maxIters: Int): this.type = {
    this.maxIters = maxIters
    this
  }

  /**
    * set the convergence bound
    *
    * @param stopBound
    * @return this
    */
  def setStopBound(stopBound: Double): this.type = {
    this.stopBound = stopBound
    this
  }

  /**
    * set the init Centers
    *
    * @param initCenters
    * @return this
    */
  def setInitCenters(initCenters: Array[(Vector, Double)]): this.type = {
    require(initCenters.length == k)
    this.initCenters = Some(initCenters)
    this
  }

  /**
    * Do K-Means train
    *
    * @param data
    * @tparam T
    * @return (KMeansModel,cost)
    */
  def train[T](data: RDD[(T, Vector)]): (KMeansModel, Double) = {
    val centers = initCenters.getOrElse(initCenter(data))

    val trainData = data.map(e => (e._2, e._2.norm2)).cache()
    val squareStopBound = stopBound * stopBound

    var isConvergence = false
    var i = 0
    val costs = data.sparkContext.doubleAccumulator

    while (!isConvergence && i < maxIters) {
      costs.reset()
      val br_centers = data.sparkContext.broadcast(centers)

      val res = trainData.mapPartitions { iter =>
        val counts = new Array[Int](k)
        util.Arrays.fill(counts, 0)
        val partSum = (0 until k).map(e => new DenseVector(br_centers.value(0)._1.size))

        iter.foreach { e =>
          val (index, cost) = KMeans.findNearest(e, br_centers.value)
          costs.add(cost)
          counts(index) += 1
          partSum(index) += e._1
        }
        counts.indices.filter(j => counts(j) > 0).map(j => (j -> (partSum(j), counts(j)))).iterator
      }.reduceByKey { case ((s1, c1), (s2, c2)) =>
        (s1 += s2, c1 + c2)
      }.collectAsMap()
      br_centers.unpersist(false)


      println(s"cost at iter: $i is: ${costs.value}")
      isConvergence = true
      res.foreach { case (index, (sum, count)) =>
        sum /= count
        val sumNorm2 = sum.norm2()
        val squareDist = math.pow(centers(index)._2, 2.0) + math.pow(sumNorm2, 2.0) - 2 * (centers(index)._1 * sum)
        if (squareDist >= squareStopBound) {
          isConvergence = false
        }
        centers(index) = (sum, sumNorm2)
      }
      i += 1
    }
    (new KMeansModel(centers), costs.value)
  }


  private def initCenter[T](data: RDD[(T, Vector)]): Array[(Vector, Double)] = {
    data.takeSample(false, k, System.currentTimeMillis())
      .map(_._2).distinct.map(e => (e, e.norm2))
  }

  override def equals(other: Any): Boolean = other match {
    case that: KMeans =>
      (that canEqual this) &&
        initCenters == that.initCenters &&
        k == that.k &&
        maxIters == that.maxIters &&
        stopBound == that.stopBound
    case _ => false
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[KMeans]

  override def hashCode(): Int = {
    val state = Seq(initCenters, k, maxIters, stopBound)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object KMeans {
  def findNearest(e: (Vector, Double), centers: Array[(Vector, Double)]): (Int, Double) = {
    var cost = Double.MaxValue
    var index = 0;
    for (i <- 0 until centers.length) {
      val center = centers(i)
      if (math.pow(e._2 - center._2, 2.0) < cost) {
        val squarePart = math.pow(e._2, 2.0) + math.pow(center._2, 2.0)
        val squareDist = squarePart - 2 * (e._1 * center._1)
        if (squareDist < cost) {
          cost = squareDist
          index = i
        }
      }
    }
    (index, cost)
  }
}


class KMeansModel(centers: Array[(Vector, Double)]) extends Serializable {

  def clustering[T](data: RDD[(T, Vector)]): RDD[(T, Int)] = {
    val br_center = data.sparkContext.broadcast(centers)
    data.map { e =>
      val res = KMeans.findNearest((e._2, e._2.norm2), br_center.value)
      (e._1, res._1)
    }

  }


}





