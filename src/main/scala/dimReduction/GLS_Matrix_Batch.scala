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
package libble.dimReduction

import libble.linalg.{DenseVector, Vector}
import libble.context.Instance
import libble.linalg.implicits._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import java.util.Calendar

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD


/**
 *
 * This class is the Generalized Linear Algorithms for PCA model which uses mini-batch strategy during optimization process
 *
 * @param stepSize
 * @param regParam
 * @param factor
 * @param iters
 * @param parts
 * @param batchSize
 * @param K
 */
class GLS_Matrix_Batch (var stepSize: Double,
                        var regParam: Double,
                        var factor: Double,
                        var iters: Int,
                        var parts: Int,
                        var batchSize:Int,
                        var K:Int) extends Logging with Serializable {
  def this() = this(1.0, 0.0001, 0.0001, 5, 2, 1, 1)

  private[this] var stopBound: Double = 0.0
  var weightsVector: Option[Vector] = None

  /**
   *
   * @param value
   * @return this.type
   */
  def setStopBound(value: Double): this.type = {
    stopBound = value
    this
  }

  /**
   * Training the model on training data.
   *
   * @param input
   * @return principle components and loss array
   */
  def train(input: RDD[Instance]): (Array[Vector], Array[Double]) = {
    val dims = input.first().features.size
    val W0 = new Array[Vector](K)
    for(i<-0 to K-1) {
      val arr = new Array[Double](dims)
      for(j<-arr.indices)
        arr(j) = Random.nextGaussian()
      W0(i) = new DenseVector(arr.clone())
      val n = W0(i).norm2()
      W0(i) /= n
    }

    train(input, W0)
  }


  /**
   * Training on training data with initial weights.
   *
   * @param input
   * @param initialWs
   * @return principle components and loss array
   */
  def train(input: RDD[Instance], initialWs: Array[Vector]): (Array[Vector], Array[Double]) = {
    if (parts == (-1)) parts = input.partitions.length
    val data = {
      if (parts == input.partitions.length)
        input.map(e => (e.label, e.features)).cache()
      else
        input.map(e => (e.label, e.features)).coalesce(parts, true).cache()
    }
    runEngine(data, initialWs)
  }


  /**
   * the PCA engine, execute the Algorithm of PCA which use iterative optimization process
   *
   * @param data
   * @param initialWs
   * @return
   */
  private[this] def runEngine(data: RDD[(Double, Vector)], initialWs: Array[Vector]): (Array[Vector], Array[Double]) = {

    val K = initialWs.length
    val count = data.count()
    var weights = new Array[Vector](K)
    for(k<-0 to K-1)
      weights(k) = initialWs(k).copy
    val n = weights(0).size
    var convergenced = false

    val startTime = Calendar.getInstance().getTimeInMillis

    /**
     * outer loop
     */
    val lossArray = ArrayBuffer[Double]()
    var i = 0
    var time = 0l

    while (i < iters && !convergenced) {

      val w = data.context.broadcast(weights)
      var time = Calendar.getInstance().getTimeInMillis
      val temp = new Array[Vector](K)
      for(k<-0 to K-1)
        temp(k) = new DenseVector(n)

      val (mu, lossTotal, diag) = data.treeAggregate(temp, 0.0, new Array[Double](K))(
        seqOp = (c, v) => {
          var lossTemp = 0.0
          for(k<-0 to K-1) {
            val inner = v._2 * w.value(k)
            val loss = -1.0*inner*inner
            c._1(k).plusax(inner, v._2)
            c._3(k) += loss
            lossTemp += loss
          }
          (c._1, c._2 + lossTemp, c._3)
        },
        combOp = (c1, c2) => {
          for(k<-0 to K-1) {
            c2._1(k) += c1._1(k)
            c2._3(k) += c1._3(k)
          }
          (c2._1, c1._2 + c2._2, c2._3)
        }
      )
      for(k<-0 to K-1)
        mu(k) /= count.toDouble

      val loss = lossTotal / count.toDouble
      println(s"$loss ${time - startTime} ")
      for(k<-0 to K-1)
        println(diag(k)/count.toDouble)
      println()
      lossArray += loss


      val temp2 = new Array[Vector](K)
      for(k<-0 to K-1)
        temp2(k) = new DenseVector(n)

      val w_0 = data.context.broadcast(weights)
      val weightsAll = data.mapPartitions({ iter =>
        val omiga = new Array[Vector](K)
        for(k<-0 to K-1)
          omiga(k) = w_0.value(k).copy
        val indexSeq = iter.toIndexedSeq
        val pNum = indexSeq.size

        /**
         * inner loop
         */
        for (j <- 1 to pNum/batchSize) {

          val delta = new Array[Vector](K)
          for(k<-0 to K-1)
            delta(k) = new DenseVector(n)

          for (b <- 1 to batchSize) {
            val e = indexSeq(Random.nextInt(pNum))
            for(k<-0 to K-1) {
              val f1 = e._2 * omiga(k)
              val f2 = e._2 * w_0.value(k)
              delta(k).plusax(f1-f2, e._2)
            }
          }

          for(k<-0 to K-1) {
            delta(k) /= batchSize
            delta(k) += mu(k)
            omiga(k).plusax(stepSize, delta(k))
          }

          GramSchmidt(omiga)
        }
        Iterator(omiga)
      }, true)
        .treeAggregate(temp2)(seqOp = (c, w) => {
        for(k<-0 to K-1)
          c(k) += w(k)
        c
      }, combOp = { (w1, w2) =>
        for(k<-0 to K-1)
          w1(k) += w2(k)
        w1
      })

      for(k<-0 to K-1)
        weightsAll(k) /= parts.toDouble

      GramSchmidt(weightsAll)

      weights = weightsAll

      if(i>=2)
        convergenced = isConvergenced(lossArray)
      i += 1
      time = Calendar.getInstance().getTimeInMillis
    }
    logInfo(s"losses of the last 10 iteration are:${lossArray.takeRight(5).mkString(",")}")

    (weights, lossArray.toArray)

  }

  /**
   *
   * @param lossArray
   * @return Boolean
   */
  private[this] def isConvergenced(lossArray:ArrayBuffer[Double]): Boolean = {
    val len = lossArray.length
    (math.abs(lossArray(len-1) - lossArray(len-2)) < stopBound) && (lossArray(len-1) < lossArray(len-2))
  }

  /**
   *
   * This method is the implementation of GramSchmidt orthonormalization which is invoked in each inner loop
   *
   * @param weights
   */
  def GramSchmidt(weights:Array[Vector]): Unit = {
    val beta = new Array[Vector](K)
    for(k<-0 to K-1) {
      weights(k) /= parts.toDouble
      beta(k) = weights(k).copy
      for(j<-0 to k-1) {
        val xishu = (beta(j) * weights(k)) / (beta(j) * beta(j))
        beta(k).plusax(-1.0*xishu, beta(j))
      }
    }
    for(k<-0 to K-1) {
      val normk = beta(k).norm2()
      beta(k) /= normk
      weights(k) = beta(k).copy
    }
  }

}
