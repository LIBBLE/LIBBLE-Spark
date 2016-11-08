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
package libble.generalizedLinear

import java.util

import libble.linalg.{DenseVector, Vector}
import libble.linalg.implicits._
import org.apache.spark.rdd.RDD

import scala.math._
import scala.util.Random

/**
  *
  */
abstract class Updater extends Serializable {


  def update(data: RDD[(Double, Vector)], weights: Vector, mu: Vector, lossfunc: LossFunc, stepSize: Double, factor: Double, regParam: Double): Vector

  /**
    * In this method, we give the cost of the regularizer.
    *
    * @param weight
    * @param regParam
    * @return regCost
    */
  def getRegVal(weight: Vector, regParam: Double): Double


  protected def findChkSize(prefac: Double): Double = {
    var i = 1.0
    while (pow(prefac, i) >= 1e-8) {
      i *= 10
    }
    i / 10.0
  }

}

/**
  *
  */
class SimpleUpdater extends Updater {


  /**
    * In this method, we give the cost of the regularizer
    *
    * @param weight
    * @param regParam
    * @return regCost
    */
  override def getRegVal(weight: Vector, regParam: Double): Double = {
    0.0
  }

  override def update(data: RDD[(Double, Vector)], weights: Vector, mu: Vector, lossfunc: LossFunc, stepSize: Double, factor: Double, regParam: Double): Vector = {

    val preFact = 1.0 - stepSize * factor
    val upFact = -stepSize / preFact
    mu.plusax(-factor, weights)
    val w_0 = data.sparkContext.broadcast(weights)
    val fix = data.sparkContext.broadcast(mu)
    val partsNum = data.partitions.length
    val chkSize = findChkSize(preFact)

    data.mapPartitions(iter => {
      val omiga = new WeightsVector(w_0.value.copy, fix.value)
      val indexedSeq = iter.toIndexedSeq
      val pNum = indexedSeq.size

      val rand = new Random(partsNum * pNum)

      for (j <- 1 to pNum) {
        val e = indexedSeq(rand.nextInt(pNum))
        val f1 = lossfunc.deltaF(e._2, e._1, omiga)
        f1 -= lossfunc.deltaF(e._2, e._1, w_0.value)
        //          val delta = f1 x e._2
        //          delta += mu
        if (j % chkSize == 0)
          omiga.merge()

        omiga.partA.plusax(upFact / omiga.fac_a, f1 x e._2)
        omiga.fac_a *= preFact
        omiga.fac_b *= preFact
        omiga.fac_b -= stepSize

      }
      Iterator(omiga.toDense())

    }, true).treeAggregate(new DenseVector(weights.size))(seqOp = (c, w) => {
      c += w
    }, combOp = (c1, c2) => {
      c1 += c2
    }) /= (partsNum)

  }
}

/**
  *
  */
class L1Updater extends Updater {


  /**
    * In this method, we give the cost of the regularizer
    *
    * @param weight
    * @param regParam
    * @return regCost
    */
  override def getRegVal(weight: Vector, regParam: Double): Double = {
    weight.norm1 * regParam
  }

  override def update(data: RDD[(Double, Vector)], weights: Vector, mu: Vector, lossfunc: LossFunc, stepSize: Double, factor: Double, regParam: Double): Vector = {

    val preFact = 1.0 - stepSize * (regParam + factor)

    val upFact = -stepSize / preFact

    mu.plusax(-factor, weights)
    val w_0 = data.sparkContext.broadcast(weights)
    val fix = data.sparkContext.broadcast(mu)
    val partsNum = data.partitions.length
    val chkSize = findChkSize(preFact)

    data.mapPartitions(iter => {
      val omiga = new WeightsVector(w_0.value.copy, fix.value)
      val indexedSeq = iter.toIndexedSeq
      val pNum = indexedSeq.size

      val rand = new Random(partsNum * pNum)

      val flags = new Array[Int](omiga.size)
      util.Arrays.fill(flags, 0)

      for (j <- 1 to pNum) {
        val e = indexedSeq(rand.nextInt(pNum))
        val f1 = lossfunc.deltaF(e._2, e._1, omiga)
        f1 -= lossfunc.deltaF(e._2, e._1, w_0.value)
        //          val delta = f1 x e._2
        //          delta += mu

        if (j % chkSize == 0)
          omiga.merge()

        val oValues = omiga.partA.toArray
        e._2.foreachActive { (i, v) =>
          val wi = omiga.apply(i)
          oValues(i) = (math.signum(wi) * max(0.0, abs(wi) - (j - 1 - flags(i)) * stepSize * regParam) - omiga.fac_b * omiga.partB(i)) / omiga.fac_a
          flags(i) = j - 1
        }


        omiga.partA.plusax(upFact / omiga.fac_a, f1 x e._2)
        omiga.fac_a *= preFact
        omiga.fac_b *= preFact
        omiga.fac_b -= stepSize

      }
      Iterator(omiga.toDense())

    }, true).treeAggregate(new DenseVector(weights.size))(seqOp = (c, w) => {
      c += w
    }, combOp = (c1, c2) => {
      c1 += c2
    }) /= (partsNum)

  }
}

/**
  *
  */
class L2Updater extends Updater {


  /**
    * In this method, we give the cost of the regularizer.
    *
    * @param weight
    * @param regParam
    * @return regCost
    */
  override def getRegVal(weight: Vector, regParam: Double): Double = {
    val norm = weight.norm2
    0.5 * regParam * norm * norm
  }

  override def update(data: RDD[(Double, Vector)], weights: Vector, mu: Vector, lossfunc: LossFunc, stepSize: Double, factor: Double, regParam: Double): Vector = {

    val preFact = 1.0 - stepSize * (regParam + factor)
    val upFact = -stepSize / preFact
    mu.plusax(-factor, weights)
    val w_0 = data.sparkContext.broadcast(weights)
    val fix = data.sparkContext.broadcast(mu)
    val partsNum = data.partitions.length
    val chkSize = findChkSize(preFact)

    data.mapPartitions(iter => {
      val omiga = new WeightsVector(w_0.value.copy, fix.value)
      val indexedSeq = iter.toIndexedSeq
      val pNum = indexedSeq.size

      val rand = new Random(partsNum * pNum)

      for (j <- 1 to pNum) {
        val e = indexedSeq(rand.nextInt(pNum))
        val f1 = lossfunc.deltaF(e._2, e._1, omiga)
        f1 -= lossfunc.deltaF(e._2, e._1, w_0.value)
        //          val delta = f1 x e._2
        //          delta += mu
        if (j % chkSize == 0)
          omiga.merge()

        omiga.partA.plusax(upFact / omiga.fac_a, f1 x e._2)
        omiga.fac_a *= preFact
        omiga.fac_b *= preFact
        omiga.fac_b -= stepSize

      }
      Iterator(omiga.toDense())

    }, true).treeAggregate(new DenseVector(weights.size))(seqOp = (c, w) => {
      c += w
    }, combOp = (c1, c2) => {
      c1 += c2
    }) /= (partsNum)

  }
}

