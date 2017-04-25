/**
  * Created by syh on 2016/12/9.
  */

package libble.collaborativeFiltering

import libble.linalg.implicits._
import libble.linalg.{DenseVector, Vector}
import libble.utils.{WorkerStore, XORShiftRandom}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.hashing.byteswap64

/**
  * This is an acceleration version of matrix factorization,
  * but it require that numParts equal to the actual number of machines.
  */
class MatrixFactorizationByScope extends Serializable{
  /**
    * initialize the user factors and item factors randomly
    *
    * @param indices user(item) indices
    * @param rank the length of factor
    * @return
    */
  def initialize(indices: Set[Int], rank :Int) : Map[Int, Vector]= {
    val seedGen = new XORShiftRandom()
    val random = new XORShiftRandom(byteswap64(seedGen.nextLong()))
    val vectors = new Array[Vector](indices.size)
    for (i <- vectors.indices) {
      val factors = Array.fill(rank)(random.nextGaussian())
      val v = new DenseVector(factors)
      v /= v.norm2()
      vectors(i) = v
    }
    indices.zip(vectors).toMap
  }

  /**
    * This is an acceleration version of matrix factorization,
    * but it require that numParts equal to the actual number of machines.
    *
    * @param trainSet RDD of ratings
    * @param numIters number of outer loop
    * @param numParts number of workers
    * @param rank length of factor
    * @param lambda_u regularization parameter of users
    * @param lambda_v regularization parameter of items
    * @param stepSize stepsize for update the factors.
    * @return matrix factorization model
    */
  def train (trainSet: RDD[Rating],
             numIters: Int,
             numParts: Int,
             rank: Int,
             lambda_u: Double,
             lambda_v: Double,
             stepSize: Double) : MatrixFactorizationModel = {
    var stepsize = stepSize
    val itemsSize = trainSet.map(r=> (r.index_y, 1)).countByKey()
    val items = itemsSize.keys.toSet
    val numRatings = trainSet.count()
    //random hash the data by row
    val ratingsByRow = trainSet.groupBy(_.index_x)
      .repartition(numParts)
      .values
      .flatMap(i=>i)
      .cache()
    //number of inner iterations is the maximum number of ratings in p workers
    val numInnerIters = ratingsByRow.mapPartitions( i => Iterator.single(i.length)).reduce((a,b)=>math.max(a,b))
    //initialize item factors in master
    var itemFactors = initialize(items, rank)
    //initialize U in p workers
    ratingsByRow.mapPartitionsWithIndex { (index,iter) =>
      val (indices_x, indices_y) = iter.map(r => (r.index_x, r.index_y)).toSet.unzip
      val userFactors = initialize(indices_x,rank)
      MatrixFactorizationByScope.workerstore.put(s"userFactors_$index", userFactors)
      val deltaFactorByOldV = mutable.Map[(Int, Int),Double]()
      MatrixFactorizationByScope.workerstore.put(s"deltaFactorByOldV_$index" , deltaFactorByOldV)
      Iterator.single(0)
    }.count()
    //main loop
    val startTime = System.currentTimeMillis()
    val lossList = new ArrayBuffer[Double]()
    var i = 0
    while (i < numIters) {
      //broadcast V to p workers
      val bc_itemFactors = ratingsByRow.context.broadcast(itemFactors)
      //for each woker i parallelly do
      val fullgradient = ratingsByRow.mapPartitionsWithIndex{ case(index,iter) =>
        val localRatings = iter.toArray
        val numLocalRatings = localRatings.length
        val localV = bc_itemFactors.value
        val localU = MatrixFactorizationByScope.workerstore.get[Map[Int, Vector]](s"userFactors_$index")
        val seedGen = new XORShiftRandom()
        val random = new XORShiftRandom(byteswap64(seedGen.nextLong() ^ index))
        //inner loop
        for(i <- 1 to numInnerIters){
          //randomly select an instance r_h,k from R_i
          val ranRating = localRatings(random.nextInt(numLocalRatings))
          val uh = localU.get(ranRating.index_x).get
          val vj = localV.get(ranRating.index_y).get
          //update uh
          val residual = ranRating.rating - uh.dot(vj)
          uh *= (1- stepsize * lambda_u)
          uh.plusax(stepsize * residual, vj)
        }
        val localRatingsByItem = localRatings.groupBy(r => r.index_y)
        val deltaFactorByOldV = MatrixFactorizationByScope.workerstore.get[mutable.Map[(Int, Int),Double]](s"deltaFactorByOldV_$index")
        val allDeltaSum = localRatingsByItem.map{case (item, ratings) =>
          val deltaSum = new DenseVector(new Array[Double](rank))
          val vj = localV.get(item).get
          var lossByItem = 0.0
          ratings.foreach { r =>
            val uh = localU.get(r.index_x).get
            val minusResidual = uh.dot(vj) - r.rating
            deltaFactorByOldV += ((r.index_x, r.index_y) -> minusResidual)
            deltaSum.plusax(minusResidual, uh)
          }
          item -> deltaSum
        }
        Iterator.single(allDeltaSum)
      }
        .reduce { (a, b) =>
          a ++ b.map{case (k,v) =>
            val ak = a.get(k)
            if(!ak.isEmpty)
              v.plusax(1.0,ak.get)
            k -> v
          }
        }
      //Compute full gradient
      fullgradient.foreach { case (k,v) =>
          v /= itemsSize.get(k).get.toDouble
      }
      //Broadcast full gradient to p workers
      val bc_fullgrad = ratingsByRow.context.broadcast(fullgradient)
      //for each worker i parallelly do
      val (newItemFactors, lossSum) = ratingsByRow.mapPartitionsWithIndex { case (index, iter) =>
        val localRatings = iter.toArray
        val numLocalRatings = localRatings.length
        val localV = bc_itemFactors.value
        val localU = MatrixFactorizationByScope.workerstore.get[Map[Int, Vector]](s"userFactors_$index")
        val seedGen = new XORShiftRandom()
        val random = new XORShiftRandom(byteswap64(seedGen.nextLong() ^ index))
        val deltaFactorByOldV = MatrixFactorizationByScope.workerstore.get[mutable.Map[(Int, Int),Double]](s"deltaFactorByOldV_$index")
        var loss = 0.0
        val fullgra = bc_fullgrad.value
        //inner loop
        for(i <- 1 to numInnerIters) {
          //randomly select an instance r_h,k from R_i
          val ranRating = localRatings(random.nextInt(numLocalRatings))
          val uh = localU.get(ranRating.index_x).get
          val vj = localV.get(ranRating.index_y).get
          //update vj
          val minusResidual =  uh.dot(vj) - ranRating.rating
          val delta = uh.copy
          delta *= (minusResidual - deltaFactorByOldV.get((ranRating.index_x, ranRating.index_y)).get)
          delta.plusax(1.0, fullgra(ranRating.index_y))
          vj *= (1-stepsize*lambda_v)
          vj.plusax(-stepsize, delta)
          //approximate loss
          loss += minusResidual * minusResidual
        }
        Iterator.single((bc_itemFactors.value, loss))
      }
        .reduce { (a, b) =>
          val temp = a._1
          b._1.foreach{case (i, v) =>
            v.plusax(1.0, temp.get(i).get)
          }
          (b._1, a._2 + b._2)
        }
      itemFactors = newItemFactors
      itemFactors.foreach(ui => ui._2 /= numParts.toDouble)
      bc_itemFactors.unpersist()
      //update stepsize
      val approxLoss = lossSum / (numParts * numInnerIters)
      if (i != 0) {
        val oldLoss = lossList.last
        if (approxLoss > oldLoss)
          stepsize = stepsize * 0.5
        else
          stepsize *= 1.05
      }
      lossList.append(approxLoss)
      i += 1
    }
    //training loss
    val trainOver = System.currentTimeMillis()
    val bc_test_itemFactors = ratingsByRow.context.broadcast(itemFactors)
    val loss = ratingsByRow.mapPartitionsWithIndex { (index,iter) =>
      val localV = bc_test_itemFactors.value
      val localU = MatrixFactorizationByScope.workerstore.get[Map[Int, Vector]](s"userFactors_$index")
      val reguV = localV.mapValues(v => lambda_v * v.dot(v))
      val reguU = localU.mapValues(u => lambda_u * u.dot(u))
      val ls = iter.foldLeft(0.0) { (l, r) =>
        val uh = localU.get(r.index_x).get
        val vj = localV.get(r.index_y).get
        val residual = r.rating - uh.dot(vj)
        l + residual * residual + reguU.get(r.index_x).get + reguV.get(r.index_y).get
      }
      Iterator.single(ls)
    }
      .reduce(_ + _) / numRatings
    bc_test_itemFactors.unpersist()
    println(s"loss: $loss\t")
    println(s"cputime of training process(ms): ${ trainOver - startTime }")
    //build model
    val userFactorsRDD = ratingsByRow.mapPartitionsWithIndex{ (index,iter) =>
      val factors = MatrixFactorizationByScope.workerstore.get[Map[Int, Vector]](s"userFactors_$index")
      factors.toIterator
    }.cache()
    val itemFactorsRDD = ratingsByRow.context.parallelize(itemFactors.toSeq, numParts).cache()
    new MatrixFactorizationModel(rank, userFactorsRDD, itemFactorsRDD)
  }
}

object MatrixFactorizationByScope {
  val workerstore = new WorkerStore()
}
