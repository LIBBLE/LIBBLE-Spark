package libble.matrixDecomposition

import scala.collection.mutable.ArrayBuffer
import java.util.Calendar

import libble.context.Instance
import libble.linalg.{DenseVector, Vector}
import libble.linalg.implicits._
import libble.dimReduction.GLS_Matrix_Batch

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

/**
 * Created by gao on 2016/5/26.
 */
class SVD(var K: Int,
          var bound: Double,
          var stepSize: Double,
          var iteration: Int,
          var parts: Int,
          var batchSize: Int) extends Logging with Serializable {
  var eigenvalues = new ArrayBuffer[Double]()
  var eigenvectors = new ArrayBuffer[Vector]()

  def train(training: RDD[Instance]): (Array[Double], Array[Vector]) = {

    val st = Calendar.getInstance().getTimeInMillis
    val m = new GLS_Matrix_Batch(stepSize, 0.0, 0.0, iteration, parts, batchSize, K)
    m.setStopBound(bound)
    val model = m.train(training)

    for (k <- 0 to K - 1) {
      val v = model._1(k)
      val lambda = training.map(x => Math.pow(x.features * v, 2)).reduce(_ + _) // standard: count-1
      eigenvalues.append(math.sqrt(lambda))
      eigenvectors.append(v)
    }

    println(s"time to calculate the top ${K} eigen is: " + (Calendar.getInstance().getTimeInMillis - st))
    (eigenvalues.toArray, eigenvectors.toArray)

  }

  def transform(rawData:RDD[Instance], pc:Array[Vector]): RDD[Instance] = {
    val projected = rawData.map{ ins =>
      val arr = new ArrayBuffer[Double]()
      for(k <- pc.indices) {
        arr.append(ins.features * pc(k))
      }
      Instance(ins.label, new DenseVector(arr.toArray))
    }
    projected
  }

}
