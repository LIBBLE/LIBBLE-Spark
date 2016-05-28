package libble.features

import libble.linalg.implicits.vectorAdOps
import libble.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import scala.beans.BeanProperty


/**
  * With this class, we scal the data to standard normal space in feature-wise.
  * @param centerlized
  * @param scalStd
  */

class Scaller(var centerlized: Boolean = false, var scalStd: Boolean = true) extends Logging with Serializable {
  @BeanProperty var center: Option[Vector] = None
  @BeanProperty var std: Option[Vector] = None

  /**
    * compute center or std of the data.
    * @param data
    */
  def computeFactor(data: RDD[Vector]): Unit = (centerlized, scalStd) match {
    case (true, false) => {
      center = Some(computeCenter(data))

    }
    case (true, true) => {
      center = Some(computeCenter(data))
      std = Some(coputeVariance(data))
    }
    case (false, true) => {
      std = Some(coputeVariance(data))
    }
    case (false, false) => {
      throw new IllegalArgumentException("you need not a scaller!!!")
    }


  }

  private def computeCenter(data: RDD[Vector]): Vector = {
    val n = data.first().size
    val (cum, num) = data.treeAggregate((new DenseVector(n), 0l))(seqOp = (c, v) =>
      (c._1 += v, c._2 + 1),
      combOp = (c1, c2) => (c1._1 += c2._1, c1._2 + c2._2)
    )
    cum /= num
  }

  private def coputeVariance(data: RDD[Vector]): Vector = centerlized match {
    case true => {
      val cen = center.get
      val n = cen.size
      val (total, num) = data.treeAggregate(new DenseVector(n), 0)(seqOp = (c, v) => {
        val temp = v - cen
        temp.bitwisePow(2.0)
        (c._1 += temp, c._2 + 1)
      }, combOp = (c1, c2) => {
        (c1._1 += c2._1, c1._2 + c2._2)

      })
      total /= num
      total.bitwisePow(0.5)
    }
    case false => {
      val n = data.first().size
      val (total, num) = data.treeAggregate(new DenseVector(n), 0)(seqOp = (c, v) => {
        val temp = v.copy
        temp.bitwisePow(2.0)
        (c._1 += temp, c._2 + 1)
      }, combOp = (c1, c2) => {
        (c1._1 += c2._1, c1._2 + c2._2)

      })
      total /= num
      total.bitwisePow(0.5)
    }
  }


  /**
    * Transform the data : RDD[Vector] with the factors.
    * @param data
    * @return
    */
  def transform(data: RDD[Vector]): RDD[Vector] = {
    val panning: (Vector => Vector) = data.first match {
      case dv: DenseVector => panningD
      case sv: SparseVector => panningS
    }

    (centerlized, scalStd) match {
      case (true, false) => {
        if (center != None) {
          data.map(panning)
        } else {
          throw new IllegalAccessError("you should call computeFactor first!!!")
        }
      }
      case (true, true) => {
        if (center != None && std != None) {
          data.map(panning).map(scaling)
        }
        else {
          throw new IllegalAccessError("you should call computeFactor first!!!")
        }
      }
      case (false, true) => {
        if (std != None) {
          data.map(scaling)
        }
        else {
          throw new IllegalAccessError("you should call computeFactor first!!!")
        }
      }
      case (false, false) => {
        throw new IllegalArgumentException("you need not a scaller!!!")
      }
    }
  }

  /**
    *   Transform the data : Vector with the factors
    */
  def transform(data: Vector): Vector = {
    val panning: (Vector => Vector) = data match {
      case sv: SparseVector => panningS
      case dv: DenseVector => panningD
    }

    (centerlized, scalStd) match {
      case (true, false) => {
        if (center != None) {
          panning(data)
        } else {
          throw new IllegalAccessError("you should call computeFactor first!!!")
        }

      }
      case (true, true) => {
        if (center != None && std != None) {
          panning(data)
          scaling(data)
        }
        else {
          throw new IllegalAccessError("you should call computeFactor first!!!")
        }
      }
      case (false, true) => {
        if (std != None) {
          scaling(data)
        }
        else {
          throw new IllegalAccessError("you should call computeFactor first!!!")
        }
      }
      case (false, false) => {
        throw new IllegalArgumentException("you need not a scaller!!!")
      }
    }
  }

  private def panningS(vec: Vector): Vector = {
    vec - center.get
  }

  private def panningD(vec: Vector): Vector = {
    vec -= center.get
  }

  private def scaling(vec: Vector): Vector = {
    val s = std.get
    vec match {
      case de: DenseVector => {
        val eValues = de.values
        var offset = 0
        while (offset < eValues.length) {
          eValues(offset) /= s.apply(offset)
          offset += 1
        }
        de
      }
      case se: SparseVector => {
        val eIndices = se.indices
        val eValues = se.values
        var offset = 0
        while (offset < eValues.length) {
          eValues(offset) /= s.apply(eIndices(offset))
          offset += 1
        }
        se
      }
    }
  }

}
