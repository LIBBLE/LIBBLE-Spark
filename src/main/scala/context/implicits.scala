/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */

package libble.context

import libble.linalg.{DenseVector, SparseVector}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Here we define the implicit convert function.
  */
object implicits {
  implicit def sc2LibContext(sc: SparkContext) = new LibContext(sc)

  implicit def RDD2LibRDD(data: RDD[Instance]) = new libbleRDD(data)
}

/**
  * This class includes the methods of load libbleFILE from the file system.
  *
  * @param sc
  */
class LibContext(val sc: SparkContext) {
  /**
    * Load LibSVM file from the File System with default parallelization.
    *
    * @param path
    * @return RDD[Instance]
    * @deprecated replaced by function loadlibbleFile
    */
  def loadLibSVMFile(path: String): RDD[Instance] = {
    loadLibSVMFile(path, sc.defaultMinPartitions)
  }

  /**
    * Load LibSVM file from the File System with given parallelization.
    *
    * @param path
    * @param partsNum
    * @return RDD[Instance]
    * @deprecated replaced by function loadlibbleFile
    */
  def loadLibSVMFile(path: String, partsNum: Int): RDD[Instance] = {
    val lines = sc.textFile(path, partsNum)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
    val terms = lines.filter(_.split(" ").length != 1).map { line =>
      val items = line.split(" ")
      val label = items.head.toDouble
      val term = items.tail.filter(!_.isEmpty).map { item =>
        val temp = item.split(":")
        (temp.head.toInt - 1, temp.last.toDouble)
      }.unzip
      (label, term._1.toArray, term._2.toArray)
    }.cache()
    val d = terms.map(_._2.lastOption.getOrElse(0))
      .reduce(math.max) + 1
    terms.map { term =>
      new Instance(term._1, new SparseVector(term._2, term._3, d))

    }
  }

  /**
    * Load libble file from File System with default parallelization
    * Compatible with LibSVM file.
    *
    * @param path
    * @return RDD[Instance]
    */
  def loadlibbleFile(path: String): RDD[Instance] = {
    loadlibbleFile(path, sc.defaultMinPartitions)
  }

  /**
    * Load libble file from File System with given parallelization.
    * Compatible with LibSVM file.
    *
    * @param path
    * @param partsNum
    * @return RDD[Instance]
    */
  def loadlibbleFile(path: String, partsNum: Int): RDD[Instance] = {
    val lines = sc.textFile(path, partsNum)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
    lines.first().contains(":") match {
      case true => {
        val terms = lines.map { line =>
          val items = line.split(' ')
          val label = items.head.toDouble
          val term = items.tail.filter(_.nonEmpty).map { item =>
            val temp = item.split(':')
            (temp.head.toInt - 1, temp.last.toDouble)
          }.unzip
          (label, term._1.toArray, term._2.toArray)
        }.cache()

        val d = terms.map(_._2.lastOption.getOrElse(0)).reduce(math.max) + 1
        terms.map { term =>
          new Instance(term._1, new SparseVector(term._2, term._3, d))
        }
      }
      case false => {
        lines.map { line =>
          val items = line.split(' ')
          new Instance(items.head.toDouble, new DenseVector(items.drop(1).map(_.toDouble)))
        }
      }
    }

  }


}


/**
  * With this class,we add save data method to the RDD[Instance].
  *
  * @param data
  */
class libbleRDD(val data: RDD[Instance]) {
  /**
    * Save data to File System in LibSVM format.
    *
    * @param path
    * @deprecated
    */
  def saveAsLibSVMFile(path: String): Unit = {
    data.map { term =>
      val line = new StringBuilder(term.label.toString)
      term.features.foreachActive { (i, v) =>
        line ++= s" ${i + 1}:$v"
      }
      line.mkString
    }.saveAsTextFile(path)
  }

  /**
    * Save data to File System in libble format.
    *
    * @param path
    */
  def saveAslibbleFile(path: String): Unit = {
    val first = data.first()
    first.features match {
      case sv: SparseVector => {
        data.map { term =>
          val line = new StringBuilder(term.label.toString)
          term.features.foreachActive { (i, v) =>
            line ++= s" ${i + 1}:$v"
          }
          line.mkString
        }.saveAsTextFile(path)
      }
      case dv: DenseVector => {
        data.map { term =>
          (term.label +: term.features.toArray).mkString(" ")
        }.saveAsTextFile(path)
      }
    }
  }


}


