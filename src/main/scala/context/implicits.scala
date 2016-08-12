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
package libble.context

import libble.linalg.{DenseVector, SparseVector}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Here we define the implicit convert function.
  */
object implicits {
  implicit def sc2LibContext(sc: SparkContext) = new LibContext(sc)

  implicit def RDD2LIBBLERDD(data: RDD[Instance]) = new LIBBLERDD(data)
}

/**
  * This class includes the methods of loading LIBBLEFILE from the file system.
  *
  * @param sc
  */
class LibContext(val sc: SparkContext) {
  /**
    * Load LibSVM file from the File System with default parallelization.
    *
    * @param path
    * @return RDD[Instance]
    * @deprecated replaced by function loadLibSVMFile
    */
  def loadLibSVMFile(path: String): RDD[Instance] = {
    loadLibSVMFile(path, -1)
  }

  /**
    * Load LibSVM file from the File System with given parallelization.
    *
    * @param path
    * @param partsNum
    * @return RDD[Instance]
    * @deprecated replaced by function loadLibSVMFile
    */
  def loadLibSVMFile(path: String, partsNum: Int): RDD[Instance] = {
    val lines = {
      if (partsNum > 0) sc.textFile(path, partsNum) else sc.textFile(path)
    }.map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
    val terms = lines.filter(_.split(" ").length != 1).map { line =>
      val items = line.split(" ")
      val label = items.head.toDouble
      val term = items.tail.filter(!_.isEmpty).map { item =>
        val temp = item.split(":")
        (temp.head.toInt - 1, temp.last.toDouble)
      }.unzip
      (label, term._1, term._2)
    }.cache()
    val d = terms.map(_._2.lastOption.getOrElse(0))
      .reduce(math.max) + 1
    terms.map { term =>
      new Instance(term._1, new SparseVector(term._2, term._3, d))

    }
  }

  /**
    * Load LIBBLE file from File System with default parallelization
    * Compatible with LibSVM file.
    *
    * @param path
    * @return RDD[Instance]
    */
  def loadLIBBLEFile(path: String): RDD[Instance] = {
    loadLIBBLEFile(path, -1)
  }

  /**
    * Load LIBBLE file from File System with given parallelization.
    * Compatible with LibSVM file.
    *
    * @param path
    * @param partsNum
    * @return RDD[Instance]
    */
  def loadLIBBLEFile(path: String, partsNum: Int): RDD[Instance] = {
    val lines = {
      if (partsNum > 0) sc.textFile(path, partsNum) else sc.textFile(path)
    }.map(_.trim)
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
  * With this class, we add save-data methods to the RDD[Instance].
  *
  * @param data
  */
class LIBBLERDD(val data: RDD[Instance]) {
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
    * Save data to File System in LIBBLE format.
    *
    * @param path
    */
  def saveAsLIBBLEFile(path: String): Unit = {
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


