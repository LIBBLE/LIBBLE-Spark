/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
package libble.examples

import org.apache.spark.{SparkConf, SparkContext}

/***
  * here we test the function of loadlibSVMFile and loadlibbleFile, the class  LibContext pass
  */
object LoadFile {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "D:\\Program Files\\hadoop-2.6.0")

    val conf = new SparkConf()
      .setAppName("myTest")
    val sc = new SparkContext(conf)


    import libble.context.implicits._
    val training=sc.loadlibbleFile("sparse.data")
    println(training.count())
    training.saveAslibbleFile("this.data")



  }
}
