package libble.examples

import libble.features.Scaller
import org.apache.spark.{SparkConf, SparkContext}

object testScaller {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "D:\\Program Files\\hadoop-2.6.0")

    val conf = new SparkConf()
      .setAppName("myTest")
    val sc = new SparkContext(conf)


    import libble.context.implicits.sc2LibContext
    val training=sc.loadlibbleFile("sparse.data")

    val scaller=new Scaller(true,true)
    val features= training.map(_.features)
    scaller.computeFactor(features)



    println("center:"+scaller.getCenter.get)
    println("std:"+scaller.getStd.get)


    val result=scaller.transform(features).collect()
    println(result.mkString(", "))





  }

}
