/**
  * Created by syh on 2016/12/9.
  */
package libble.utils

import java.nio.ByteBuffer
import java.util.{Random => JavaRandom}

import scala.util.hashing.MurmurHash3

/**
  * This part of code is borrowed from Spark MLlib.
  */
class XORShiftRandom(init: Long) extends JavaRandom(init) {

  def this() = this(System.nanoTime)

  private var seed = XORShiftRandom.hashSeed(init)

  // we need to just override next - this will be called by nextInt, nextDouble,
  // nextGaussian, nextLong, etc.
  override protected def next(bits: Int): Int = {
    var nextSeed = seed ^ (seed << 21)
    nextSeed ^= (nextSeed >>> 35)
    nextSeed ^= (nextSeed << 4)
    seed = nextSeed
    (nextSeed & ((1L << bits) -1)).asInstanceOf[Int]
  }

  override def setSeed(s: Long) {
    seed = XORShiftRandom.hashSeed(s)
  }
}

object XORShiftRandom {
  /** Hash seeds to have 0/1 bits throughout. */
  private def hashSeed(seed: Long): Long = {
    val bytes = ByteBuffer.allocate(java.lang.Long.SIZE).putLong(seed).array()
    MurmurHash3.bytesHash(bytes)
  }
}