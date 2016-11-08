package libble.linalg

import libble.linalg.SparseVector
import org.scalatest.FunSuite

/**
  * Created by Aplysia_x on 2016/11/7.
  */
class VectorsOpt extends FunSuite {
  val sparse = new SparseVector(Array(0, 2), Array(1.0, 3.0), 3)
  val dense = new DenseVector(Array(1.0, 2.0, 3.0))
  import libble.linalg.implicits.vectorAdOps

  test("norm1"){
    assert(sparse.norm1()==4)
    assert(dense.norm1()==6)
  }
  test("norm2"){
    assert(sparse.norm2()==math.sqrt(10))
    assert(dense.norm2()==math.sqrt(14))
  }


  test("dot"){
    assert(sparse*dense==10)
  }

  test("plusax"){
    assert(dense.plusax(1.0,sparse).norm1==10)
  }

  test("scal"){
    assert(dense.scal(2.0).norm1()==20)
  }




}
