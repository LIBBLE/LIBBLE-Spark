/**
 * Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
 * All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */
package libble.classification

import libble.generalizedLinear.{L2Updater, LinearScope, LogisticLoss}

/**
  * This class is the model of LogisticRegression with default regularization L2Reg.
  *
  * @param stepSize
  * @param regParam
  * @param factor
  * @param iters
  * @param partsNum
  */
class LogisticRegression(stepSize: Double,
                         regParam: Double,
                         factor: Double,
                         iters: Int,
                         partsNum: Int) extends LinearScope(stepSize, regParam, factor, iters, partsNum) {

  setLossFunc(new LogisticLoss())
  setUpdater(new L2Updater())


  /**
    * Default threshold is 0.5.
    */
  setThreshold(0.5)

  /**
    * Set the classNum
    *
    * @param classNum
    * @return this
    */
  override def setClassNum(classNum: Int): LogisticRegression.this.type ={
    super.setClassNum(classNum)
    setLossFunc(new LogisticLoss(classNum))

  }
}
