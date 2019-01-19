package <package-name-goes-here>    //Enter package name here

object cnn{
  /** NAME:           SHAH NEEL KAUSHIK
    * ID:             2016A7PS0076P
    * SUBMITTED:      02 NOVEMBER 2018
    * PPL ASSIGNMENT: CNN IN SCALA*/


  /**Private helper function to slice the image matrix. Takes the matrix, starting row and column, and number of rows and columns required.
    * Uses horizontal and vertical helper functions for iterating within and over rows respectively.
    * Used by the convolution and pooling layers*/
  private def makeMatrix(image: List[List[Double]], startRow: Int, startCol: Int, kRow: Int, kCol: Int): List[List[Double]] = {
    def horiStart(list: List[Double], startCol: Int): List[Double] = {
      startCol match{
        case 0 => list
        case _ => horiStart(list.tail, startCol - 1)
      }
    }
    def horiEnd(list: List[Double], kCol: Int): List[Double] = {
      kCol match{
        case 0 => Nil
        case _ => List.concat(List(list.head), horiEnd(list.tail, kCol - 1))
      }
    }
    def vertiStart(matrix: List[List[Double]], startRow: Int): List[List[Double]] = {
      startRow match{
        case 0 => matrix
        case _ => vertiStart(matrix.tail, startRow - 1)
      }
    }
    def vertiEnd(matrix: List[List[Double]], startCol: Int, kRow: Int, kCol: Int): List[List[Double]] = {
      kRow match{
        case 0 => Nil
        case _ => List.concat(List(horiEnd(horiStart(matrix.head, startCol), kCol)), vertiEnd(matrix.tail, startCol, kRow - 1, kCol))
      }
    }
    vertiEnd(vertiStart(image, startRow), startCol, kRow, kCol)
  }

  /**Takes two matrices (of type Double) of the same size and returns their dot product (a single value of type Double)
    * With horizontal and vertical helper functions*/
  def dotProduct(matrix_1: List[List[Double]], matrix_2: List[List[Double]]): Double = {
    def horizontalDP(row1: List[Double], row2: List[Double], numC: Int): Double = {
      numC match {
        case 0 => 0
        case _ => (row1.head * row2.head) + horizontalDP(row1.tail, row2.tail, numC - 1)
      }
    }
    def verticalDP(matrix_1: List[List[Double]], matrix_2: List[List[Double]], numR: Int): Double = {
      numR match {
        case 0 => 0
        case _ => horizontalDP(matrix_1.head, matrix_2.head, matrix_1.head.length) + verticalDP(matrix_1.tail, matrix_2.tail, numR - 1)
      }
    }
    verticalDP(matrix_1, matrix_2, matrix_1.head.length)
  }

  /**Convolution Layer
    * With horizontal and vertical helper functions*/
  def convolute(image: List[List[Double]], kernel: List[List[Double]], imageSize: List[Int], kernelSize: List[Int]): List[List[Double]] = {
    def horizontalConvolute(image: List[List[Double]], kernel: List[List[Double]], imageSize: List[Int], kernelSize: List[Int], row: Int, col: Int): List[Double] = {
      if (col == imageSize.tail.head - kernelSize.tail.head + 1) Nil
      else List.concat(List(dotProduct(makeMatrix(image, row, col, kernelSize.head, kernelSize.tail.head), kernel)), horizontalConvolute(image, kernel, imageSize, kernelSize, row, col + 1))
    }
    def verticalConvolute(image: List[List[Double]], kernel: List[List[Double]], imageSize: List[Int], kernelSize: List[Int], row: Int, col: Int): List[List[Double]] = {
      if (row == imageSize.head - kernelSize.head + 1) Nil
      else List.concat(List(horizontalConvolute(image, kernel, imageSize, kernelSize, row, 0)), verticalConvolute(image, kernel, imageSize, kernelSize, row + 1, col))
    }
    verticalConvolute(image, kernel, imageSize, kernelSize, 0, 0)
  }

  /**Activation Layer
    * With horizontal and vertical helper functions*/
  def activationLayer(activationFunc: Double => Double, image: List[List[Double]]): List[List[Double]] = {
    def horizontalActivation(row: List[Double], fun: Double => Double): List[Double] = {
      if (row.isEmpty) Nil
      else List.concat(List(fun(row.head)), horizontalActivation(row.tail, fun))
    }
    def verticalActivation(matrix: List[List[Double]], fun: Double => Double): List[List[Double]] = {
      if (matrix.isEmpty) Nil
      else List.concat(List(horizontalActivation(matrix.head, fun)), verticalActivation(matrix.tail, fun))
    }
    verticalActivation(image, activationFunc)
  }

  /**Max-pooling function definition*/
  private def maxPool(list: List[Double]): Double = {
    @annotation.tailrec
    def maxPoolHelp(list: List[Double], big: Double): Double = {
      if (list.isEmpty) big
      else {
        if (list.head > big) maxPoolHelp(list.tail, list.head)
        else maxPoolHelp(list.tail, big)
      }
    }
    maxPoolHelp(list, 0)
  }

  /**Average-pooling function definition*/
  private def avgPool(list: List[Double]): Double = {
    list match{
      case Nil => 0
      case _ :: tail => (list.head + avgPool(list.tail)*tail.length)/list.length
    }
  }

  /**Returns a single row of pooled values*/
  def singlePooling(poolingFunc: List[Double] => Double, image: List[List[Double]], k: Int): List[Double] = {
    def flatList(list: List[List[Double]]): List[Double] = {
      if (list.isEmpty) Nil
      else List.concat(list.head, flatList(list.tail))
    }
    def horizontalPool(image: List[List[Double]], row: Int, col: Int, k: Int): List[Double] = {
      if(col >= image.head.length) Nil
      else List.concat(List(poolingFunc(flatList(makeMatrix(image, row, col, k, k)))), horizontalPool(image, row, col + k, k))
    }
    horizontalPool(image, 0, 0, k)
  }

  /**Pooling Layer: Returns a complete pooled matrix*/
  def poolingLayer(poolingFunc: List[Double] => Double, image: List[List[Double]], k: Int): List[List[Double]] = {
    def poolingHelp(poolingFunc: List[Double] => Double, image: List[List[Double]], k: Int): List[List[Double]] = {
      if(image == Nil) Nil
      else List.concat(List(singlePooling(poolingFunc, image, k)), poolingHelp(poolingFunc, makeMatrix(image, k, 0, image.length - k, image.head.length), k))
    }
    poolingHelp(poolingFunc, image, k)
  }

  /**Mixed Layer*/
  def mixedLayer(Image:List[List[Double]], Kernel:List[List[Double]], imageSize:List[Int], kernelSize:List[Int], activationFunc:Double => Double, poolingFunc:List[Double]=>Double, K:Int): List[List[Double]] = {
    poolingLayer(poolingFunc, activationLayer(activationFunc, convolute(Image, Kernel, imageSize, kernelSize)), K)
  }

  /**Normalising Layer
    * With horizontal and vertical helper functions*/
  def normalise(image: List[List[Double]]): List[List[Int]] = {
    def matrixMax(image: List[List[Double]]): Double = {                                                    /**Finds the maximum element in a matrix*/
      @annotation.tailrec
      def listMax(x: List[Double], max: Double): Double = {
        if (x == Nil) max
        else {
          if (x.head > max) listMax(x.tail, x.head)
          else listMax(x.tail, max)
        }
      }
      @annotation.tailrec
      def matrixMaxHelper(x: List[List[Double]], max: Double): Double = {
        if (x == Nil) max
        else {
          if (listMax(x.head, max) > max) matrixMaxHelper(x.tail, listMax(x.head, max))
          else matrixMaxHelper(x.tail, max)
        }
      }
      matrixMaxHelper(image, 0)
    }
    def matrixMin(image: List[List[Double]]): Double = {                                                   /**Finds the minimum element in a matrix*/
      @annotation.tailrec
      def listMin(x: List[Double], min: Double): Double = {
        if(x == Nil) min
        else{
          if(x.head < min) listMin(x.tail, x.head)
          else listMin(x.tail, min)
        }
      }
      @annotation.tailrec
      def matrixMinHelper(x: List[List[Double]], min: Double): Double = {
        if(x == Nil) min
        else{
          if(listMin(x.head, min) < min) matrixMinHelper(x.tail, listMin(x.head, min))
          else matrixMinHelper(x.tail, min)
        }
      }
      matrixMinHelper(image, image.head.head)
    }
    def normaliser(d: Double, max: Double, min: Double): Int = {                                           /**Definition of min-max normalisation (0 to 255)*/
      ((d - min)/(max - min) * 255).floatValue().round
    }
    val big = matrixMax(image)
    val small = matrixMin(image)
    @annotation.tailrec
    def horizontalNormalise(list: List[Double], result: List[Int]): List[Int] = {
      if(list == Nil) result
      else horizontalNormalise(list.tail, List.concat(result, List(normaliser(list.head, big, small))))
    }
    @annotation.tailrec
    def verticalNormalise(image: List[List[Double]], result: List[List[Int]]): List[List[Int]] = {
      if(image == Nil) result
      else verticalNormalise(image.tail, List.concat(result, List(horizontalNormalise(image.head, Nil))))
    }
    verticalNormalise(image, Nil)
  }

  /**ReLu activation function definition*/
  def reLu(x: Double): Double = {
    if(x > 0) x
    else 0
  }

  /**LeakyReLu activation function definition*/
  def leakyReLu(x: Double): Double = {
    if(x > 0) x
    else 0.5 * x
  }

  /**Assembly Layer
    * With horizontal and vertical helper functions*/
  def assembly(Image:List[List[Double]], imageSize:List[Int], w1:Double, w2:Double, b:Double, Kernel1:List[List[Double]], kernelSize1:List[Int], Kernel2:List[List[Double]], kernelSize2:List[Int], Kernel3:List[List[Double]], kernelSize3:List[Int], Size: Int): List[List[Int]] = {
    val temp1: List[List[Double]] = mixedLayer(Image, Kernel1, imageSize, kernelSize1, reLu, avgPool, Size)             /**Temporary_Output_1*/
    val temp2: List[List[Double]] = mixedLayer(Image, Kernel2, imageSize, kernelSize2, reLu, avgPool, Size)             /**Temporary_Output_2*/
    @annotation.tailrec
    def horizontalCombiner(row1: List[Double], row2: List[Double], w1: Double, w2: Double, bias: Double, result: List[Double]): List[Double] = {
      if(row1.isEmpty) result
      else horizontalCombiner(row1.tail, row2.tail, w1, w2, bias, List.concat(result, List((row1.head*w1) + (row2.head*w2) + bias)))
    }
    @annotation.tailrec
    def verticalCombiner(matrix1: List[List[Double]], matrix2: List[List[Double]], w1: Double, w2: Double, bias: Double, result: List[List[Double]]): List[List[Double]] = {
      if(matrix1.isEmpty) result
      else verticalCombiner(matrix1.tail, matrix2.tail, w1, w2, bias, List.concat(result, List(horizontalCombiner(matrix1.head, matrix2.head, w1, w2, bias, Nil))))
    }
    val temp3 = verticalCombiner(temp1, temp2, w1, w2, b, Nil)                                                          /**Temporary_Output_3*/
    normalise(mixedLayer(temp3, Kernel3, List(temp3.length, temp3.head.length), kernelSize3, leakyReLu, maxPool, Size)) /**Final Output!*/
  }
}
