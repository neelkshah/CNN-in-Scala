package pplAssignment
/*
DONT EDIT THIS CODE. ANOTHER DRIVER WITH SIMILAR STRUCTURE WILL BE USED TO CHECK YOUR CODE
ONLY CHANGE <Student_ID> to Your ID
*/
object Driver{

  def main(args:Array[String]){
    val lines = scala.io.Source.fromFile(args(0), "utf-8").getLines.toList

    def getImage(iter:Int,index:Int):List[List[Double]]={
	    if(iter==index){
		    Nil	
	    }else{
		    lines(iter).split(' ').toList.map(_.toDouble)::getImage(iter+1,index)	
	    }
    }

    var curr=0
    val imagesize = lines(curr).split(' ').toList.map(_.toInt)
    var row=imagesize.head
    curr = curr+1
    val image:List[List[Double]]=getImage(curr,curr+row)
    curr = curr+row
    val kernel1size = lines(curr).split(' ').toList.map(_.toInt)
    row=kernel1size.head
    curr = curr+1
    val kernel1=getImage(curr,curr+row)
    curr = curr+row
    val kernel2size = lines(curr).split(' ').toList.map(_.toInt)
    row=kernel2size.head
    curr = curr+1
    val kernel2=getImage(curr,curr+row)
    curr = curr+row
    val kernel3size = lines(curr).split(' ').toList.map(_.toInt)
    row=kernel3size.head
    curr = curr+1
    val kernel3=getImage(curr,curr+row)
    curr = curr+row
    var l=lines(curr).split(' ').toList.map(_.toDouble)
    val w1=l.head
    l=l.tail
    val w2=l.head
    l=l.tail
    val b=l.head
    curr = curr+1
    val nmatsize= lines(curr).split(' ').toList.map(_.toInt)
    row=nmatsize.head
    curr = curr+1
    val nmat=getImage(curr,curr+row)
    curr = curr+row
    val poolmatsize= lines(curr).split(' ').toList.map(_.toInt)
    row=poolmatsize.head
    curr = curr+1
    val poolmat=getImage(curr,curr+row)
    curr = curr+row
    
    var size=lines(curr).split(' ').toList.map(_.toInt).head//Pooling size
    
    def max(xs: List[Double]): Double = {
    if(xs.tail.nonEmpty){
      val tl = max(xs.tail)
      if(tl > xs.head) tl
      else xs.head
    }else{
      xs.head
    }
  }

    //Dot product
    val res1=<STUDENT_ID>.dotProduct(kernel1,kernel2)

    // println(prod)
    //convolution 
    val res2=<STUDENT_ID>.convolute(image,kernel1,imagesize,kernel1size) 

    // println(result)

    //activation 
    val res3=<STUDENT_ID>.activationLayer((x:Double)=>if(x>150) x else 0,image) 

    // println(result)

    //single pooling

    val res4=<STUDENT_ID>.singlePooling(max,poolmat,size)
    //pooling

    val res5=<STUDENT_ID>.poolingLayer(max,image,size) // max is a function

    //Normalize

    val res6=<STUDENT_ID>.normalise(nmat) 

    //mixed layer

    val res7=<STUDENT_ID>.mixedLayer(image,kernel1,imagesize,kernel1size,(x:Double)=>x,max,size) // max is a function

    //assembly layer
    def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0)/1000000000.0 + "s")
    result
}
// THIS COMMENTED SECTION CAN BE REMOVED TO TIME YOUR CODE. TOP LINE AFTER REMOVING THIS WILL BE YOUR TIMED OUTPUT. 
//  COMMENT NEXT LINE WHEN YOU UNCOMMENT THIS
// val res8=time{<STUDENT_ID>.assembly(image,imagesize,w1,w2,b,kernel1,kernel1size,kernel2,kernel2size,kernel3,kernel3size,size)}
    val res8=<STUDENT_ID>.assembly(image,imagesize,w1,w2,b,kernel1,kernel1size,kernel2,kernel2size,kernel3,kernel3size,size)

    println("res1")
    println(res1)
    println("res2")
    println(res2)    
    println("res3")
    println(res3)    
    println("res4")
    println(res4)    
    println("res5")
    println(res5)    
    println("res6")
    println(res6)
    println("res7")
    println(res7)
    println("res8")
    println(res8)
    //}
  }
}
