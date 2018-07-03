package com.stlife

import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object LineReg {
  def main(args: Array[String]){
    val input_path = "F:\\02_data\\uci\\aps\\train.csv"
    val sqlContext = SparkSession.builder()
      .appName("LingReg").master("local[1]")
      .getOrCreate()
    val data = sqlContext.read.csv(input_path)
//    var table = data.createOrReplaceTempView("train")
//    data.printSchema()
//    println(data.schema)
//    data.columns.foreach(println)
//    data.write.text("D:\\data\\table1.txt")
//    sqlContext.sql("select * from train").show()
//    val topData = data.take(10)
//    topData.foreach(println)
//    csv_data()
//    var rdd_data = data.map(row =>create_label_point(row.toString()))
//    val topData = data.take(10)
//    data.saveAsTextFile("train.txt")
//    data.write
//      .save("D:\\data\\table1.txt")
////    data.select()
    val rdd = data.rdd
    rdd.map(row=>row.mkString())

//    var rdd = data.map(row=>row.mkString(","))
    val len = data.columns.length
//    val lable = rdd.map(row => create_lp(row,len))

//    lable.take(10).foreach(println)
  }

  def create_lp(line:Row,len:Int):LabeledPoint = {
//    val len:Int = data.columns.size
    var dt = new Array[Double](len-1)
    for(col <- 1 to len){
      var cvalue = line.getString(col-1)
      print("value:" + cvalue)
      if("na".equalsIgnoreCase(cvalue)){
        dt(col-1) = 0
      }else{
        try{
          dt(col-1) = line.getDouble(col-1)
        }catch {
          case e:Exception=>{
            dt(col-1) = 0
          }
        }
      }
    }

    val vector = Vectors.dense(dt)
    //返回标签向量
    var clsValue:Double = 1
    if("neg" == line.get(0)){
      clsValue = 0
    }
    LabeledPoint(clsValue,vector)
  }

  def csv_data(): Unit ={
    val input_path = "F:\\02_data\\uci\\aps\\train.csv"
    val sc = new SparkContext(new SparkConf().setMaster("local[1]").setAppName("LineReg"))
    val data = sc.textFile(input_path)
//    data.top(10).foreach(println)

    val tf = new HashingTF(100)
//    val topdata = data.map(row => tf.transform(row.split(","))).take(19)
    val topdata = data.map(row => row.split(",")).take(19)
    topdata.foreach(println)
  }

  def create_label_point(line:String):LabeledPoint = {
    //字符串去空格，以逗号分隔转为数组
    val linearr = line.trim().split(",")
    val linedoublearr = linearr.map(x=>x.toDouble)
    //定长数组转可变数组
    val linearrbuff = linedoublearr.toBuffer
    //移除label元素（将linedoublearr的第一个元素作为标签）
    linearrbuff.remove(0)
    //将剩下的元素转为向量
    val vectorarr = linearrbuff.toArray
    val vector = Vectors.dense(vectorarr)
    //返回标签向量
    LabeledPoint(linedoublearr(0),vector)
  }
}
