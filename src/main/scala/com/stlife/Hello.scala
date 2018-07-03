package com.stlife

import org.apache.spark.{SparkConf, SparkContext}

import scala.math.Ordering

object Hello {
  def main(args: Array[String]) {
    var masterUrl = "local" //[1]
//    var inputPath = "hdfs://192.168.99.100:8082/data/root.txt" //hdfs://192.168.99.100:8082 D:\data\data.txt
    var inputPath = "D:\\data\\data.txt"
    var outputPath = "D:\\data\\output\\6"

    println(s"masterUrl:${masterUrl}, inputPath: ${inputPath}, outputPath: ${outputPath}")

    val sparkConf = new SparkConf().setMaster(masterUrl).setAppName("WordCount")
    val sc = new SparkContext(sparkConf)
    val acc = sc.longAccumulator("acc")
    var data:Array[String] = Array("one","two")
    var bdata = sc.broadcast(data)

    val rowRdd = sc.textFile(inputPath).repartition(2)
    val resultRdd = rowRdd.flatMap(line => {
      acc.add(1)
      for(key <- bdata.value){
        if(line.toString().equals(key)){
          acc.add(1)
        }
      }
      line.split("\\s")
    }).map(word => (word, 1)).reduceByKey(_ + _)
//    resultRdd.foreach(f=>println(f))
    val scollect = resultRdd.sortBy(f=>f._2,false).collect()
    scollect.take(10).foreach(println)
//    resultRdd.foreach(f=>{
//      if(f._2 > 20){
//        println(f)
//      }
//    })
    println("acc:" + acc.value)
//    resultRdd.saveAsTextFile(outputPath)
    //    resultRdd.saveAsSequenceFile(outputPath)
  }
}
