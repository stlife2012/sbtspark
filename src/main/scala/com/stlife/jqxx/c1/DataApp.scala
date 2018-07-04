package com.stlife.jqxx.c1

import org.apache.spark.SparkContext

object DataApp {
  def main(arr: Array[String]): Unit = {
    val sc = new SparkContext("local[1]", "total")
    val data = sc.textFile("data/UserPurchaseHistory.csv")

    //    计算购买次数
    val buyCount = data.count()
    //    计算有多少个不同用户购买过商品
    val rdd = data.map(line=>line.split(",")).map(line=>(line(0),line(1),line(2)))  //user,product,price
    val buyUserCount = rdd.map({case(user,product,price)=>user}).distinct().count()
    //    计算总收入
    val priceTotal = rdd.map({case(user,product,price)=>price.toDouble}).sum()
    //    计算最畅销的商品
    val bestProduct = rdd.map({
      case(user,product,price)=>(product,1)
    }).reduceByKey((x,y)=>x+y).sortBy(line=>line._2,true).top(1)

    println("购买次数:" + buyCount)
    println("多少个不同用户购买过商品:" + buyUserCount)
    println("总收入:" + priceTotal.formatted("%.2f"))
    println("畅销的商品:" + bestProduct(0)._1 + ",购买次数:" + bestProduct(0)._2)
  }
}
