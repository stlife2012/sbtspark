package com.stlife.jqxx.c10


import org.apache.spark.streaming.{Seconds, StreamingContext}

object SimpleStreamingApp {
  def main(arg:Array[String]): Unit ={
    val ssc = new StreamingContext("local[2]","stream app",Seconds(10))
    val stream = ssc.socketTextStream("localhost",9999)
    stream.print()
    ssc.start()
    ssc.awaitTermination()
  }
}
