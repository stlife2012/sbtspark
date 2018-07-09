package com.stlife.jqxx.c10
import java.io.PrintWriter
import java.net.ServerSocket
import java.text.{SimpleDateFormat, DateFormat}
import java.util.Date

import org.apache.spark.SparkContext._
import org.apache.spark.streaming.{Seconds, StreamingContext}

import scala.util.Random

object StreamingProducer {

  def main(args: Array[String]) {

    val random = new Random()

    // Maximum number of events per second
    val MaxEvents = 6

    // Read the list of possible names
//    val namesResource = this.getClass.getResourceAsStream("/00_src/spark/sbtspark/out/names.csv")
//    val names = scala.io.Source.fromInputStream(namesResource)
//      .getLines()
//      .toList
//      .head
//      .split(",")
//      .toSeq

    val str_name = "Miguel,Eric,James,Juan,Shawn,James,Doug,Gary,Frank,Janet,Michael,James,Malinda,Mike,Elaine,Kevin,Janet,Richard,Saul,Manuela"
    val names = str_name.split(",").flatMap(line=>line.split(",")).toSeq

    // Generate a sequence of possible products
    val products = Seq(
      "iPhone Cover" -> 9.99,
      "Headphones" -> 5.49,
      "Samsung Galaxy Cover" -> 8.95,
      "iPad Cover" -> 7.49
    )

    /** Generate a number of random product events */
    def generateProductEvents(n: Int) = {
      (1 to n).map { i =>
        val (product, price) = products(random.nextInt(products.size))
        val user = random.shuffle(names).head
        (user, product, price)
      }
    }

    // create a network producer
    val listener = new ServerSocket(9999)
    println("Listening on port: 9999")

    while (true) {
      val socket = listener.accept()
      new Thread() {
        override def run = {
          println("Got client connected from: " + socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream(), true)

          while (true) {
            Thread.sleep(1000)
            val num = random.nextInt(MaxEvents)
            val productEvents = generateProductEvents(num)
            productEvents.foreach{ event =>
              out.write(event.productIterator.mkString(","))
              out.write("\n")
            }
            out.flush()
            println(s"Created $num events...")
          }
          socket.close()
        }
      }.start()
    }
  }
}
