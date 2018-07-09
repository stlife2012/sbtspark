name := "sbtspark"
version := "0.1"
scalaVersion := "2.11.12"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0"
)
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.2.0"
libraryDependencies += "org.scalanlp" %% "breeze" % "0.13.2"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.2.0"

//指定主函数
//mainClassinCompile := Some("com.stlife.Hello")

//libraryDependencies += "org.apache.spark" % "spark-streaming-kafka-0-8_2.11" % "2.2.0"
//
//libraryDependencies += "org.apache.spark" % "spark-streaming-flume_2.11" % "2.2.0"
//
//libraryDependencies += "org.apache.spark" % "spark-hive_2.11" % "2.2.0" % "provided"
//
//libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.11"
//
//libraryDependencies += "org.scalanlp" % "breeze-natives_2.11" % "0.11"
