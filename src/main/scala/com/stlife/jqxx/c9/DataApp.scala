package com.stlife.jqxx.c9

import org.apache.spark.SparkContext

object DataApp {
  def main(args:Array[String]): Unit ={
    val sc = new SparkContext("local[1]","c9")
    val rdd = sc.wholeTextFiles("data/20news-bydate-train/*")//rec.sport.hockey
    val newsgroups = rdd.map { case (file, text) => file.split("/").takeRight(2).head }
    val countByGroup = newsgroups.map(n => (n, 1)).reduceByKey(_ + _).collect.sortBy(-_._2).mkString("\n")
    println(countByGroup)

    // Tokenizing the text data
    val text = rdd.map { case (file, text) => text }
    val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase))
    println(whiteSpaceSplit.distinct.count)
    // 402978
    // inspect a look at a sample of tokens - note we set the random seed to get the same results each time
    println(whiteSpaceSplit.sample(true, 0.3, 42).take(100).mkString(","))

    // split text on any non-word tokens
    val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))
    println(nonWordSplit.distinct.count)

    // inspect a look at a sample of tokens
    println(nonWordSplit.distinct.sample(true, 0.3, 42).take(100).mkString(","))

    // filter out numbers
    val regex = """[^0-9]*""".r
    val filterNumbers = nonWordSplit.filter(token => regex.pattern.matcher(token).matches)
    println(filterNumbers.distinct.count)
    // 84912
    println(filterNumbers.distinct.sample(true, 0.3, 42).take(100).mkString(","))

    // examine potential stopwords
    val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + _)
    val oreringDesc = Ordering.by[(String, Int), Int](_._2)
    println(tokenCounts.top(20)(oreringDesc).mkString("\n"))

    // filter out stopwords
    val stopwords = Set(
      "the","a","an","of","or","in","for","by","on","but", "is", "not", "with", "as", "was", "if",
      "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to"
    )
    val tokenCountsFilteredStopwords = tokenCounts.filter { case (k, v) => !stopwords.contains(k) }
    println(tokenCountsFilteredStopwords.top(20)(oreringDesc).mkString("\n"))

    // filter out tokens less than 2 characters
    val tokenCountsFilteredSize = tokenCountsFilteredStopwords.filter { case (k, v) => k.size >= 2 }
    println(tokenCountsFilteredSize.top(20)(oreringDesc).mkString("\n"))

    // examine tokens with least occurrence
    val oreringAsc = Ordering.by[(String, Int), Int](-_._2)
    println(tokenCountsFilteredSize.top(20)(oreringAsc).mkString("\n"))

    // filter out rare tokens with total occurence < 2
    val rareTokens = tokenCounts.filter{ case (k, v) => v < 2 }.map { case (k, v) => k }.collect.toSet
    val tokenCountsFilteredAll = tokenCountsFilteredSize.filter { case (k, v) => !rareTokens.contains(k) }
    println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString("\n"))

    def tokenize(line: String): Seq[String] = {
      line.split("""\W+""")
        .map(_.toLowerCase)
        .filter(token => regex.pattern.matcher(token).matches)
        .filterNot(token => stopwords.contains(token))
        .filterNot(token => rareTokens.contains(token))
        .filter(token => token.size >= 2)
        .toSeq
    }

    // check that our tokenizer achieves the same result as all the steps above
    println(text.flatMap(doc => tokenize(doc)).distinct.count)

    // tokenize each document
    val tokens = text.map(doc => tokenize(doc))
    println(tokens.first.take(20))

    // === train TF-IDF model === //

  }

}
