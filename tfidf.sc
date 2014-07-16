// run this in spark shell

// load & process df 
val dfs = sc.textFile("/tmp/token_df")
val dfs_2 = dfs.map(_.stripPrefix("(").stripSuffix(")"))
val dfs_3 = dfs_2.map(_.split(","))
val dfs_4 = dfs_3.map(x => (x(1), x(0)))
dfs_4.cache()
 
//process tf
val docs = sc.textFile("/tmp/proj_id_essay_tokens.csv")
val doc_tokens = docs.map(_.split(","))
val val doc_tokens_2 = doc_tokens.map(x => (x(1), x(0)))
val doc_tokens_3 = doc_tokens_2.mapValues(x => x.split('|'))
val doc_tokens_4 = doc_tokens_3.mapValues(_.groupBy(x => x))
val doc_tokens_5 = doc_tokens_4.mapValues(_.map(x => (x._1, x._2.length)))
val doc_tokens_6 =  doc_tokens_5.mapValues(_.toArray).flatMap(x => x._2.map(y => (y._1, x._1, y._2)))
val doc_tokens_map = doc_tokens_6.map(x => (x._1, (x._2, x._3)))
 
//join
val tfidf = dfs_4.join(doc_tokens_map)
val tfidf_2 = tfidf.map(x => (x._2._2._1, x._1, x._2._2._2, x._2._1))
//val tfidf_2 = tfidf.map(x =>
//    val token = x._1
//    val df = x._2._1
//    val proj_id = x._2._2._1
//    val tf = x._2._2._2
//    (proj_id, token, tf, df)
//)
val tfidf_3 = tfidf_2.map(x => (x._1, (x._2, x._3, x._4))).groupByKey()
val tfidf_4 = tfidf_3.mapValues(x => x.map(_.productIterator.mkString(":")))
val tfidf_5 = tfidf_4.mapValues(_.mkString("|"))
val tfidf_5 = tfidf_4.mapValues(_.mkString("|")).sortByKey()
val output = tfidf_5.map(x => "%s,%s".format(x._1, x._2))
output.saveAsTextFile("/tmp/tfidf")
