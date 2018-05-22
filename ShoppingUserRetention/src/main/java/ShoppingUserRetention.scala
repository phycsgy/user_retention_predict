package com.sankuai.data

import java.util.Date
import java.text.SimpleDateFormat

import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler

/**
  * Created by phycsgy on 17/8/10.
  * 预测
  */
object ShoppingUserRetention extends App{
  /*set up date args*/
  val dateFormat = new SimpleDateFormat("yyyy-MM-dd")
  var beginDate = ""
  var endDate = ""
  if(args.length == 0)
  {
    beginDate = dateFormat.format(new Date())
    endDate = beginDate
  }
  else
  {
    beginDate = args(0).split(",")(0)
    if(args(0).split(",").length > 1)
      endDate = args(0).split(",")(1)
    else
      endDate = beginDate
  }

  println(s"Begin date is $beginDate")
  println(s"End date is $endDate")
  /*set up conf*/
  val sc = new SparkContext()
  val hiveContext = new HiveContext(sc)

  val all_data = hiveContext.sql(
    """select os, last_visit_date_diff, last_visit_pv, has_search, result as label
        from mart_shstrategybi.shopping_user_active_predict_trainning_data
        where week_end_date = '2018-05-06'
    """)

  val assembler = new VectorAssembler().setInputCols(Array("os","last_visit_date_diff","last_visit_pv","has_search"))
                          .setOutputCol("features")

  val trainning_data = assembler.transform(all_data)

  val lr = new LogisticRegression()
  lr.setMaxIter(100).setRegParam(0.01)
  val model = lr.fit(trainning_data)



}
