# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance: model training
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC We saw [previously]($./01-Wind-Turbine-DLT-pipeline) how to create our ingestion pipeline to build our Dataset. Data Engineers have granted read access to this file and our Data Scientist team can now start building a model to detect anomaly.
# MAGIC 
# MAGIC 
# MAGIC Our dataset consists of vibration readings coming off sensors located in the gearboxes of wind turbines. 
# MAGIC 
# MAGIC 
# MAGIC This is the flow we'll be implementing:
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/wind_turbine/turbine-ds-flow-0.png" width="1000px" />
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right; margin: -10px 50px 0px 50px">
# MAGIC   <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width="400px" /><br/>
# MAGIC   *locations of the sensors*
# MAGIC </div>
# MAGIC 
# MAGIC We will use Gradient Boosted Tree Classification to predict which set of vibrations could be indicative of a failure.
# MAGIC 
# MAGIC One the model is trained, we'll use MFLow to track its performance and save it in the registry to deploy it in production
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC *Data Source Acknowledgement: This Data Source Provided By NREL*
# MAGIC 
# MAGIC *https://www.nrel.gov/docs/fy12osti/54530.pdf*
# MAGIC 
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fmanufacturing%2Fwind_turbine%2Fnotebook_ml&dt=MANUFACTURING_WIND_TURBINE">

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## 1/ Data exploration
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/wind_turbine/turbine-ds-flow-1.png" width="700px" style="float: right" />
# MAGIC 
# MAGIC 
# MAGIC Databricks let you run all kind of data exploration, including on big data using pandas on spark.
# MAGIC 
# MAGIC Databricks provide out of the box Data Profiling and visualization, but we can also use standard python libraries to run some custom Analysis.
# MAGIC 
# MAGIC Let's see what the distributions of sensor readings look like for our turbines.
# MAGIC 
# MAGIC *Notice the much larger stdev in AN8, AN9 and AN10 for Damaged turbined.*

# COMMAND ----------

dataset = spark.read.table("turbine_dataset_for_ml")
display(dataset)

# COMMAND ----------

# DBTITLE 1,Sensor metrics distribution
gold_turbine_dfp = dataset.sample(0.005).toPandas()
g = sns.PairGrid(gold_turbine_dfp[['AN3', 'AN4' ,'AN9' ,'status']], diag_sharey=False, hue="status")
g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(sns.kdeplot)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## 2/ Model Creation: Workflows with Pyspark.ML Pipeline
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/wind_turbine/turbine-ds-flow-2.png" width="700px" style="float: right" />
# MAGIC 
# MAGIC 
# MAGIC In this example, we'll be using SparkML to build our Model because our dataset is quite big. 
# MAGIC 
# MAGIC However, Databricks let you run all kind of flavor: SKlearn, XGBoost, Pytorch/TF or any extra libraries you'd like to install.
# MAGIC 
# MAGIC In addition, you can use Databricks AutoML to try multiple model flavors and bootstrap this notebook for you with the best model available, without having to write any code (see below).
# MAGIC 
# MAGIC We'll use MLFlow to track all our model metrics but also the model itself. This will allow us to have a full reproducibility and enforce ML governance while saving a lot of time to DS teams. 
# MAGIC 
# MAGIC *Note: see how the Run starts in the right Experiment menu. Feel free to explore the run using MLFlow UI.*

# COMMAND ----------

# DBTITLE 1,Splitting training / test dataset
  training, test = dataset.sample(0.2).randomSplit([0.9, 0.1], seed = 42)

# COMMAND ----------

# DBTITLE 1,ML pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
init_experiment_for_batch("manuf_wind_turbine", "demos_wind_turbine_predictive_maintenance")

with mlflow.start_run():
  #the source table will automatically be logged to mlflow
  mlflow.spark.autolog()
  
  gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
  grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3,4,5]).build()

  metrics = MulticlassClassificationEvaluator(metricName="f1")
  cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=metrics, numFolds=2)

  featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
  stages = [VectorAssembler(inputCols=featureCols, outputCol="va"), StandardScaler(inputCol="va", outputCol="features"), StringIndexer(inputCol="status", outputCol="label"), cv]
  pipeline = Pipeline(stages=stages)

  pipelineTrained = pipeline.fit(training)
  
  predictions = pipelineTrained.transform(test)
  metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd)
  
  mlflow.log_metric("precision", metrics.precision(1.0))
  mlflow.log_metric("recall", metrics.recall(1.0))
  mlflow.log_metric("f1", metrics.fMeasure(1.0))
  
  mlflow.spark.log_model(pipelineTrained, f"turbine_gbt", input_example={"AN3":-1.4746, "AN4":-1.8042, "AN5":-2.1093, "AN6":-5.1975, "AN7":-0.45691, "AN8":-7.0763, "AN9":-3.3133, "AN10":-0.0059799})
  mlflow.set_tag("model", "turbine_gbt") 
  
  #Add confusion matrix to the model:
  labels = pipelineTrained.stages[2].labels
  fig = plt.figure()
  sns.heatmap(pd.DataFrame(metrics.confusionMatrix().toArray()), annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
  plt.suptitle("Turbine Damage Prediction. F1={:.2f}".format(metrics.fMeasure(1.0)), fontsize = 18)
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  mlflow.log_figure(fig, "confusion_matrix.png") #Requires MLFlow 1.13 (%pip install mlflow==1.13.1)

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC 
# MAGIC ## 3/ Saving our model to MLFLow registry
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/wind_turbine/turbine-ds-flow-3.png" width="700px" style="float: right" />
# MAGIC 
# MAGIC Our model is now fully packaged. MLflow tracked every steps, and logged the full model for us.
# MAGIC 
# MAGIC The next step is now to get the best run out of MLFlow and move it to our model registry. Our Data Engineering team will then be able to retrieve it and use it to run inferences at scale, or deploy it using REST api for real-time use cases.
# MAGIC 
# MAGIC *Note: this step is typically involving hyperparameter tuning. Databricks AutoML setup all that for you.*

# COMMAND ----------

# DBTITLE 1,Save our new model to the registry as a new version
#get the best model having the best metrics.f1 from the registry
best_models = mlflow.search_runs(filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED" and metrics.f1 > 0', order_by=['metrics.f1 DESC'], max_results=1)
model_registered = mlflow.register_model("runs:/" + best_models.iloc[0].run_id + "/turbine_gbt", "field_demos_wind_turbine_maintenance")

# COMMAND ----------

# DBTITLE 1,Flag this version as production ready
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
client.transition_model_version_stage(name = f"field_demos_wind_turbine_maintenance", version = model_registered.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detecting damaged turbine in a production pipeline
# MAGIC 
# MAGIC That's it! Our model is deployed and our Data Engineering team can load it in the ingestion pipeline like any other component.
# MAGIC 
# MAGIC Let's see how this can also be done in this notebook directly:

# COMMAND ----------

# DBTITLE 1,Load the model from our registry
get_cluster_udf = mlflow.pyfunc.spark_udf(spark, "models:/field_demos_wind_turbine_maintenance/Production", "string")
#Save the mdoel as SQL function (we could call it using python too)
spark.udf.register("get_turbine_status", get_cluster_udf)

# COMMAND ----------

# DBTITLE 1,Let's call our model and make our predictive maintenance!
# MAGIC %sql SELECT get_turbine_status(struct(AN3, AN4, AN5, AN6, AN7, AN8, AN9, AN10)) as status_prediction, * FROM turbine_dataset_for_ml

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Going further with Databricks AutoML
# MAGIC 
# MAGIC <img style="float: right" width="500px" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC 
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC 
# MAGIC Bootstraping new ML projects can still be long and inefficient.<br/>
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC 
# MAGIC You can generate this notebook containing the best model and state of the art ML within 1 click!
# MAGIC 
# MAGIC To give it a try, switch to the Machine Learning menu, and click "Create -> AutoML experiment"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC %md 
# MAGIC ### We can now explore our prediction in a new dashboard
# MAGIC 
# MAGIC Go back to the [DLT ingestion pipeline notebook]($./01-Wind-Turbine-DLT-pipeline) or open the [DBSQL Predictive Maintenance dashboard]([DBSQL Dashboard](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/048c6d42-ad56-4667-ada1-e35f80164248-turbine-demo-predictions?o=1444828305810485)
# MAGIC <br/><br/>
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/wind_turbine/wind-turbine-dashboard.png" width="1000px">
