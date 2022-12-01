# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ../../../_resources/00-global-setup $reset_all_data=$reset_all_data $db_prefix=manufacturing

# COMMAND ----------

# DBTITLE 1,Package imports
from pyspark.sql.functions import rand, input_file_name, from_json, col, unix_timestamp, to_timestamp, avg
from pyspark.sql.types import *
from pyspark.sql.window import Window
 
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

#ML import
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from mlflow.utils.file_utils import TempDir
import mlflow.spark
import mlflow
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from time import sleep
import re

# COMMAND ----------

raw_data_location = "/demos/manufacturing/iot_turbine" 

data_available = test_not_empty_folder(raw_data_location+"/incoming-data-json") and test_not_empty_folder(raw_data_location+"/incoming-data-json")
if not data_available:
  dbutils.fs.cp("/mnt/field-demos/manufacturing/iot_turbine/incoming-data-json", raw_data_location+"/incoming-data-json", True)
  dbutils.fs.cp("/mnt/field-demos/manufacturing/iot_turbine/status", raw_data_location+"/status", True)

# COMMAND ----------

# DBTITLE 1,Download data from source
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
download = False
cloud_storage_path = cloud_storage_path + "/iot_turbine"
raw_data_location = "/demos/manufacturing/iot_turbine" 
field_demo_bucket = test_not_empty_folder("/mnt/field-demos/manufacturing/iot_turbine") 
if field_demo_bucket:
  data_available = test_not_empty_folder(raw_data_location+"/incoming-data-json") and test_not_empty_folder(raw_data_location+"/incoming-data-json")
  if not data_available or reset_all_data:
    print(f"data isn't available under {raw_data_location}, copying data from local bucket")
    dbutils.fs.cp("/mnt/field-demos/manufacturing/iot_turbine/incoming-data-json", raw_data_location+"/incoming-data-json", True)
    dbutils.fs.cp("/mnt/field-demos/manufacturing/iot_turbine/status", raw_data_location+"/status", True)
  else:
    print(f"Data already existing, skip download.")
else:
  data_exists = test_not_empty_folder(raw_data_location)
  if not data_exists or reset_all_data:
    print(f"Couldn't find data saved in the default mounted bucket. Will download it instead under {raw_data_location}. This can take a few minutes...")
  #   print("Note: you need to specify your Kaggle Key under ./_resources/_kaggle_credential ...")
    result = dbutils.notebook.run("./_resources/01_download", 1200, {"cloud_storage_path": raw_data_location})
    if result is not None and "ERROR" in result:
      print("-------------------------------------------------------------")
      print("---------------- !! ERROR DOWNLOADING DATASET !!-------------")
      print("-------------------------------------------------------------")
      print(result)
      print("-------------------------------------------------------------")
      raise RuntimeError(result)
    else:
      print(f"Success. Dataset downloaded and saved under {raw_data_location}.")
  else:
    print(f"Data already existing, skip download.")


# COMMAND ----------

# DBTITLE 1,Create "gold" tables for autoML(remove ID/Timestamp columns) and ML purposes
if reset_all_data or not spark._jsparkSession.catalog().tableExists('turbine_dataset_for_ml'):
  spark.read.load(f"/mnt/field-demos/manufacturing/iot_turbine/gold-data-for-ml").write.mode('overwrite').saveAsTable("turbine_dataset_for_ml")
