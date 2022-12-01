# Databricks notebook source
dbutils.widgets.text("cloud_storage_path", "/demos/manufacturing/iot_turbine")

# COMMAND ----------

# MAGIC %pip install faker

# COMMAND ----------

try:
    cloud_storage_path
except NameError:
    cloud_storage_path = dbutils.widgets.get("cloud_storage_path")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wind Turbine Sensors Datasets
# MAGIC 
# MAGIC * The dataset used in this accelerator is from [NREL](https://www.nrel.gov/). National Renewable Energy Laboratory (NREL) in the US specializes in the research and development of renewable energy, energy efficiency, energy systems integration, and sustainable transportation. NREL is a federally funded research and development center sponsored by the Department of Energy and operated by the Alliance for Sustainable Energy, a joint venture between MRIGlobal and Battelle.
# MAGIC 
# MAGIC * NREL published this dataset on [OEDI](https://data.openei.org/submissions/738) in 2014. NREL collected data from a healthy and a damaged gearbox of the same design tested by the GRC. Vibration data were collected by accelerometers along with high-speed shaft RPM signals during the dynamometer testing.
# MAGIC   
# MAGIC * Further details about this dataset
# MAGIC   * Dataset title: Wind Turbine Gearbox Condition Monitoring Vibration Analysis Benchmarking Datasets
# MAGIC   * Dataset source URL: https://data.openei.org/submissions/738
# MAGIC   * Dataset license: please see dataset source URL above

# COMMAND ----------

print("Downloading dataset, this can take a few minutes, please be patient...")

# COMMAND ----------

# DBTITLE 1,Download wind turbine sensor dataset
# MAGIC %sh
# MAGIC mkdir -p /tmp/wind_turbine_download/
# MAGIC curl -o /tmp/wind_turbine_download/Healthy.zip "https://data.openei.org/files/738/Healthy.zip" -s
# MAGIC curl -o /tmp/wind_turbine_download/Damaged.zip "https://data.openei.org/files/738/Damaged.zip" -s

# COMMAND ----------

print("Extracting dataset, this can take a few minutes, please be patient...")

# COMMAND ----------

# DBTITLE 1,Unzip wind turbine sensors dataset
# MAGIC %sh
# MAGIC cd /tmp/wind_turbine_download
# MAGIC unzip -o Damaged.zip
# MAGIC unzip -o Healthy.zip
# MAGIC ls .

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Raw Layer & Save
# MAGIC Reformatting the data from mat to dataframes

# COMMAND ----------

# MAGIC %md
# MAGIC #### Process the data

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
import os
from scipy.io import loadmat

schema = StructType([
                    StructField('Speed', FloatType(), True),
                    StructField('Torque', FloatType(), True),
                    StructField('AN3', FloatType(), True),
                    StructField('AN4', FloatType(), True),
                    StructField('AN5', FloatType(), True),
                    StructField('AN6', FloatType(), True),
                    StructField('AN7', FloatType(), True),
                    StructField('AN8', FloatType(), True),
                    StructField('AN9', FloatType(), True),
                    StructField('AN10', FloatType(), True),
                    StructField('status', StringType(), True),
  ])
turbine_sensor_data = spark.createDataFrame([], schema)
print("formating data, the first run will take a few minutes...")

# COMMAND ----------

import os 
import pyspark.pandas as pd
damaged_root_dir = '/tmp/wind_turbine_download/Damaged'
healthy_root_dir = '/tmp/wind_turbine_download/Healthy'
keys = ['Speed', 'Torque', 'AN3', 'AN4', 'AN5', 'AN6', 'AN7', 'AN8', 'AN9', 'AN10']

for root, dirs, files in os.walk(damaged_root_dir):
    for file in files:
      mat_file = {k: v.flatten() for k, v in loadmat(damaged_root_dir+'/'+file).items() if k in keys}
      turbine_sensor_data = turbine_sensor_data.unionByName(pd.DataFrame(mat_file, columns=keys, dtype=float).to_spark().withColumn('status', lit('damaged')))

# COMMAND ----------

for root, dirs, files in os.walk(healthy_root_dir):
    for file in files:
      mat_file = {k: v.flatten() for k, v in loadmat(healthy_root_dir+'/'+file).items() if k in keys}
      turbine_sensor_data = turbine_sensor_data.unionByName(pd.DataFrame(mat_file, columns=keys, dtype=float).to_spark().withColumn('status', lit('healthy')))

# COMMAND ----------

turbine_sensor_data = turbine_sensor_data.withColumn('id', when(col('status')=='healthy', (rand(seed=1)*500).cast('int')).otherwise((rand(seed=1)*(1501-500)+500).cast('int'))).withColumn('key', monotonically_increasing_id())

# COMMAND ----------

from faker import Faker
import datetime
fake = Faker()
def fake_time():
  return fake.date_time_between(start_date=datetime.date(2018,1,1), end_date=datetime.date(2018,12,31))
fake_time_udf = udf(fake_time, TimestampType())

turbine_sensor_data = turbine_sensor_data.withColumn('Timestamp', fake_time_udf())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Bronze Data & Move to Storage

# COMMAND ----------

turbine_sensor_data = turbine_sensor_data.select(*[c.upper() for c in turbine_sensor_data.columns if c not in ['status', 'key']], 'key', 'status')
turbine_sensor_data = turbine_sensor_data.cache()
#display(turbine_sensor_data)

# COMMAND ----------

# DBTITLE 1,incoming_data_json
turbine_sensor_data.drop('key', 'status').write.mode('overwrite').json(f'{cloud_storage_path}/incoming-data-json')

# COMMAND ----------

# DBTITLE 1,incoming_data parquet
#turbine_sensor_data.withColumn('value', to_json(struct(col("SPEED"), col("TORQUE"), col("AN3"), col("AN4"), col("AN5"), col("AN6"), col("AN7"), col("AN8"), col("AN9"), col("AN10"), col("ID"), col("TIMESTAMP")))).select("key", "value").write.mode('overwrite').parquet(f'{cloud_storage_path}/incoming-data')

# COMMAND ----------

# DBTITLE 1,status
turbine_sensor_data.select('id', 'status').distinct().write.mode('overwrite').parquet(f'{cloud_storage_path}/status')

# COMMAND ----------

# DBTITLE 1,Gold data for ML
turbine_sensor_data.select("AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "status").write.mode('overwrite').format("delta").save(f'{cloud_storage_path}/gold-data-for-ml')
turbine_sensor_data.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2022]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party data are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Dataset Name|Dataset license | Dataset License URL | Dataset Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Wind Turbine Gearbox Condition Monitoring Vibration Analysis Benchmarking Datasets|Creative Commons 4.0 License| https://data.openei.org/files/738/WindTurbineConditionMonitoringLicenseInfo.txt | https://data.openei.org/submissions/738
# MAGIC 
# MAGIC <!-- |Kaggle|Apache-2.0 License |https://github.com/Kaggle/kaggle-api/blob/master/LICENSE|https://github.com/Kaggle/kaggle-api| -->
