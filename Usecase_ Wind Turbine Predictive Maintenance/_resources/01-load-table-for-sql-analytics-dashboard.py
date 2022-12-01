# Databricks notebook source
# MAGIC %md #Table for SQL Analytics loader
# MAGIC ##Run this notebook to mount the final table and directly run queries on top of them
# MAGIC The tables are in a separate database from the one in the data ingestion notebook to prevent conflict, so that we can run SELECT queries with SQL Analytics  

# COMMAND ----------

# MAGIC %md ### Please don't delete/edit these table, just create them to access them on SQL Analytics, don't edit them on the demo notebooks 

# COMMAND ----------

# MAGIC %sql
# MAGIC create database if not exists field_demos_manufacturing;
# MAGIC CREATE TABLE if not exists `field_demos_manufacturing`.`turbine_bronze` USING delta LOCATION 'dbfs:/mnt/field-demos/manufacturing/iot_turbine/bronze';
# MAGIC CREATE TABLE if not exists `field_demos_manufacturing`.`turbine_silver` USING delta LOCATION 'dbfs:/mnt/field-demos/manufacturing/iot_turbine/silver';
# MAGIC CREATE TABLE if not exists `field_demos_manufacturing`.`turbine_gold`   USING delta LOCATION 'dbfs:/mnt/field-demos/manufacturing/iot_turbine/gold' ;
# MAGIC 
# MAGIC CREATE TABLE if not exists `field_demos_manufacturing`.`turbine_power_prediction` USING delta LOCATION 'dbfs:/mnt/field-demos/manufacturing/iot_turbine/power';
