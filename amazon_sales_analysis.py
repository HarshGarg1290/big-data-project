from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Amazon Sales Analysis") \
    .getOrCreate()

# Load dataset into a Spark DataFrame
file_path = "/Amazon_Sale_Report.csv"  
df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

# Data Cleaning: Remove rows with null values
df_cleaned = df.na.drop()

# Convert Date column to proper date format
df_cleaned = df_cleaned.withColumn("Date", to_date(col("Date"), "MM-dd-yy"))

# Perform basic analysis: Total sales amount grouped by Category
category_sales = df_cleaned.groupBy("Category").sum("Amount").orderBy("sum(Amount)", ascending=False)

# Show results
category_sales.show()

# Save the output to HDFS
# Save the output to HDFS with overwrite mode
category_sales.write.mode("overwrite").csv("hdfs://namenode:9000/datasets/amazon_dataset/category_output_2.csv")

# Perform additional analysis (optional):
# 1. Filter cancelled orders
cancelled_orders = df_cleaned.filter(df_cleaned["Status"] == "Cancelled")
cancelled_orders.show()

# 2. Total orders by Sales Channel
sales_channel_orders = df_cleaned.groupBy("Sales Channel").count()
sales_channel_orders.show()

# Stop the Spark session
spark.stop()
