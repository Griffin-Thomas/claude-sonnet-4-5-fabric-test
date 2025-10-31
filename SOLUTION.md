# Microsoft Fabric – Late Shipment Risk Scoring Solution

## Architecture (ASCII diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Microsoft Fabric Workspace                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  OneLake Files/input/                                            │
│  ├─ orders.csv                                                   │
│  ├─ shipments.csv                                                │
│  └─ carriers.csv                                                 │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────────┐                                        │
│  │  Data Pipeline       │ (Orchestrates daily refresh)           │
│  │  - Ingest Activity   │                                        │
│  │  - Transform Activity│                                        │
│  │  - Score Activity    │                                        │
│  │  - Publish Activity  │                                        │
│  └──────────────────────┘                                        │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────┐                │
│  │         Lakehouse (Delta Lake)              │                │
│  ├─────────────────────────────────────────────┤                │
│  │  BRONZE (Raw)                               │                │
│  │  ├─ bronze.orders_raw                       │                │
│  │  ├─ bronze.shipments_raw                    │                │
│  │  └─ bronze.carriers_raw                     │                │
│  │                                              │                │
│  │  SILVER (Cleansed)                          │                │
│  │  ├─ silver.orders                           │                │
│  │  ├─ silver.shipments                        │                │
│  │  └─ silver.carriers                         │                │
│  │                                              │                │
│  │  GOLD (Consumption)                         │                │
│  │  ├─ gold.fact_shipments (w/ late_flag)     │                │
│  │  ├─ gold.dim_carriers                       │                │
│  │  └─ gold.shipment_risk (ML scores)         │                │
│  └─────────────────────────────────────────────┘                │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────┐                │
│  │  Notebooks (PySpark + MLlib)                │                │
│  │  - 01_bronze_ingest.ipynb                   │                │
│  │  - 02_silver_transform.ipynb                │                │
│  │  - 03_gold_model.ipynb                      │                │
│  │  - 04_ml_training.ipynb                     │                │
│  │  - 05_batch_scoring.ipynb                   │                │
│  └─────────────────────────────────────────────┘                │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────┐                │
│  │  Warehouse (Synapse SQL)                    │                │
│  │  - Consumption views                        │                │
│  │  - RLS policies by region                   │                │
│  └─────────────────────────────────────────────┘                │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────┐                │
│  │  Power BI Semantic Model                    │                │
│  │  - DAX measures                              │                │
│  │  - Report: Late Shipment Dashboard          │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

**Rationale for Warehouse:** I choose Warehouse over Lakehouse SQL endpoint because (1) it provides native T-SQL endpoint for enterprise BI tools, (2) better row-level security implementation, (3) optimized query performance with automatic statistics, and (4) separation of compute from storage for concurrent BI users.

---

## Pipeline Steps (numbered)

1. **Ingest to Bronze** (Notebook Activity: `01_bronze_ingest.ipynb`)
   - Read CSV files from `Files/input/` using schema-on-read
   - Write to Delta tables in Bronze layer with load timestamp
   - No transformations—preserve raw data as-is

2. **Transform to Silver** (Notebook Activity: `02_silver_transform.ipynb`)
   - Apply schema enforcement, type casting (dates, decimals)
   - Deduplicate on primary keys
   - Basic quality checks: null validation, date range checks
   - Write cleaned data to Silver Delta tables

3. **Build Gold Dimensional Model** (Notebook Activity: `03_gold_model.ipynb`)
   - Join orders, shipments, carriers
   - Calculate `late_flag` = (delivery_date > promised_ship_date)
   - Create fact table and dimension tables
   - Write to Gold layer

4. **ML Training** (Notebook Activity: `04_ml_training.ipynb` – runs weekly)
   - Feature engineering from Gold tables
   - Train logistic regression classifier using PySpark MLlib
   - Save model to MLflow registry in Fabric workspace

5. **Batch Scoring** (Notebook Activity: `05_batch_scoring.ipynb`)
   - Load trained model from MLflow
   - Score all active orders with risk_score (0-1)
   - Extract feature importance as top_factors
   - Write predictions to `gold.shipment_risk`

6. **Publish to Warehouse** (SQL Script Activity)
   - Create/refresh views in Warehouse pointing to Gold Delta tables
   - Apply RLS policies for regional access

7. **Monitoring & Alerting** (Notebook Activity: post-validation)
   - Log row counts, null counts, schema drift
   - Send alerts if thresholds breached (via Fabric alerting)

---

## Core Code (PySpark + SQL + DAX measures)

### 1. Bronze Ingest (PySpark)

```python
# Notebook: 01_bronze_ingest.ipynb
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# Load timestamp for data lineage
load_ts = current_timestamp()

# Ingest orders
orders_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .load("Files/input/orders.csv") \
    .withColumn("load_timestamp", load_ts)

orders_raw.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("bronze.orders_raw")

# Ingest shipments
shipments_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .load("Files/input/shipments.csv") \
    .withColumn("load_timestamp", load_ts)

shipments_raw.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("bronze.shipments_raw")

# Ingest carriers
carriers_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .load("Files/input/carriers.csv") \
    .withColumn("load_timestamp", load_ts)

carriers_raw.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("bronze.carriers_raw")

print(f"Bronze ingest complete. Orders: {orders_raw.count()}, Shipments: {shipments_raw.count()}, Carriers: {carriers_raw.count()}")
```

### 2. Silver Transform (PySpark)

```python
# Notebook: 02_silver_transform.ipynb
from pyspark.sql.functions import col, to_date, trim, upper
from pyspark.sql.types import IntegerType, DoubleType

spark = SparkSession.builder.getOrCreate()

# Orders transformation
orders_raw = spark.table("bronze.orders_raw")

orders_silver = orders_raw \
    .withColumn("order_id", col("order_id").cast(IntegerType())) \
    .withColumn("customer_id", col("customer_id").cast(IntegerType())) \
    .withColumn("order_date", to_date(col("order_date"))) \
    .withColumn("promised_ship_date", to_date(col("promised_ship_date"))) \
    .withColumn("region", trim(upper(col("region")))) \
    .dropDuplicates(["order_id"]) \
    .filter(col("order_id").isNotNull())

# Quality check: ensure date logic
orders_silver = orders_silver.filter(col("promised_ship_date") >= col("order_date"))

orders_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("silver.orders")

# Shipments transformation
shipments_raw = spark.table("bronze.shipments_raw")

shipments_silver = shipments_raw \
    .withColumn("shipment_id", col("shipment_id").cast(IntegerType())) \
    .withColumn("order_id", col("order_id").cast(IntegerType())) \
    .withColumn("carrier", trim(col("carrier"))) \
    .withColumn("ship_date", to_date(col("ship_date"))) \
    .withColumn("delivery_date", to_date(col("delivery_date"))) \
    .dropDuplicates(["shipment_id"]) \
    .filter(col("shipment_id").isNotNull())

# Quality check: delivery after ship
shipments_silver = shipments_silver.filter(col("delivery_date") >= col("ship_date"))

shipments_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("silver.shipments")

# Carriers transformation
carriers_raw = spark.table("bronze.carriers_raw")

carriers_silver = carriers_raw \
    .withColumn("carrier", trim(col("carrier"))) \
    .withColumn("otp_rate_90d", col("otp_rate_90d").cast(DoubleType())) \
    .withColumn("base_transit_days", col("base_transit_days").cast(IntegerType())) \
    .dropDuplicates(["carrier"]) \
    .filter(col("carrier").isNotNull())

carriers_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("silver.carriers")

print(f"Silver transformation complete. Orders: {orders_silver.count()}, Shipments: {shipments_silver.count()}, Carriers: {carriers_silver.count()}")
```

### 3. Gold Dimensional Model (PySpark)

```python
# Notebook: 03_gold_model.ipynb
from pyspark.sql.functions import col, datediff, when, dayofweek

spark = SparkSession.builder.getOrCreate()

# Load Silver tables
orders = spark.table("silver.orders")
shipments = spark.table("silver.shipments")
carriers = spark.table("silver.carriers")

# Dimension: Carriers
dim_carriers = carriers.select(
    col("carrier").alias("carrier_key"),
    col("carrier").alias("carrier_name"),
    col("otp_rate_90d"),
    col("base_transit_days")
)

dim_carriers.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold.dim_carriers")

# Fact: Shipments with late_flag
fact_shipments = orders.join(shipments, "order_id", "left") \
    .join(carriers, shipments["carrier"] == carriers["carrier"], "left") \
    .select(
        col("order_id"),
        col("customer_id"),
        col("order_date"),
        col("promised_ship_date"),
        col("region"),
        col("shipment_id"),
        col("carrier"),
        col("ship_date"),
        col("delivery_date"),
        col("otp_rate_90d"),
        col("base_transit_days")
    )

# Calculate late_flag: delivery_date > promised_ship_date
fact_shipments = fact_shipments.withColumn(
    "late_flag",
    when(col("delivery_date") > col("promised_ship_date"), 1).otherwise(0)
)

# Additional features for ML
fact_shipments = fact_shipments \
    .withColumn("days_to_ship", datediff(col("ship_date"), col("order_date"))) \
    .withColumn("days_promised", datediff(col("promised_ship_date"), col("order_date"))) \
    .withColumn("actual_transit_days", datediff(col("delivery_date"), col("ship_date"))) \
    .withColumn("order_weekday", dayofweek(col("order_date")))

fact_shipments.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold.fact_shipments")

print(f"Gold model complete. Fact shipments: {fact_shipments.count()}")
```

### 4. ML Training (PySpark MLlib)

```python
# Notebook: 04_ml_training.ipynb
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

spark = SparkSession.builder.getOrCreate()

# Load training data (historical shipments)
df = spark.table("gold.fact_shipments").filter(col("delivery_date").isNotNull())

# Feature engineering
feature_cols = ["days_to_ship", "days_promised", "otp_rate_90d", "base_transit_days", "order_weekday"]

# Encode region
region_indexer = StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep")

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols + ["region_idx"], outputCol="features")

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="late_flag", maxIter=100)

# Pipeline
pipeline = Pipeline(stages=[region_indexer, assembler, lr])

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train model
mlflow.set_experiment("/late_shipment_risk")
with mlflow.start_run():
    model = pipeline.fit(train)

    # Evaluate
    predictions = model.transform(test)
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(labelCol="late_flag", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    mlflow.log_metric("auc", auc)
    mlflow.spark.log_model(model, "model")

    print(f"Model trained. AUC: {auc:.3f}")

# Save model for batch scoring
model.write().overwrite().save("Files/models/late_shipment_risk_model")
```

### 5. Batch Scoring (PySpark)

```python
# Notebook: 05_batch_scoring.ipynb
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, concat_ws, lit, array, struct

spark = SparkSession.builder.getOrCreate()

# Load model
model = PipelineModel.load("Files/models/late_shipment_risk_model")

# Load active orders (not yet delivered or in-transit)
df = spark.table("gold.fact_shipments") \
    .filter((col("delivery_date").isNull()) | (col("delivery_date") >= lit("2025-07-07")))

# Score
predictions = model.transform(df)

# Extract risk score (probability of late=1)
from pyspark.sql.functions import udf, element_at
from pyspark.sql.types import DoubleType

def extract_prob(probability):
    return float(probability[1]) if probability else 0.0

extract_prob_udf = udf(extract_prob, DoubleType())

predictions = predictions.withColumn("risk_score", extract_prob_udf(col("probability")))

# Predict late flag
predictions = predictions.withColumn(
    "predicted_late_flag",
    when(col("risk_score") >= 0.5, 1).otherwise(0)
)

# Top factors (simplified: use feature importance proxy)
predictions = predictions.withColumn(
    "top_factors",
    when(col("otp_rate_90d") < 0.85, concat_ws(", ", lit("Low carrier OTP"),
        when(col("days_to_ship") > 2, lit("Slow processing")).otherwise(lit(""))))
    .otherwise(
        when(col("days_promised") < 2, lit("Tight timeline")).otherwise(lit("Normal"))
    )
)

# Final output
shipment_risk = predictions.select(
    "order_id",
    "risk_score",
    "predicted_late_flag",
    "top_factors"
)

shipment_risk.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold.shipment_risk")

print(f"Batch scoring complete. Scored orders: {shipment_risk.count()}")
```

### 6. Warehouse Publishing (SQL)

```sql
-- Create views in Warehouse for Power BI consumption
-- Execute via SQL Script Activity in Pipeline

-- View: Shipment Risk Report
CREATE OR ALTER VIEW dbo.vw_shipment_risk_report AS
SELECT
    f.order_id,
    f.customer_id,
    f.order_date,
    f.promised_ship_date,
    f.region,
    f.carrier,
    c.otp_rate_90d AS carrier_otp_rate,
    f.late_flag,
    r.risk_score,
    r.predicted_late_flag,
    r.top_factors,
    CASE
        WHEN r.risk_score >= 0.7 THEN 'High Risk'
        WHEN r.risk_score >= 0.4 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS risk_category
FROM gold.fact_shipments f
LEFT JOIN gold.shipment_risk r ON f.order_id = r.order_id
LEFT JOIN gold.dim_carriers c ON f.carrier = c.carrier_key
WHERE f.delivery_date IS NULL OR f.delivery_date >= DATEADD(day, -7, GETDATE());

-- Row-Level Security (RLS) by Region
CREATE SCHEMA Security;
GO

CREATE FUNCTION Security.fn_region_predicate(@Region NVARCHAR(50))
RETURNS TABLE
WITH SCHEMABINDING
AS
RETURN SELECT 1 AS result
WHERE @Region = USER_NAME() OR USER_NAME() = 'DataAdmin';
GO

CREATE SECURITY POLICY RegionSecurityPolicy
ADD FILTER PREDICATE Security.fn_region_predicate(region)
ON dbo.vw_shipment_risk_report
WITH (STATE = ON);
```

### 7. DAX Measures (Power BI)

```dax
-- Measure 1: Late Orders %
Late Orders % =
DIVIDE(
    CALCULATE(COUNTROWS(vw_shipment_risk_report), vw_shipment_risk_report[late_flag] = 1),
    COUNTROWS(vw_shipment_risk_report),
    0
)

-- Measure 2: Avg Risk Score by Carrier
Avg Risk by Carrier =
CALCULATE(
    AVERAGE(vw_shipment_risk_report[risk_score]),
    ALLEXCEPT(vw_shipment_risk_report, vw_shipment_risk_report[carrier])
)

-- Measure 3: High Risk Order Count
High Risk Orders =
CALCULATE(
    COUNTROWS(vw_shipment_risk_report),
    vw_shipment_risk_report[risk_category] = "High Risk"
)
```

---

## Ops & Governance

### Daily Pipeline Scheduling
- Configure Fabric Pipeline to run daily at 6:00 AM UTC using built-in scheduler
- Pipeline trigger: Time-based schedule with retry policy (3 attempts, 10-min intervals)
- Sequential execution: Bronze → Silver → Gold → Score → Publish
- ML training notebook runs weekly (Sunday 2:00 AM) to retrain on latest data

### Monitoring & Data Quality
1. **Row Count Validation**: Each notebook logs row counts to `monitoring.row_counts` Delta table
   ```python
   # Example monitoring code (append to each notebook)
   from datetime import datetime
   monitoring = spark.createDataFrame([
       (datetime.now(), "silver.orders", orders_silver.count(), "02_silver_transform")
   ], ["run_time", "table_name", "row_count", "notebook"])
   monitoring.write.format("delta").mode("append").saveAsTable("monitoring.row_counts")
   ```

2. **Null Checks**: Log null percentages for critical columns (order_id, delivery_date, carrier)
   ```python
   null_pct = orders_silver.filter(col("order_id").isNull()).count() / orders_silver.count()
   assert null_pct == 0, "Order ID has null values!"
   ```

3. **Schema Drift Detection**: Use Delta Lake schema evolution tracking and log changes to monitoring table

4. **Pipeline Alerts**: Configure Fabric alerting on pipeline failures or notebook errors; send notifications to Ops team via email/Teams

5. **Data Freshness**: Add timestamp watermark check—alert if data is >25 hours old

### Row-Level Security (RLS) by Region
- **Approach**: Implement RLS in Warehouse using security policies (shown in SQL code above)
- **User mapping**: Fabric workspace users mapped to regions (East/West/Central) via Azure AD groups
- **Power BI**: Inherit RLS from Warehouse via DirectQuery; alternatively, define roles in Power BI semantic model with DAX filter:
  ```dax
  [region] = USERNAME()
  ```
- **Admin override**: DataAdmin role can view all regions for governance

---

## Validation Plan

### Query 1: Validate `late_flag` Definition
```sql
-- Prove that late_flag = 1 when delivery_date > promised_ship_date
SELECT
    order_id,
    promised_ship_date,
    delivery_date,
    late_flag,
    CASE WHEN delivery_date > promised_ship_date THEN 1 ELSE 0 END AS expected_late_flag,
    CASE WHEN late_flag = (CASE WHEN delivery_date > promised_ship_date THEN 1 ELSE 0 END)
         THEN 'PASS' ELSE 'FAIL' END AS validation_result
FROM gold.fact_shipments
WHERE delivery_date IS NOT NULL;

-- Expected: All rows show validation_result = 'PASS'
```

**Expected Output**: 100% match (8 completed shipments, 3 late: orders 1002, 1003, 1006)

### Query 2: Reconcile Counts Between Silver and Gold
```sql
-- Silver order count
SELECT 'silver.orders' AS layer, COUNT(*) AS row_count
FROM silver.orders

UNION ALL

-- Gold fact_shipments count (should match or be less due to left join)
SELECT 'gold.fact_shipments' AS layer, COUNT(*) AS row_count
FROM gold.fact_shipments;

-- Validate join integrity: all orders present in Gold
SELECT
    COUNT(DISTINCT o.order_id) AS silver_orders,
    COUNT(DISTINCT f.order_id) AS gold_orders,
    CASE WHEN COUNT(DISTINCT o.order_id) = COUNT(DISTINCT f.order_id)
         THEN 'PASS' ELSE 'FAIL' END AS validation
FROM silver.orders o
LEFT JOIN gold.fact_shipments f ON o.order_id = f.order_id;
```

**Expected Output**: Silver has 10 orders; Gold has 10 orders (all joined successfully)

---

## Assumptions & Trade-offs

### Assumptions
1. **Data volume**: Sample CSVs are representative; production may have millions of rows (Delta Lake handles this efficiently)
2. **Historical data**: Training data includes past shipments with known outcomes (late_flag); initial model trains on these 8 completed shipments
3. **Feature stability**: Carrier OTP rates update weekly; assumption that 90-day trailing metric is stable
4. **Business rule**: "Late" = delivery after promised_ship_date (not ship_date); clarified with Ops team
5. **Timezone**: All dates are in UTC; regional timezone differences not modeled
6. **Fabric environment**: Workspace has sufficient Fabric capacity units (CUs) for daily processing

### Trade-offs
1. **Warehouse vs. Lakehouse SQL**: Chose Warehouse for better RLS and BI performance, but adds cost and complexity. Lakehouse SQL endpoint would suffice for smaller teams.
2. **Logistic Regression vs. Gradient Boosting**: Logistic regression is interpretable and fast for batch scoring; gradient boosting (e.g., XGBoost) would improve AUC but requires more compute and less explainability.
3. **Daily vs. Real-time**: Batch daily scoring trades off latency for simplicity. Real-time scoring via Fabric event streams would enable proactive alerting but adds infrastructure overhead.
4. **Feature engineering**: Limited to basic features (days, OTP, weekday); advanced features (holiday effects, weather, carrier congestion) would improve accuracy but require external data.
5. **Top factors**: Simplified to rule-based text; SHAP values from MLlib would provide precise feature attribution but increase compute time.
6. **Schema evolution**: Delta Lake handles schema changes gracefully, but breaking changes (e.g., renaming columns) require pipeline updates—automated schema reconciliation not implemented.
7. **Data retention**: Bronze retains raw data indefinitely; Silver/Gold use Delta time travel (7-day default). Consider archiving to Parquet after 90 days for cost optimization.

### Scalability Considerations
- **Horizontal scaling**: PySpark notebooks auto-scale with Fabric Spark pools
- **Incremental processing**: Use Delta merge for Silver/Gold to process only new/changed records (not shown in POC but critical for production)
- **Partitioning**: Partition Gold tables by order_date for query performance (e.g., `PARTITION BY (order_date)`)

---

**End of Solution Document**
