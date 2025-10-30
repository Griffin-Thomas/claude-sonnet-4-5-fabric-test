**Title:** Microsoft Fabric – Late Shipment Risk Scoring (batch)  
**Role:** You are an AI engineer working **in Microsoft Fabric**.  
**Business goal:** Operations wants a daily **“late shipment risk”** table and a simple report so they can proactively reach out to customers.

### Data you have
Put these small CSVs into OneLake at `Files/input/...` (or assume they already exist):

**`orders.csv`**
```
order_id,customer_id,order_date,promised_ship_date,region
1001,501,2025-07-01,2025-07-03,East
1002,502,2025-07-02,2025-07-05,West
1003,503,2025-07-02,2025-07-04,Central
1004,504,2025-07-03,2025-07-06,East
1005,505,2025-07-04,2025-07-07,West
1006,506,2025-07-05,2025-07-08,Central
1007,507,2025-07-06,2025-07-09,East
1008,508,2025-07-06,2025-07-08,West
1009,509,2025-07-07,2025-07-10,Central
1010,510,2025-07-07,2025-07-09,East
```

**`shipments.csv`**
```
shipment_id,order_id,carrier,ship_date,delivery_date
7001,1001,Contoso,2025-07-02,2025-07-04
7002,1002,Fabrikam,2025-07-04,2025-07-07
7003,1003,Northwind,2025-07-03,2025-07-07
7004,1004,Contoso,2025-07-05,2025-07-06
7005,1005,Fabrikam,2025-07-06,2025-07-09
7006,1006,Northwind,2025-07-07,2025-07-11
7007,1007,Contoso,2025-07-08,2025-07-09
7008,1008,Fabrikam,2025-07-07,2025-07-10
```

**`carriers.csv`**
```
carrier,otp_rate_90d,base_transit_days
Contoso,0.94,2
Fabrikam,0.88,3
Northwind,0.81,3
```

> **Late definition:** An order is “late” if `delivery_date` > `promised_ship_date`.

### What you must build (deliverables)

1. **Target architecture (Fabric-native):**  
   - Lakehouse **Bronze/Silver/Gold** layout (Delta): raw CSV → cleaned → model outputs.  
   - A **Fabric Pipeline** to orchestrate ingest → transform → score → publish.  
   - A **Warehouse** or semantic model for reporting (choose one and **justify**).

2. **Data engineering plan:**  
   - **PySpark** to land CSVs into **Bronze** (schema-on-read), clean ⇒ **Silver** (types, dedup, simple quality checks), and build **Gold** fact/dims with a `late_flag`.

3. **ML plan (batch scoring):**  
   - Features: e.g., `days_to_ship`, `days_promised`, `carrier_otp_rate_90d`, `region`, weekday effects.  
   - Train a simple classifier (e.g., **logistic regression**) in a **Fabric Notebook**; write the **daily batch scoring** that refreshes `gold.shipment_risk` with:
     ```
     order_id, risk_score (0-1), predicted_late_flag, top_factors (short text)
     ```

4. **Publishing for BI:**  
   - **SQL** to create a consumable table/view in **Warehouse** (or Lakehouse SQL endpoint).  
   - **2–3 DAX measures** you’d add to a Power BI model (e.g., *Late Orders %*, *Avg Risk by Carrier*).

5. **Ops & governance:**  
   - How the **Pipeline** is scheduled daily; where to add **monitoring** (row counts, null checks).  
   - **Row-level security** by `region` (describe approach).

6. **Validation:**  
   - Show **one** query that proves your `late_flag` definition matches the data.  
   - Show **one** query to reconcile counts between Silver and Gold.

7. **Output format (return exactly these sections in this order):**  
   - `Architecture (ASCII diagram)`  
   - `Pipeline Steps (numbered)`  
   - `Core Code (PySpark + SQL + 2–3 DAX measures)`  
   - `Ops & Governance`  
   - `Validation Plan`  
   - `Assumptions & Trade-offs`

**Constraints:** stay within **Microsoft Fabric primitives** (Lakehouse/Delta, Notebooks, Pipelines, Warehouse/Power BI). **No external services** or private packages. Keep prose ≤ **~1,000 words** (code excluded).

**Other constraints:**
   - Return **one** answer containing: an architecture, a step-by-step plan, code snippets, and a validation plan.  
   - Assume a Fabric workspace with **Lakehouse, Warehouse, Pipelines, Notebooks, and Power BI** enabled.  
   - Don’t call external services; stick to **Fabric + standard PySpark/SQL/DAX**.  
   - Timebox reasoning: **no more than ~1,000 words of prose** (code excluded).
