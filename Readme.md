# Neo4j EDA & ML on Orders–People–Returns Dataset

**Team Members:**
- Mohan Kumar Tulabandu  
- Shravan Shankar  
- Glen Correia  

---

## 📖 Project Overview

This repository demonstrates an end‑to‑end Exploratory Data Analysis (EDA) and predictive modeling workflow on a retail **Orders–People–Returns** dataset, leveraging:

- **Data cleaning & feature engineering** in Python (pandas, scikit‑learn, imbalanced‑learn)  
- **Graph modeling & analytics** in Neo4j 5 with APOC & GDS Community Edition  
- **Machine learning** (leakage‑free LightGBM pipeline)  
- **Visualizations** of key insights and model performance  

---

## 📂 Repository Structure

```
.
├── data/
│   ├── orders.csv
│   ├── people.csv
│   └── returns.csv
│
├── import/
│   ├── orders_clean.csv
│   ├── people.csv
│   └── returns_clean.csv
│
├── notebooks/
│   └── eda_cleaning.ipynb        ← Data cleaning & export to CSV
│
├── cypher/
│   ├── import_people.cypher      ← Load Manager & Region
│   ├── import_orders.cypher      ← Load Customer, Order, Region
│   └── import_products.cypher    ← Load Product & CONTAINS
│
├── scripts/
│   ├── ml_pipeline_noleak.py     ← Leakage‑free LightGBM training & evaluation
│   └── ml_pipeline_smote_xgb.py  ← SMOTE + XGBoost alternative
│
├── visuals/
│   ├── roc_curve_noleak.png
│   ├── feature_importances_noleak.png
│   └── ...                       ← Other output charts
│
├── eda_presentation.pptx         ← Final slide deck
└── README.md
```

---

## ⚙️ Prerequisites

1. **Python 3.10 or 3.11** (recommended)  
2. **Neo4j Desktop 5.x** with:
   - APOC 5.24.2  
   - GDS 2.12.0  
3. **Java 8+** (for Neo4j)  
4. **pip** or **conda** for Python package management  

---

## 🛠️ Setup & Installation

1. **Clone the repo**  
   ```bash
   git clone <repo_url> && cd project-2
   ```

2. **Create & activate a venv**  
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install pandas numpy openpyxl scikit-learn matplotlib seaborn neo4j lightgbm imbalanced-learn
   ```

---

## 1️⃣ Data Cleaning

Open `notebooks/eda_cleaning.ipynb` and run all cells to:

- Parse dates (`Order Date`, `Ship Date`) and numeric fields  
- Drop null key rows, dedupe  
- Aggregate `returns.csv` into `returned_count` per order  
- Rename to snake_case  
- Export **`import/orders_clean.csv`**  

---

## 2️⃣ Neo4j: Setup & Import

1. Copy CSVs to Neo4j import folder  
2. Configure `neo4j.conf` for APOC & GDS  
3. Enable APOC 5.24.2 & GDS 2.12.0, restart  
4. Run schema & import scripts:

```cypher
CREATE CONSTRAINT FOR (c:Customer) ASSERT c.id IS UNIQUE;
CREATE CONSTRAINT FOR (o:Order)    ASSERT o.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Product)  ASSERT p.id IS UNIQUE;
CREATE CONSTRAINT FOR (r:Region)   ASSERT r.name IS UNIQUE;
CREATE CONSTRAINT FOR (m:Manager)  ASSERT m.name IS UNIQUE;
CREATE INDEX FOR (p:Product) ON (p.category);

// Managers & Regions
:source cypher/import_people.cypher

// Customers & Orders
:source cypher/import_orders.cypher

// Products & CONTAINS
:source cypher/import_products.cypher
```  

---

## 3️⃣ Exploratory Queries

```cypher
// Top Products by Sales
MATCH (p:Product)<-[c:CONTAINS]-()
RETURN p.name AS Product, sum(c.sales) AS TotalSales
ORDER BY TotalSales DESC LIMIT 10;

// Top Customers by Profit
MATCH (c:Customer)-[:PLACED]->(o:Order)
RETURN c.name AS Customer, sum(o.profit) AS TotalProfit
ORDER BY TotalProfit DESC LIMIT 10;

// Return Rates by Region
MATCH (r:Region)<-[:SHIPPED_TO]-(o:Order)
RETURN r.name AS Region,
       round(sum(CASE WHEN o.returned THEN 1 ELSE 0 END)*100.0/count(o),2) AS ReturnRatePct
ORDER BY ReturnRatePct DESC;

// Avg Shipping Delay by Mode
MATCH (o:Order)
RETURN o.shipMode AS Mode,
       round(avg(duration.inDays(o.orderDate, o.shipDate).days),2) AS AvgDelayDays
ORDER BY AvgDelayDays;
```  

---

## 4️⃣ Graph Algorithms (GDS)

```cypher
CALL gds.graph.project(
  'ecomGraph',
  ['Customer','Order','Product'],
  { PLACED:{}, CONTAINS:{}, SHIPPED_TO:{} }
);
CALL gds.pageRank.write(
  'ecomGraph',
  { nodeProjection:'Product', relationshipWeightProperty:'sales', writeProperty:'pr_score' }
);
CALL gds.louvain.write(
  'ecomGraph',
  { nodeLabels:['Product'], relationshipTypes:['CONTAINS'], writeProperty:'communityId' }
);
```  

---

## 5️⃣ ML Pipeline (no‑leak LightGBM)

Run:
```bash
python scripts/ml_pipeline_noleak.py
```
- Splits train/test before computing group features  
- LightGBM with balanced class weights & hyperparameter tuning  
- Saves `visuals/roc_curve_noleak.png` & `visuals/feature_importances_noleak.png`  
- Writes back predictions to Neo4j and exports `order_return_predictions_noleak.csv`  

---

## 🎯 Results

- **Test ROC AUC:** 0.785  
- **Actual vs Predicted Returns:** 296 vs. 545 (Accuracy 94.9%)  
- **Key features:** historical return rates, shipping delay, unit price, etc.

---

## 📦 Deliverables

- Cleaned CSVs: `import/*.csv`  
- Cypher scripts: `cypher/*.cypher`  
- ML scripts: `scripts/*.py`  
- Visuals: `visuals/*.png`  
- Slide deck: `eda_presentation.pptx`  
- **README.md**  

---

## 🙋‍♂️ Questions?

Feel free to propose enhancements like threshold tuning, confusion‑matrix analysis, or a live PyVis dashboard!

