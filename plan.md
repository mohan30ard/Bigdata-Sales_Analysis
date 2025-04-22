end‑to‑end plan for your Orders–People–Returns EDA in Neo4j 5 + GDS Community, with Python‑powered cleaning & visualization. It follows the structure of your reference tutorial and incorporates all the fixes we applied.

---

## 1. Data Cleaning (Jupyter Notebook)

1. **Environment setup**
   - Create a new conda/venv environment.
   - Install:
     ```bash
     pip install pandas numpy openpyxl neo4j matplotlib seaborn networkx pyvis
     ```

2. **Load & inspect raw CSVs**
   ```python
   import pandas as pd
   orders = pd.read_csv('orders.csv')
   people = pd.read_csv('people.csv')
   returns = pd.read_csv('returns.csv')

   orders.info()
   orders.isna().sum()
   ```

3. **Cleaning transformations**
   - **Dates:**
     ```python
     orders['order_date'] = pd.to_datetime(orders['Order Date'], errors='coerce')
     orders['ship_date']  = pd.to_datetime(orders['Ship Date'],  errors='coerce')
     ```
   - **Numerics:** cast `Sales, Quantity, Discount, Profit`:
     ```python
     for c in ['Sales','Quantity','Discount','Profit']:
         orders[c.lower()] = pd.to_numeric(orders[c], errors='coerce')
     ```
   - **Nulls & duplicates:** drop rows missing `Order ID`/`Customer ID`/`Product ID`, then `drop_duplicates()`.
   - **Returns flag:**
     ```python
     ret = returns.groupby('Order ID').size().reset_index(name='returned_count')
     orders = orders.merge(ret, on='Order ID', how='left').fillna({'returned_count':0})
     ```

4. **Rename & export for Neo4j**
   ```python
   orders = orders.rename(columns={
     'Order ID':'order_id', 'Customer ID':'customer_id', 'Customer Name':'customer_name',
     'Customer Segment':'customer_segment', 'Ship Mode':'ship_mode',
     'Country/Region':'country_region', 'City':'city', 'State/Province':'state_province',
     'Postal Code':'postal_code', 'Region':'region',
     'Product ID':'product_id', 'Product Name':'product_name',
     'Category':'category', 'Sub-Category':'sub_category'
   })
   keep = [
     'order_id','order_date','ship_date','ship_mode',
     'customer_id','customer_name','customer_segment',
     'country_region','city','state_province','postal_code','region',
     'product_id','product_name','category','sub_category',
     'sales','quantity','discount','profit','returned_count'
   ]
   orders_clean = orders[keep]
   orders_clean.to_csv('orders_clean.csv', index=False)
   people.to_csv('people_clean.csv', index=False)
   ```

---

## 2. Neo4j Desktop Setup & Schema

1. **Neo4j Desktop** & **Import folder**
   - Copy `orders_clean.csv`, `people.csv` to `import/`.
2. **Plugins**
   - Enable **APOC 5.24.2** and **GDS 2.12.0** under **Plugins** → **Manage**.
3. **Constraints & Indexes**
   ```cypher
   CREATE CONSTRAINT customer_id_unique IF NOT EXISTS
     FOR (c:Customer) REQUIRE c.id IS UNIQUE;
   CREATE CONSTRAINT order_id_unique IF NOT EXISTS
     FOR (o:Order)   REQUIRE o.id IS UNIQUE;
   CREATE CONSTRAINT product_id_unique IF NOT EXISTS
     FOR (p:Product) REQUIRE p.id IS UNIQUE;
   CREATE CONSTRAINT region_name_unique IF NOT EXISTS
     FOR (r:Region)  REQUIRE r.name IS UNIQUE;
   CREATE CONSTRAINT manager_name_unique IF NOT EXISTS
     FOR (m:Manager) REQUIRE m.name IS UNIQUE;
   CREATE INDEX product_category_index IF NOT EXISTS
     FOR (p:Product) ON (p.category);
   ```

_No need to edit `neo4j.conf` when importing in Neo4j Browser._

---

## 3. Import Cleaned Data via Cypher

### 3.1 Managers & Regions (`people.csv`)
```cypher
LOAD CSV WITH HEADERS FROM 'file:///people.csv' AS row
WITH row
WHERE row.`Regional Manager` IS NOT NULL AND row.`Regional Manager` <> ''
MERGE (m:Manager {name: row.`Regional Manager`})
MERGE (r:Region  {name: row.Region})
MERGE (m)-[:MANAGES]->(r);
```

### 3.2 Customers & Orders (`orders_clean.csv`)
```cypher
LOAD CSV WITH HEADERS FROM 'file:///orders_clean.csv' AS row
WITH row,
     date(row.order_date)               AS od,
     date(row.ship_date)                AS sd,
     toFloat(row.sales)                 AS sales,
     toInteger(row.quantity)            AS qty,
     toFloat(row.discount)              AS disc,
     toFloat(row.profit)                AS prof,
     (toInteger(row.returned_count)>0)  AS returned_flag
WHERE row.order_id IS NOT NULL

MERGE (c:Customer {id: row.customer_id})
  ON CREATE SET c.name = row.customer_name,
                c.segment = row.customer_segment
MERGE (o:Order {id: row.order_id})
  ON CREATE SET
    o.orderDate = od,
    o.shipDate  = sd,
    o.shipMode  = row.ship_mode,
    o.sales     = sales,
    o.quantity  = qty,
    o.discount  = disc,
    o.profit    = prof,
    o.returned  = returned_flag
MERGE (c)-[:PLACED]->(o)
WITH row, o
MATCH (r:Region {name: row.region})
MERGE (o)-[:SHIPPED_TO]->(r);
```

### 3.3 Products & CONTAINS
```cypher
LOAD CSV WITH HEADERS FROM 'file:///orders_clean.csv' AS row
WITH row
WHERE row.product_id IS NOT NULL

MERGE (p:Product {id: row.product_id})
  ON CREATE SET
    p.name        = row.product_name,
    p.category    = row.category,
    p.subCategory = row.sub_category
WITH row, p
MATCH (o:Order {id: row.product_id})  // typo fixed below
MERGE (o)-[:CONTAINS {quantity: toInteger(row.quantity), sales: toFloat(row.sales)}]->(p);
```  
*Fix:* should match on `row.order_id`, not `row.product_id`.

---

## 4. Exploratory Cypher Queries

**4.1 Top 10 Products by Sales**
```cypher
MATCH (:Order)-[c:CONTAINS]->(p:Product)
RETURN p.name AS Product, sum(c.sales) AS TotalSales
ORDER BY TotalSales DESC LIMIT 10;
```

**4.2 Top 10 Customers by Profit**
```cypher
MATCH (c:Customer)-[:PLACED]->(o:Order)
RETURN c.name AS Customer, sum(o.profit) AS TotalProfit
ORDER BY TotalProfit DESC LIMIT 10;
```

**4.3 Return Rates by Region**
```cypher
MATCH (r:Region)<-[:SHIPPED_TO]-(o:Order)
WITH r.name AS Region, count(o) AS Total, sum(toInteger(o.returned)) AS Ret
RETURN Region, Total, Ret,
       round(Ret*1.0/Total*100,2) AS ReturnRatePct
ORDER BY ReturnRatePct DESC;
```

**4.4 Avg Shipping Delay by Mode**
```cypher
MATCH (o:Order)
WITH o.shipMode AS Mode,
     avg(duration.inDays(o.orderDate,o.shipDate).days) AS AvgDelay
RETURN Mode, round(AvgDelay,2) AS AvgDelayDays
ORDER BY AvgDelayDays;
```

---

## 5. Graph Algorithms (GDS)

### 5.1 Product PageRank
```cypher
CALL gds.graph.project(
  'productGraph', ['Order','Product'],
  { CONTAINS:{type:'CONTAINS',orientation:'UNDIRECTED', properties:['sales']} }
)
YIELD graphName,nodeCount,relationshipCount;
CALL gds.pageRank.write(
  'productGraph',{writeProperty:'pr_score',relationshipWeightProperty:'sales'}
)
YIELD nodePropertiesWritten,ranIterations;
```

### 5.2 Product Co‑purchase Louvain
```cypher
CALL gds.graph.project.cypher(
  'prodCoGraph',
  'MATCH (p:Product) RETURN id(p) AS id',
  '
    MATCH (o:Order)-[:CONTAINS]->(p1:Product),
          (o)-[:CONTAINS]->(p2:Product)
    WHERE id(p1)<id(p2)
    RETURN id(p1) AS source,id(p2) AS target, count(*) AS weight
  '
)
YIELD graphName,nodeCount,relationshipCount;
CALL gds.louvain.write(
  'prodCoGraph',{relationshipWeightProperty:'weight',writeProperty:'communityId'}
)
YIELD communityCount,modularity;
```

---

## 6. Python Integration & Visuals

1. **Env**
   ```bash
   conda create -n neo4j-eda python=3.10 pandas neo4j matplotlib seaborn networkx pyvis
   ```

2. **Connect & query**
   ```python
   from neo4j import GraphDatabase
   driver = GraphDatabase.driver('bolt://localhost:7687',auth=('neo4j','pwd'))
   ```

3. **DataFrames & plots**
   - Top 10 products by `pr_score` (bar chart)
   - Product cluster sizes from Louvain (bar chart)
   - Optional interactive with PyVis:
     ```python
     from pyvis.network import Network
     net = Network()
     # load nodes/edges from Neo4j and build network
     ```

---

## 7. Deliverables

- **EDA_Tutorial.md** (this outline + code + screenshots)
- **eda_cleaning.ipynb** (cleaning steps)
- **import_orders.cypher**, **gds_algos.cypher**
- **visualize_eda.py**
- **Charts PNGs** & **network.html**

---

Let me know if any section needs further correction or expansion!

