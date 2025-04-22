from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt

# ─── Neo4j connection parameters ──────────────────────────
uri      = "bolt://localhost:7687"
user     = "neo4j"
password = "password"  
# ─────────────────────────────────────────────────────────

# 1) Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    # 2) Drop existing prodCoGraph if present
    session.run("CALL gds.graph.drop('prodCoGraph', false)")

    # 3) Project the product co‑purchase graph
    session.run("""
        CALL gds.graph.project.cypher(
          'prodCoGraph',
          'MATCH (p:Product) RETURN id(p) AS id',
          '
            MATCH (o:Order)-[:CONTAINS]->(p1:Product),
                  (o)-[:CONTAINS]->(p2:Product)
            WHERE id(p1) < id(p2)
            RETURN id(p1) AS source,
                   id(p2) AS target,
                   count(*) AS weight
          '
        )
    """)

    # 4) Run Louvain community detection
    session.run("""
        CALL gds.louvain.write(
          'prodCoGraph',
          {
            relationshipWeightProperty: 'weight',
            writeProperty: 'communityId'
          }
        )
    """)

    # 5) Fetch top 10 clusters by size
    result = session.run("""
        MATCH (p:Product)
        WHERE p.communityId IS NOT NULL
        RETURN p.communityId AS cluster, count(p) AS size
        ORDER BY size DESC
        LIMIT 10
    """)
    cluster_df = pd.DataFrame(result.data())

driver.close()

# 6) Plot horizontal bar chart of top 10 product clusters
plt.figure(figsize=(8, 5))
plt.barh(cluster_df['cluster'].astype(str)[::-1],
         cluster_df['size'][::-1])
for idx, (cluster, size) in enumerate(zip(
        cluster_df['cluster'][::-1],
        cluster_df['size'][::-1]
    )):
    plt.text(size + 2, idx, str(size), va='center')
plt.xlabel('Number of Products')
plt.ylabel('Cluster ID')
plt.title('Top 10 Product Co‑purchase Clusters')
plt.tight_layout()
plt.show()
