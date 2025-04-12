# app/DBServices/neo4j_writer.py

from neo4j import GraphDatabase
from app.app_config import CONFIG  # Make sure app is on PYTHONPATH

class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            CONFIG["NEO4J_URI"], 
            auth=(CONFIG["NEO4J_USER"], CONFIG["NEO4J_PASSWORD"])
        )

    def close(self):
        self.driver.close()

    def write_node(self, label, properties: dict):
        with self.driver.session() as session:
            session.write_transaction(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        props = ", ".join([f"{k}: ${k}" for k in properties])
        query = f"CREATE (n:{label} {{{props}}})"
        tx.run(query, **properties)
