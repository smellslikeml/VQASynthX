import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# This represents a simplified, structured output from a VQASynth scene analysis pipeline.
# It identifies objects and their spatial relationships within a single image.
SCENE_DATA = {
    "nodes": [
        {"id": "obj1", "label": "red forklift"},
        {"id": "obj2", "label": "brown boxes"},
        {"id": "obj3", "label": "man in red hat"},
        {"id": "obj4", "label": "wooden pallet"},
    ],
    "relationships": [
        {"source": "obj1", "target": "obj2", "type": "NEAR"},
        {"source": "obj1", "target": "obj4", "type": "ADJACENT_TO"},
        {"source": "obj2", "target": "obj4", "type": "ON_TOP_OF"},
        {"source": "obj3", "target": "obj4", "type": "NEAR"},
    ],
}


class SceneGraphExperiment:
    """
    Connects to Neo4j to run a graph reasoning experiment on scene data.
    The data loading and execution pattern is inspired by `import_data.py`
    and the GDS tooling from the `neo4j-contrib/gds-agent` repository.
    """

    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logging.info("Successfully connected to Neo4j.")
        except Exception as e:
            logging.error(
                f"Failed to connect to Neo4j at {uri}. Please check credentials and database status."
            )
            raise e

    def close(self):
        self.driver.close()

    def setup_scene_graph(self, scene_data):
        """Cleans the database and loads the scene data."""
        with self.driver.session() as session:
            logging.info("Cleaning database for a fresh start...")
            session.run("MATCH (n) DETACH DELETE n")

            logging.info("Creating constraints...")
            session.run(
                "CREATE CONSTRAINT scene_object_id IF NOT EXISTS FOR (o:SceneObject) REQUIRE o.id IS UNIQUE"
            )

            logging.info(f"Loading {len(scene_data['nodes'])} scene objects...")
            session.run(
                """
            UNWIND $nodes AS node_data
            CREATE (o:SceneObject {id: node_data.id, label: node_data.label})
            """,
                nodes=scene_data["nodes"],
            )

            logging.info(
                f"Loading {len(scene_data['relationships'])} spatial relationships..."
            )
            session.run(
                """
            UNWIND $relationships AS rel_data
            MATCH (source:SceneObject {id: rel_data.source})
            MATCH (target:SceneObject {id: rel_data.target})
            CREATE (source)-[:SPATIALLY_RELATED {type: rel_data.type}]->(target)
            """,
                relationships=scene_data["relationships"],
            )
            logging.info("Scene graph setup complete.")

    def run_centrality_analysis(self):
        """
        Runs GDS Degree Centrality to find the most 'central' or 'connected'
        object in the scene based on its number of direct relationships.
        This demonstrates deriving higher-order insights from the graph.
        """
        graph_name = "scene_graph_projection"
        with self.driver.session() as session:
            logging.info("Running GDS Degree Centrality analysis...")

            # Drop the projected graph if it exists, for idempotency
            session.run(
                f"""
            CALL gds.graph.exists('{graph_name}') YIELD exists
            WHERE exists
            CALL gds.graph.drop('{graph_name}') YIELD graphName
            RETURN graphName
            """
            )

            # Project the graph into GDS's in-memory store
            project_query = f"""
            CALL gds.graph.project(
                '{graph_name}',
                'SceneObject',
                'SPATIALLY_RELATED',
                {{
                    orientation: 'UNDIRECTED'
                }}
            )
            """
            result = session.run(project_query)
            projection_info = result.single()
            if not projection_info:
                raise RuntimeError(
                    "Graph projection failed. Ensure GDS is installed and the graph is loaded correctly."
                )
            logging.info(
                f"Projected graph '{projection_info['graphName']}' into memory."
            )

            # Run the Degree Centrality algorithm and stream the results
            stream_query = f"""
            CALL gds.degree.stream('{graph_name}')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).label AS object_label, score
            ORDER BY score DESC
            """
            result = session.run(stream_query)

            centrality_results = [
                {"object": record["object_label"], "score": record["score"]}
                for record in result
            ]
            logging.info("Centrality analysis complete.")
            return centrality_results


def main():
    """Main execution function."""
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise ValueError(
            "Missing Neo4j environment variables. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in a .env file or the environment."
        )

    try:
        experiment = SceneGraphExperiment(uri, user, password)
        experiment.setup_scene_graph(SCENE_DATA)
        centrality_scores = experiment.run_centrality_analysis()

        print("\n" + "=" * 50)
        print(" VQASynth Graph Reasoning Experiment: Centrality")
        print("=" * 50)
        print("Identifies the most connected objects in the scene graph.\n")

        for record in centrality_scores:
            print(
                f"  - Object: '{record['object']}', Connectivity Score: {int(record['score'])}"
            )

        print("\n" + "-" * 50)
        if centrality_scores:
            most_central = centrality_scores[0]
            print(
                f"Key Insight: The '{most_central['object']}' is the most central object in this scene."
            )
            print("This insight could fuel more complex VQA data generation, like:")
            print(
                " -> 'Describe the objects immediately surrounding the wooden pallet.'"
            )
        print("=" * 50)

    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
    finally:
        if "experiment" in locals() and experiment.driver:
            experiment.close()
            logging.info("Neo4j connection closed.")


if __name__ == "__main__":
    main()
