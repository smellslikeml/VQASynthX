import json


class ExplainableReasoningGraph:
    """
    Generates a reasoning graph with self-explaining nodes, inspired by X-Node.

    In the X-Node framework, each node in a graph can generate an explanation for
    its classification based on its topological features. We adapt this concept
    to the VQASynth pipeline by treating a reasoning chain (like Chain-of-Thought)
    as a directed graph. Each node in this graph represents a step in the reasoning
    process and includes a self-explanation justifying its purpose within the plan.

    This enriches the synthetic VQA data by not only providing the reasoning steps
    but also the "meta-reasoning" behind why each step is necessary.
    """

    def __init__(self, scene_description, question):
        """
        Initializes the generator with a scene and a question.

        Args:
            scene_description (dict): A dictionary describing objects in the scene.
            question (str): The spatial reasoning question.
        """
        self.scene = scene_description
        self.question = question
        self.reasoning_graph = []

    def generate_plan(self):
        """
        Generates the sequence of reasoning steps (nodes) for the graph.
        This is a simplified planner for demonstration purposes.
        """
        # Node 1: Deconstruct the query
        self.add_step(
            "Deconstruct Query",
            "Identify key objects ('red forklift', 'brown cardboard boxes') and the spatial relation ('left side of').",
            "The first step in any VQA task is to parse the natural language query to understand the user's intent and identify the entities and relationships involved.",
        )

        # Node 2: Localize the first object
        self.add_step(
            "Localize Object 1",
            "Find the 'red forklift' in the scene description. Found at coordinates [150, 300].",
            "Grounding the identified objects in the visual context is necessary before their spatial relationship can be evaluated. This node corresponds to object detection/segmentation.",
        )

        # Node 3: Localize the second object
        self.add_step(
            "Localize Object 2",
            "Find the 'brown cardboard boxes' in the scene description. Found at coordinates [400, 310].",
            "Similarly, the second object must be grounded in the scene to provide a reference for the spatial comparison.",
        )

        # Node 4: Evaluate the spatial relationship
        self.add_step(
            "Evaluate Spatial Predicate",
            "Compare the x-coordinate of the forklift (150) with the boxes (400). Since 150 < 400, the forklift is to the left.",
            "This node applies the specific spatial logic requested in the query. It uses the features (coordinates) extracted from the previously localized objects to perform the comparison.",
        )

        # Node 5: Formulate the final answer
        self.add_step(
            "Synthesize Answer",
            "Construct the final answer based on the evaluation: 'Yes, the red forklift appears on the left side of the brown cardboard boxes.'",
            "The final step translates the logical conclusion from the evaluation into a coherent, human-readable natural language response.",
        )

    def add_step(self, name, description, explanation):
        """
        Adds a self-explaining node to the reasoning graph.

        Args:
            name (str): The high-level name of the reasoning step.
            description (str): A description of the action taken in this step.
            explanation (str): The justification for why this step is necessary.
        """
        node = {
            "step_id": len(self.reasoning_graph) + 1,
            "name": name,
            "description": description,
            "explanation": explanation,
        }
        self.reasoning_graph.append(node)

    def to_json(self):
        """Returns the final reasoning graph as a JSON string."""
        output = {
            "question": self.question,
            "scene": self.scene,
            "reasoning_graph": self.reasoning_graph,
            "final_answer": (
                self.reasoning_graph[-1]["description"].split("'")[1]
                if self.reasoning_graph
                else "N/A"
            ),
        }
        return json.dumps(output, indent=2)


def main():
    """
    Main function to run the demonstration.
    """
    # 1. Define a sample scene (simplified from VQASynth's complex 3D data)
    scene_description = {
        "image_id": "warehouse_sample_1.jpeg",
        "objects": [
            {"name": "red forklift", "coords": [150, 300]},
            {"name": "brown cardboard boxes", "coords": [400, 310]},
            {"name": "man in red hat", "coords": [500, 250]},
        ],
    }

    # 2. Define a spatial reasoning question
    question = "Does the red forklift in warehouse appear on the left side of the brown cardboard boxes stacked?"

    # 3. Generate the explainable reasoning graph
    explainer = ExplainableReasoningGraph(scene_description, question)
    explainer.generate_plan()

    # 4. Print the resulting JSON-formatted data
    # This output could be used to fine-tune a VLM.
    print(explainer.to_json())


if __name__ == "__main__":
    main()
