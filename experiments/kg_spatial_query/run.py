import os
import sys
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define a simple Knowledge Graph for a lab environment as a list of (subject, predicate, object) triples.
# This is inspired by the structured data approach in AGENTiGraph (e.g., test_rows_triples.csv).
LAB_KG = [
    ("robot_charger", "is_located_at", "north_west_corner"),
    ("soldering_station", "is_located_at", "workbench_3"),
    ("main_computer", "is_located_at", "main_desk"),
    ("fridge", "is_located_at", "kitchenette"),
    ("robot_charger", "is_powered_by", "wall_outlet_5"),
    ("soldering_station", "is_on_surface", "workbench_3"),
    ("main_desk", "is_near", "window"),
    ("workbench_3", "contains", "oscilloscope"),
]

# The prompt template instructs the LLM on how to perform its task, similar to the specialized agents in AGENTiGraph.
PROMPT_TEMPLATE = """
You are a helpful assistant for a robotics lab. Your task is to answer questions about the lab environment.
Refer *only* to the information provided in the Knowledge Graph (KG) below. Do not use any external knowledge.
If the answer cannot be found in the KG, state that the information is not available.

**Knowledge Graph (KG) Triples:**
{kg_triples}

**User Question:**
{question}

**Answer:**
"""


class SpatialQueryAgent:
    """A simple agent to query a knowledge graph about a lab environment.

    This class encapsulates the logic for using an LLM to interpret natural
    language questions and answer them using a predefined KG, mirroring the
    functionality of AGENTiGraph's KG Interaction and Response Generation agents.
    """

    def __init__(self, kg_data, openai_api_key=None):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required.")

        # Format the KG data into a readable string for the prompt context
        self.kg_triples_str = "\n".join([f"- {triple}" for triple in kg_data])

        # Initialize the LLM. We use "gpt-3.5-turbo-instruct" as it is a good, cost-effective model for this task,
        # consistent with the tested configuration in the AGENTiGraph Dockerfile.
        self.llm = OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            temperature=0,
            openai_api_key=openai_api_key,
        )

        # Set up the LangChain prompt and chain
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["kg_triples", "question"]
        )
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def ask(self, question: str) -> str:
        """Asks a question to the agent and returns the answer."""
        print(f"\n> Processing Query: '{question}'")
        try:
            response = self.chain.run(
                {"kg_triples": self.kg_triples_str, "question": question}
            )
            return response.strip()
        except Exception as e:
            return f"An error occurred: {e}"


def main():
    """Main function to run the experiment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: The OPENAI_API_KEY environment variable must be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("--- Initializing Spatial Query Agent ---")
    agent = SpatialQueryAgent(kg_data=LAB_KG, openai_api_key=api_key)
    print("Agent initialized with the following KG data:")
    print(agent.kg_triples_str)

    # Example queries to test the agent's reasoning capabilities
    queries = [
        "Where is the robot charging station located?",
        "What is on workbench 3?",
        "What powers the robot charger?",
        "Is the soldering station near the window?",  # Test multi-hop reasoning
        "Where is the coffee machine?",  # Test for information not in the KG
    ]

    for q in queries:
        answer = agent.ask(q)
        print(f"< Agent Answer: {answer}")


if __name__ == "__main__":
    main()
