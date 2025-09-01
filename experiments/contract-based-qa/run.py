import os
from symai import Expression
from symai.strategy import contract
from symai.backend.engines.neurosymbolic.engine_openai_gptX_chat import GPTXChatEngine
from symai.models import LLMDataModel
from pydantic import Field, field_validator

# This check is to ensure the script can run in environments without an API key
# by using a mock engine. For a real test, an API key is required.
IS_DEMO_MODE = "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]


# Define a mock engine for demonstration purposes if no API key is found
class MockEngine(GPTXChatEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def forward(self, *args, **kwargs):
        self.call_count += 1
        # First call: Simulate a non-compliant answer to trigger remediation
        if self.call_count == 1:
            return [
                '{"question": "Is the cat on the mat?", "answer": "The cat is indeed on the mat."}'
            ]
        # Second call (remedy): Simulate a compliant answer
        else:
            return ['{"question": "Is the cat on the mat?", "answer": "yes"}']


# Data model for the VQA pair, using Pydantic-style validation.
# This ensures the structure and content of the LLM's output.
class VQAPair(LLMDataModel):
    question: str = Field(description="A clear, concise question about the scene.")
    answer: str = Field(description="A 'yes' or 'no' answer to the question.")

    @field_validator("answer")
    def validate_answer(cls, v):
        # The contract's post-condition will rely on this validator.
        # If the LLM output doesn't conform, the contract will trigger a remedy.
        v_lower = v.lower().strip()
        if v_lower not in ["yes", "no"]:
            raise ValueError(f"Answer must be 'yes' or 'no', but got '{v}'.")
        return v_lower


# The Agent class decorated with a contract.
# This expression is responsible for generating the VQA pair.
@contract(
    # The contract will attempt to fix the LLM's output if it fails validation.
    post_remedy=True,
    # This provides detailed logs of the contract's operations.
    verbose=True,
    # Configure retry attempts for the remediation process.
    remedy_retry_params=dict(tries=3, delay=0.5, backoff=2),
)
class QA_Agent(Expression):
    # This is the static prompt that defines the agent's task.
    # It references the VQAPair model to guide the LLM's output format.
    prompt: str = f"""
    Based on the provided scene description, generate a single VQA pair.
    The output must be a JSON object matching this structure:
    {VQAPair.schema_prompt()}
    """

    # The main method that executes the expression.
    # It is expected to return an instance of the specified data model.
    def forward(self, scene_description: str) -> VQAPair:
        # The input to the prompt template.
        self.prompt += f"Scene: {scene_description}"
        # `super().forward()` triggers the LLM call.
        # The output is automatically cast to the VQAPair model.
        # If casting or validation fails, the contract's `post_remedy` is invoked.
        return super().forward()


def main():
    print("--- Running VQA Generation Experiment with SymbolicAI Contracts ---")
    print(
        f"DEMO MODE: {'Enabled (using mock engine)' if IS_DEMO_MODE else 'Disabled (using real OpenAI API)'}\n"
    )

    # Configure the engine
    engine = MockEngine() if IS_DEMO_MODE else GPTXChatEngine(model="gpt-3.5-turbo")

    # Instantiate the agent with the chosen engine
    agent = QA_Agent(output_format=VQAPair, engine=engine)

    # A simple scene description to seed the VQA generation
    scene = "A black cat is sleeping on a red mat in front of a fireplace."

    print(f"Scene Description:\n'{scene}'\n")
    print("--- Invoking Agent ---")

    # Execute the agent. The contract will handle validation and remediation.
    try:
        result = agent(scene)
        print("\n--- Agent Execution Complete ---")
        print(f"Successfully generated VQAPair:")
        print(f"  Question: {result.question}")
        print(f"  Answer: {result.answer} (Validated)")
    except Exception as e:
        print(f"\n--- Agent Execution Failed ---")
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
