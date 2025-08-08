import os
import openai

# --- Configuration ---
# Make sure to set your OpenAI API key as an environment variable
# export OPENAI_API_KEY='your_api_key_here'
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# --- Mock Scene Data ---
# In a real VQASynth pipeline, this would come from the scene fusion stage.
SCENE_CONTEXT = """
Scene Analysis:
- An image contains a 'red toy car' and a 'blue block'.
- The 'red toy car' is at 3D coordinates (1.0, 0.5, 2.0).
- The 'blue block' is at 3D coordinates (1.2, 0.5, 4.0).
- The distance between them is approximately 2.02 meters.
- The 'red toy car' is closer to the camera than the 'blue block'.
"""

# --- Agent Functions ---

def generate_initial_qa(scene_context):
    """
    Simulates the initial prompt stage of VQASynth.
    Generates a basic question and answer.
    """
    print("--- 1. Generating Initial QA Pair ---")
    
    prompt = f"""
Given the following scene analysis, generate a simple question and a direct answer about the spatial relationship between the objects.

Scene Analysis:
{scene_context}

Output only the question and answer in a Q: and A: format.
"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    
    content = response.choices[0].message.content
    print(f"Initial QA Pair:\n{content}\n")
    return content

def validate_qa(qa_pair):
    """
    Simulates a "Domain Agent" from ChatBattery.
    This agent critiques the generated QA pair for its quality and simplicity.
    """
    print("--- 2. Validating QA Pair (Critic Agent) ---")
    
    prompt = f"""
You are an expert in creating challenging VQA datasets. Your goal is to critique generated questions to make them better for training powerful spatial reasoning models.

Critique the following question and answer pair. Is it too simple? Does it test complex reasoning? Suggest how it could be improved. A good question might involve metric distance, relative positioning, or require a multi-step thought process.

QA Pair to Critique:
{qa_pair}

Your Critique:
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    critique = response.choices[0].message.content
    print(f"Critique:\n{critique}\n")
    return critique

def regenerate_qa_with_feedback(scene_context, initial_qa, critique):
    """
    Simulates the refinement loop from ChatBattery's `problem_conceptualization`.
    It uses the critique to generate a better QA pair.
    """
    print("--- 3. Re-generating QA with Feedback ---")
    
    # This prompt structure is inspired by ChatBattery's update logic, which
    # explicitly states what was wrong and provides guidance for improvement.
    prompt = f"""
You are a VQA data generation assistant. Your first attempt at generating a question was critiqued. Use the scene data and the critique to generate a new, improved question and answer pair that addresses the feedback.

Scene Analysis:
{scene_context}

Previous (Poor) QA Pair:
{initial_qa}

Critique of Previous Pair:
{critique}

Based on the critique, generate a new, more complex and informative question and answer. The new question should require more detailed spatial reasoning. The answer should include a chain-of-thought explanation.
Format your output as:
Q: [New Question]
A: [Chain of Thought] [Final Answer]
"""

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview", # Using a more powerful model for refinement
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    refined_qa = response.choices[0].message.content
    print(f"Refined QA Pair:\n{refined_qa}\n")
    return refined_qa

def run_experiment():
    """
    Main function to run the agentic refinement experiment.
    """
    # Step 1: Generate an initial, simple QA pair
    initial_qa_pair = generate_initial_qa(SCENE_CONTEXT)
    
    # Step 2: A "critic" agent validates the QA pair
    critique = validate_qa(initial_qa_pair)
    
    # Step 3: Use the critique to refine and regenerate the QA pair
    refined_qa_pair = regenerate_qa_with_feedback(SCENE_CONTEXT, initial_qa_pair, critique)
    
    print("--- Experiment Complete ---")
    print("Final Output:")
    print(refined_qa_pair)

if __name__ == "__main__":
    run_experiment()
