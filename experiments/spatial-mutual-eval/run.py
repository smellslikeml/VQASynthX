import os
import base64
from PIL import Image
from io import BytesIO
import json

# --- Placeholder Model Integration ---
# In a real scenario, this would integrate with the project's model loading mechanism.
# For this experiment, we'll simulate two distinct models (e.g., base vs. fine-tuned).
class MockVLM:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Initialized mock VLM: {self.model_name}")

    def ask(self, image_b64, prompt):
        """Simulates a VLM generating a response to a prompt."""
        print(f"\n--- Querying {self.model_name} ---\nPrompt: {prompt[:100]}...")
        # In a real implementation, this would be an API call.
        # Here, we return a canned response for demonstration.
        if "generate a question" in prompt:
            return json.dumps({
                "question": f"Based on the image, how far is the object on the left from the object on the right? Provide a step-by-step reasoning for your estimation.",
                "reasoning": "I am acting as the Questioner. I will create a question that requires metric spatial reasoning and justification."
            })
        elif "evaluate the following answer" in prompt:
            return json.dumps({
                "evaluation_score": 8,
                "evaluation_reasoning": f"The answer from the other model was plausible but lacked a detailed breakdown of its estimation process. It correctly identified the objects but the distance was a rough guess. A score of 8/10 is appropriate.",
                "is_correct": True
            })
        else: # Answering the question
            return json.dumps({
                "answer": "The object on the left is approximately 3 meters from the object on the right.",
                "reasoning": f"As {self.model_name}, I estimate the distance based on the relative size of the objects and perspective cues in the image."
            })

# --- Prompts inspired by LLM-Crowdsourced methodology ---

QUESTION_GENERATION_PROMPT = """
You are an expert in visual spatial reasoning. Your task is to act as a "Questioner".
Given the image, generate one challenging, open-ended question that tests a model's ability to understand spatial relationships, distances, or orientations.
The question should require reasoning and not be a simple object identification task.
Output your response as a JSON object with two keys: "question" and "reasoning" (explaining why you chose this question).
"""

ANSWERING_PROMPT_TEMPLATE = """
You are an expert AI assistant. Your task is to act as an "Answerer".
Given the image and the following question, provide a concise and accurate answer.
Include your step-by-step reasoning to justify your answer.
Output your response as a JSON object with two keys: "answer" and "reasoning".

Question: {question}
"""

EVALUATION_PROMPT_TEMPLATE = """
You are an expert AI assistant. Your task is to act as a "Judge".
You will evaluate an answer to a question based on an image.
Your goal is to assess the quality of the reasoning and the correctness of the final answer.
- Review the original question.
- Review the provided answer and its reasoning.
- Provide a score from 1 (poor) to 10 (excellent).
- Provide a justification for your score.

Output your response as a JSON object with three keys: "evaluation_score" (int), "evaluation_reasoning" (string), and "is_correct" (boolean).

Original Question: {question}
Answer to Evaluate: {answer}
"""

# --- Helper Functions ---
def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- Main Experiment Logic ---
def run_mutual_evaluation(image_path, model_a, model_b):
    """
    Orchestrates the mutual evaluation flow inspired by LLM-Crowdsourced.
    1. Model A (Questioner) asks a question about the image.
    2. Model B (Answerer) answers the question.
    3. Model A (Judge) evaluates Model B's answer.
    """
    print(f"Starting mutual evaluation with image: {image_path}")
    print("="*80)

    # 1. Image Preparation
    image_b64 = encode_image(image_path)

    # 2. Model A generates a question
    print("\nStep 1: Model A generates a spatial question.")
    q_gen_response_str = model_a.ask(image_b64, QUESTION_GENERATION_PROMPT)
    q_gen_response = json.loads(q_gen_response_str)
    question = q_gen_response.get("question")
    print(f"Generated Question: {question}")
    print(f"Questioner's Reasoning: {q_gen_response.get('reasoning')}")
    print("="*80)

    # 3. Model B answers the question
    print("\nStep 2: Model B answers the question.")
    answer_prompt = ANSWERING_PROMPT_TEMPLATE.format(question=question)
    answer_response_str = model_b.ask(image_b64, answer_prompt)
    answer_response = json.loads(answer_response_str)
    answer = answer_response.get("answer")
    print(f"Answer: {answer}")
    print(f"Answerer's Reasoning: {answer_response.get('reasoning')}")
    print("="*80)


    # 4. Model A evaluates Model B's answer
    print("\nStep 3: Model A evaluates Model B's answer.")
    evaluation_prompt = EVALUATION_PROMPT_TEMPLATE.format(question=question, answer=json.dumps(answer_response))
    eval_response_str = model_a.ask(image_b64, evaluation_prompt)
    eval_response = json.loads(eval_response_str)
    print(f"Evaluation Score (1-10): {eval_response.get('evaluation_score')}")
    print(f"Evaluation Reasoning: {eval_response.get('evaluation_reasoning')}")
    print("="*80)

    # 5. Log results
    results = {
        "image": image_path,
        "questioner_model": model_a.model_name,
        "answerer_model": model_b.model_name,
        "question_generation": q_gen_response,
        "answer": answer_response,
        "evaluation": eval_response
    }

    output_filename = "spatial_mutual_eval_results.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment complete. Results saved to {output_filename}")

if __name__ == "__main__":
    # This experiment requires a sample image.
    # We assume one exists at 'assets/warehouse_sample_1.jpeg' relative to the project root.
    sample_image_path = "assets/warehouse_sample_1.jpeg"
    if not os.path.exists(sample_image_path):
        # Create a dummy image if it doesn't exist to make the script runnable.
        print(f"Warning: Sample image not found at {sample_image_path}. Creating a dummy image.")
        try:
            os.makedirs(os.path.dirname(sample_image_path), exist_ok=True)
            dummy_image = Image.new('RGB', (100, 100), color = 'red')
            dummy_image.save(sample_image_path)
        except Exception as e:
            print(f"Error creating dummy image: {e}")
            print("Please provide a sample image at 'assets/warehouse_sample_1.jpeg' to run.")
            exit(1)

    # Initialize the two models to be compared.
    # model_a could be a baseline VLM
    # model_b could be a VLM fine-tuned with VQASynth data
    model_a = MockVLM(model_name="Baseline-VLM")
    model_b = MockVLM(model_name="VQASynth-Finetuned-VLM")

    run_mutual_evaluation(sample_image_path, model_a, model_b)
