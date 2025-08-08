"""
Experiment runner to demonstrate VQA generation from a SceneBeliefState.
This script showcases how to use the belief state representation to generate
questions and answers that deal with uncertainty.
"""
import random
import os
import sys

# Add vqasynth to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqasynth.belief import BeliefObject, SceneBeliefState

def generate_uncertainty_vqa(scene: SceneBeliefState):
    """Generates sample VQA pairs based on the scene's belief state."""
    
    if not scene.objects:
        print("Scene is empty. No VQA pairs to generate.")
        return

    vqa_pairs = []

    # Select a random object to ask about
    obj_id = random.choice(list(scene.objects.keys()))
    obj = scene.get_object(obj_id)
    
    # --- VQA Pair 1: Ask for the most likely identity ---
    q1 = f"What is object {obj.obj_id} most likely to be?"
    most_likely_label, prob = obj.get_most_likely_label()
    a1 = f"Object {obj.obj_id} is most likely a '{most_likely_label}' with {prob:.0%} confidence."
    vqa_pairs.append({"question": q1, "answer": a1})

    # --- VQA Pair 2: Ask for the probability of a specific label ---
    # Choose a label from the distribution, not necessarily the most likely one
    if len(obj.label_probabilities) > 1:
        chosen_label = random.choice(list(obj.label_probabilities.keys()))
        prob_chosen = obj.label_probabilities[chosen_label]
        q2 = f"What is the likelihood that object {obj.obj_id} is a '{chosen_label}'?"
        a2 = f"The probability that object {obj.obj_id} is a '{chosen_label}' is {prob_chosen:.0%}."
        vqa_pairs.append({"question": q2, "answer": a2})

    # --- VQA Pair 3: Ask for a summary of possibilities (Chain-of-Thought style) ---
    q3 = f"Describe the possible identities for object {obj.obj_id}."
    
    sorted_labels = sorted(obj.label_probabilities.items(), key=lambda item: item[1], reverse=True)
    
    thought = f"Let's analyze the possibilities for object {obj.obj_id}. "
    thought += f"The primary candidate is '{sorted_labels[0][0]}' ({sorted_labels[0][1]:.0%}). "
    if len(sorted_labels) > 1:
        thought += f"However, there is also a possibility it could be a '{sorted_labels[1][0]}' ({sorted_labels[1][1]:.0%}). "
    
    answer_parts = [f"a '{label}' ({prob:.0%})" for label, prob in sorted_labels]
    a3 = thought + f"Based on the analysis, object {obj.obj_id} could be " + ", or ".join(answer_parts) + "."
    vqa_pairs.append({"question": q3, "answer": a3})

    return vqa_pairs


def main():
    """Main function to run the experiment demonstration."""
    print("--- Probabilistic Scene Representation Experiment ---")
    
    # Simulate a scene with uncertainty. This could come from a real model's output.
    # Object 1 is likely a 'cat' but could be a 'small dog'.
    obj1 = BeliefObject(obj_id=1, label_probabilities={'cat': 0.8, 'small dog': 0.2})
    
    # Object 2 is highly certain to be a 'table'.
    obj2 = BeliefObject(obj_id=2, label_probabilities={'table': 0.99, 'desk': 0.01})
    
    # Object 3 has high ambiguity between 'cup' and 'mug'.
    obj3 = BeliefObject(obj_id=3, label_probabilities={'cup': 0.55, 'mug': 0.45})

    scene = SceneBeliefState(objects=[obj1, obj2, obj3])
    print(f"Created Scene: {scene}\n")

    # Generate and print VQA pairs for a random object
    print("--- Generating Sample VQA Pairs ---")
    generated_pairs = generate_uncertainty_vqa(scene)
    for i, pair in enumerate(generated_pairs):
        print(f"\nSample {i+1}:")
        print(f" Q: {pair['question']}")
        print(f" A: {pair['answer']}")


if __name__ == "__main__":
    main()
