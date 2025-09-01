import optuna
import random
import json

# This is a mock representation of scene data that the VQASynth pipeline would produce.
# In a real integration, this would be loaded from an intermediate file.
MOCK_SCENE_DATA = {
    "image_id": "warehouse_sample_1.jpeg",
    "objects": [
        {"id": 1, "label": "red forklift", "position": [1.2, 0.5, 3.4]},
        {"id": 2, "label": "brown cardboard boxes", "position": [3.5, 0.8, 3.6]},
        {"id": 3, "label": "man in red hat", "position": [5.0, 1.0, 7.2]},
    ],
}


def generate_vqa_pair(scene_data, question_type, distance_format, use_colloquialisms):
    """
    A simplified mock of the vqasynth prompt generation stage.
    This function generates a single VQA pair based on the input parameters.
    """
    objects = scene_data["objects"]
    if len(objects) < 2:
        return None, None

    obj1, obj2 = random.sample(objects, 2)

    # Calculate horizontal distance for demonstration
    dist = (
        (obj1["position"][0] - obj2["position"][0]) ** 2
        + (obj1["position"][2] - obj2["position"][2]) ** 2
    ) ** 0.5

    question = ""
    answer = ""

    if question_type == "distance":
        if distance_format == "metric":
            dist_str = f"{dist:.2f} meters"
        else:  # imperial
            dist_str = f"{dist * 3.281:.2f} feet"

        if use_colloquialisms and dist < 2.0:
            proximity = "very close to"
        elif use_colloquialisms:
            proximity = "some distance from"
        else:
            proximity = "away from"

        question = f"How far is the {obj1['label']} from the {obj2['label']}?"
        answer = f"The {obj1['label']} is approximately {dist_str} {proximity} the {obj2['label']}."

    elif question_type == "relative_position":
        # Simple left/right based on x-coordinate
        if obj1["position"][0] < obj2["position"][0]:
            position_desc = "to the left of"
        else:
            position_desc = "to the right of"
        question = f"Where is the {obj1['label']} in relation to the {obj2['label']}?"
        answer = f"The {obj1['label']} is {position_desc} the {obj2['label']}."

    return {"question": question, "answer": answer}


def objective(trial):
    """
    The objective function for Optuna, driven by human feedback.
    """
    # 1. Define the hyperparameter search space
    params = {
        "question_type": trial.suggest_categorical(
            "question_type", ["distance", "relative_position"]
        ),
    }

    # Conditionally define parameters that only apply to 'distance' questions
    if params["question_type"] == "distance":
        params["distance_format"] = trial.suggest_categorical(
            "distance_format", ["metric", "imperial"]
        )
        params["use_colloquialisms"] = trial.suggest_categorical(
            "use_colloquialisms", [True, False]
        )
    else:
        params["distance_format"] = None
        params["use_colloquialisms"] = None

    # 2. Generate a sample based on the parameters
    print("-" * 50)
    print(f"Trial {trial.number}: Testing parameters -> {params}")
    vqa_pair = generate_vqa_pair(MOCK_SCENE_DATA, **params)

    if not vqa_pair or not vqa_pair.get("question"):
        print("Could not generate a valid pair for these params. Assigning low score.")
        return 0.0

    print("\nGenerated VQA Pair:")
    print(json.dumps(vqa_pair, indent=2))

    # 3. Get human feedback
    while True:
        try:
            score = float(
                input(
                    "\nRate the quality and usefulness of this VQA pair (0.0 to 5.0): "
                )
            )
            if 0.0 <= score <= 5.0:
                break
            else:
                print("Invalid score. Please enter a number between 0.0 and 5.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return score


def main():
    """
    Main function to run the HITL optimization study.
    """
    print("Starting Human-in-the-Loop optimization for VQA prompt generation.")
    print("You will be shown generated VQA pairs and asked to rate their quality.")
    print("The system will use your feedback to find the best generation parameters.")

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )

    try:
        study.optimize(objective, n_trials=15)
    except KeyboardInterrupt:
        print("\nStudy interrupted by user. Displaying best results so far.")

    print("\n" + "=" * 50)
    print("Optimization finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    if study.best_trial:
        print("Best trial found:")
        trial = study.best_trial
        print(f"  Value (Score): {trial.value:.2f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No trials were completed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
