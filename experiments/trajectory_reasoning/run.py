import json
import time
from typing import List, Dict, Any

class TrajectoryRecorder:
    """
    A simplified trajectory recorder inspired by Trae Agent's methodology.
    It logs the step-by-step reasoning process during VQA data generation,
    creating an auditable trail for analysis and debugging.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """Adds a step to the trajectory."""
        step_data = {
            "step_index": len(self.steps) + 1,
            "name": name,
            "timestamp": time.time(),
            "inputs": inputs,
            "outputs": outputs,
        }
        self.steps.append(step_data)
        print(f"[Trajectory] Added step {step_data['step_index']}: {name}")

    def get_trajectory(self) -> Dict[str, Any]:
        """Returns the complete trajectory."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": time.time(),
            "total_duration": time.time() - self.start_time,
            "steps": self.steps,
        }

def generate_spatial_vqa_with_trajectory(image_id: str):
    """
    Simulates the VQASynth pipeline for generating a single spatial VQA sample,
    while recording the reasoning process using TrajectoryRecorder.
    """
    recorder = TrajectoryRecorder(session_id=f"vqa-gen-{image_id}")

    # Step 1: Object Localization (e.g., using GroundingDINO)
    recorder.add_step(
        name="Object Localization",
        inputs={"image_id": image_id, "text_prompts": ["red forklift", "cardboard boxes"]},
        outputs={"objects": [
            {"label": "red forklift", "box": [100, 200, 150, 280]},
            {"label": "cardboard boxes", "box": [400, 220, 500, 320]},
        ]}
    )
    objects = recorder.steps[-1]["outputs"]["objects"]

    # Step 2: Depth Estimation (e.g., using VGGT)
    recorder.add_step(
        name="Depth Estimation",
        inputs={"image_id": image_id},
        outputs={"depth_map_quality": "high", "avg_depth_per_object": {
            "red forklift": 5.2,
            "cardboard boxes": 4.8,
        }}
    )
    depths = recorder.steps[-1]["outputs"]["avg_depth_per_object"]

    # Step 3: Spatial Relationship Analysis
    forklift_x_center = (objects[0]["box"][0] + objects[0]["box"][2]) / 2
    boxes_x_center = (objects[1]["box"][0] + objects[1]["box"][2]) / 2
    is_left = forklift_x_center < boxes_x_center
    distance = abs(depths["red forklift"] - depths["cardboard boxes"])

    recorder.add_step(
        name="Spatial Relationship Analysis",
        inputs={"objects": objects, "depths": depths},
        outputs={
            "relationship": "is_left_of",
            "is_left": is_left,
            "estimated_distance_meters": round(distance, 2)
        }
    )

    # Step 4: VQA Generation (using prompt templates)
    question = f"Does the {objects[0]['label']} appear to the left of the {objects[1]['label']}?"
    answer = f"Yes, the {objects[0]['label']} is on the left side of the {objects[1]['label']}. They are approximately {round(distance, 2)} meters apart in depth."
    
    recorder.add_step(
        name="VQA Generation",
        inputs={"analysis_results": recorder.steps[-1]["outputs"]},
        outputs={
            "question": question,
            "answer": answer
        }
    )

    # Final output combines the VQA pair and the recorded trajectory
    final_output = {
        "image_id": image_id,
        "question": question,
        "answer": answer,
        "trajectory": recorder.get_trajectory()
    }

    return final_output

if __name__ == "__main__":
    # Run the experiment for a sample image
    vqa_sample = generate_spatial_vqa_with_trajectory(image_id="warehouse_sample_1.jpeg")
    
    # Print the result as a pretty-printed JSON
    print("\n--- Generated VQA Sample with Trajectory ---")
    print(json.dumps(vqa_sample, indent=2))
