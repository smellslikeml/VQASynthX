# This script adapts the core idea of VLM-based keypoint proposal from the CLASP repository.
# It provides a minimal, self-contained experiment to generate semantic keypoints for an object
# in an image, which can then be used to synthesize more granular VQA data.
# Original concept evidence: https://github.com/dengyh16code/CLASP/blob/main/semantic_keypoints/semantic_keypoints.py

import os
import json
import argparse
import base64
from openai import OpenAI
from PIL import Image


def encode_image(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_keypoints_from_vlm(client: OpenAI, image_path: str, object_class: str):
    """
    Uses a Vision-Language Model to propose semantic keypoints for a given object class in an image.
    This approach is adapted from the CLASP repository, which uses a VLM to identify
    keypoints for robotic manipulation tasks. Here, we adapt it for data synthesis.
    Ref: https://github.com/dengyh16code/CLASP/blob/main/semantic_keypoints/semantic_keypoints.py
    """
    base64_image = encode_image(image_path)

    # This prompt structure and the use of function calling are directly inspired by the CLASP implementation.
    # It instructs the model to identify keypoints relevant to the object's structure and function,
    # ensuring a structured output.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are a helpful assistant for a robotics and computer vision project. Your task is to identify and list semantic keypoints for the specified object in the image. Please provide the semantic keypoints of the '{object_class}'. Focus on keypoints that are functionally important or define the object's structure. For example, for a t-shirt, keypoints could be 'collar', 'left sleeve cuff', 'right sleeve cuff', and 'bottom hem'.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "propose_keypoints",
                "description": f"Propose a list of semantic keypoints for the {object_class}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keypoints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of semantic keypoint names, e.g., ['collar', 'left_sleeve_cuff']",
                        }
                    },
                    "required": ["keypoints"],
                },
            },
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "propose_keypoints"}},
        )
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            function_args = json.loads(tool_calls[0].function.arguments)
            return function_args
        else:
            return {"error": "No tool calls were made by the model."}

    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Propose semantic keypoints for an object in an image using a VLM."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--object_class",
        type=str,
        required=True,
        help="The class of the object to find keypoints for (e.g., 't-shirt', 'mug', 'chair').",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)

    print(
        f"Proposing keypoints for object class '{args.object_class}' in image '{args.image_path}'..."
    )
    keypoints_result = get_keypoints_from_vlm(
        client, args.image_path, args.object_class
    )

    print("\n--- Proposed Keypoints ---")
    print(json.dumps(keypoints_result, indent=2))
    print(
        "\nThis output can now be used in the VQASynth pipeline to generate fine-grained questions about object parts."
    )
    print(
        "The next step would be to localize these keypoints using a vision model, similar to CLASP's use of SD+DINO."
    )


if __name__ == "__main__":
    main()
