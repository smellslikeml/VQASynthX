# Inspired by the prompt ensembling technique in SmartCLIP (eval/openai_imagenet_template.py)
# This file provides a variety of linguistic templates for generating spatial relationship questions.
# The goal is to create a more robust training dataset for VQA models.

SPATIAL_RELATIONSHIP_TEMPLATES = {
    "left_of": [
        lambda c1, c2: f"Is the {c1} to the left of the {c2}?",
        lambda c1, c2: f"Does the {c1} appear on the left side of the {c2}?",
        lambda c1, c2: f"From this viewpoint, is the {c1} located to the left of the {c2}?",
        lambda c1, c2: f"Can you confirm that the {c1} is positioned left of the {c2}?",
        lambda c1, c2: f"In the image, is the {c1} on the left when compared to the {c2}?",
    ],
    "right_of": [
        lambda c1, c2: f"Is the {c1} to the right of the {c2}?",
        lambda c1, c2: f"Does the {c1} appear on the right side of the {c2}?",
        lambda c1, c2: f"From this viewpoint, is the {c1} located to the right of the {c2}?",
        lambda c1, c2: f"Can you confirm that the {c1} is positioned right of the {c2}?",
        lambda c1, c2: f"In the image, is the {c1} on the right when compared to the {c2}?",
    ],
    "behind": [
        lambda c1, c2: f"Is the {c1} behind the {c2}?",
        lambda c1, c2: f"Is the {c1} positioned further away than the {c2}?",
        lambda c1, c2: f"In terms of depth, is the {c1} in the background relative to the {c2}?",
        lambda c1, c2: f"Looking at the scene, is the {c2} in front of the {c1}?",
    ],
    "in_front_of": [
        lambda c1, c2: f"Is the {c1} in front of the {c2}?",
        lambda c1, c2: f"Is the {c1} positioned closer than the {c2}?",
        lambda c1, c2: f"In terms of depth, is the {c1} in the foreground relative to the {c2}?",
        lambda c1, c2: f"Looking at the scene, is the {c2} behind the {c1}?",
    ],
    "above": [
        lambda c1, c2: f"Is the {c1} above the {c2}?",
        lambda c1, c2: f"Is the {c1} at a higher elevation than the {c2}?",
        lambda c1, c2: f"In the vertical axis, is the {c1} over the {c2}?",
    ],
    "below": [
        lambda c1, c2: f"Is the {c1} below the {c2}?",
        lambda c1, c2: f"Is the {c1} at a lower elevation than the {c2}?",
        lambda c1, c2: f"In the vertical axis, is the {c1} under the {c2}?",
    ],
}
