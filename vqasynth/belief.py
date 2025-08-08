"""
Inspired by the Belief-SG concept of modeling uncertainty within the state representation.
This module provides simple classes to represent a scene with probabilistic object properties.
"""
from typing import Dict, List, Any

class BeliefObject:
    """Represents an object in the scene with a belief distribution over its labels."""

    def __init__(self, obj_id: int, label_probabilities: Dict[str, float], attributes: Dict[str, Any] = None):
        """
        Initializes a BeliefObject.

        Args:
            obj_id (int): A unique identifier for the object.
            label_probabilities (Dict[str, float]): A dictionary mapping possible labels to their probabilities.
                                                     Probabilities should sum to 1.0.
            attributes (Dict[str, Any]): Other deterministic attributes (e.g., location, color).
        """
        if not abs(sum(label_probabilities.values()) - 1.0) < 1e-6:
            raise ValueError("Label probabilities must sum to 1.0")

        self.obj_id = obj_id
        self.label_probabilities = label_probabilities
        self.attributes = attributes if attributes is not None else {}

    def get_most_likely_label(self):
        """Returns the label with the highest probability."""
        if not self.label_probabilities:
            return None, 0.0
        max_label = max(self.label_probabilities, key=self.label_probabilities.get)
        return max_label, self.label_probabilities[max_label]

    def __repr__(self):
        return f"BeliefObject(id={self.obj_id}, most_likely='{self.get_most_likely_label()[0]}')"

class SceneBeliefState:
    """Represents the entire scene as a collection of BeliefObjects."""

    def __init__(self, objects: List[BeliefObject] = None):
        """
        Initializes the SceneBeliefState.

        Args:
            objects (List[BeliefObject]): A list of BeliefObjects in the scene.
        """
        self.objects = {obj.obj_id: obj for obj in objects} if objects else {}

    def add_object(self, obj: BeliefObject):
        """Adds a BeliefObject to the scene."""
        self.objects[obj.obj_id] = obj

    def get_object(self, obj_id: int) -> BeliefObject:
        """Retrieves an object by its ID."""
        return self.objects.get(obj_id)

    def __repr__(self):
        return f"SceneBeliefState(objects={list(self.objects.values())})"
