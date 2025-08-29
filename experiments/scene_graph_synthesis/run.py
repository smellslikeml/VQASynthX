import pydot
import json


def calculate_spatial_relationship(obj1, obj2):
    """
    Calculates a simple spatial relationship between two objects based on their
    bounding box centers.
    """
    center1_x = obj1["bbox"][0] + obj1["bbox"][2] / 2
    center1_y = obj1["bbox"][1] + obj1["bbox"][3] / 2
    center2_x = obj2["bbox"][0] + obj2["bbox"][2] / 2
    center2_y = obj2["bbox"][1] + obj2["bbox"][3] / 2

    dx = center1_x - center2_x
    dy = center1_y - center2_y

    if abs(dx) > abs(dy):
        return "left_of" if dx < 0 else "right_of"
    else:
        return "above" if dy < 0 else "below"


def build_scene_graph(objects):
    """
    Builds a pydot graph representing the spatial relationships between objects.
    This is analogous to the Code Property Graphs (CPGs) in ExplainVulD,
    but for visual scenes instead of code.
    """
    graph = pydot.Dot("scene_graph", graph_type="digraph")

    # Add nodes for each object
    for i, obj in enumerate(objects):
        node = pydot.Node(name=f"{obj['label']}_{i}", label=obj["label"])
        graph.add_node(node)

    # Add edges representing relationships
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue

            obj1 = objects[i]
            obj2 = objects[j]

            relationship = calculate_spatial_relationship(obj1, obj2)

            # Edge from obj1 to obj2, e.g., "cat is left_of dog"
            edge = pydot.Edge(
                f"{obj1['label']}_{i}", f"{obj2['label']}_{j}", label=relationship
            )
            graph.add_edge(edge)

    return graph


def generate_vqa_from_graph(graph, objects):
    """
    Generates simple VQA pairs by traversing the scene graph edges.
    """
    vqa_pairs = []

    object_map = {f"{obj['label']}_{i}": obj["label"] for i, obj in enumerate(objects)}

    for edge in graph.get_edge_list():
        source_node = edge.get_source()
        dest_node = edge.get_destination()
        relationship = edge.get_label()

        # Get the simple label for the question
        source_label = object_map.get(source_node, "unknown object")
        dest_label = object_map.get(dest_node, "unknown object")

        # Format relationship for a natural question
        rel_map = {
            "left_of": "to the left of",
            "right_of": "to the right of",
            "above": "above",
            "below": "below",
        }

        question = f"Is the {source_label} {rel_map.get(relationship, relationship)} the {dest_label}?"
        answer = "Yes"

        vqa_pairs.append({"question": question, "answer": answer})

    return vqa_pairs


def main():
    """
    Main function to run the scene graph synthesis experiment.
    """
    # Mock data representing detected objects in a scene.
    # In a full pipeline, this would come from an object detector.
    mock_objects = [
        {"label": "cat", "bbox": [50, 50, 100, 100]},  # x, y, w, h
        {"label": "dog", "bbox": [200, 50, 120, 100]},
        {"label": "table", "bbox": [125, 160, 150, 80]},
    ]

    print("--- Building Scene Graph from mock object data ---")
    scene_graph = build_scene_graph(mock_objects)

    output_dot_path = "scene_graph.dot"
    scene_graph.write_raw(output_dot_path)
    print(f"Scene graph saved to '{output_dot_path}'")
    print("Use `dot -Tpng scene_graph.dot -o scene_graph.png` to visualize.")
    print("")

    print("--- Generating VQA pairs from Scene Graph ---")
    vqa_data = generate_vqa_from_graph(scene_graph, mock_objects)

    print(json.dumps(vqa_data, indent=2))


if __name__ == "__main__":
    main()
