import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import json
import os
import glob

# --- Configuration ---
# These paths would be mounted into the Docker container
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "./data/prompt_stage_output")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "./data/images")
OUTPUT_DATA_DIR = os.environ.get("OUTPUT_DATA_DIR", "./data/curated_output")


# --- State Management ---
class CurationState:
    def __init__(self, data_files, image_dir, output_dir):
        self.data_files = data_files
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.current_index = 0
        self.current_data = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def has_next(self):
        return self.current_index < len(self.data_files)

    def load_next(self):
        if not self.has_next():
            self.current_data = None
            return None

        filepath = self.data_files[self.current_index]
        with open(filepath, "r") as f:
            self.current_data = json.load(f)

        self.current_index += 1
        return self.current_data

    def save_curation(self, corrected_answer):
        if not self.current_data:
            return

        output_record = self.current_data.copy()
        output_record["human_verified"] = True
        output_record["original_answer"] = self.current_data["conversations"][1][
            "value"
        ]
        output_record["conversations"][1]["value"] = corrected_answer

        base_name = os.path.basename(self.data_files[self.current_index - 1])
        output_path = os.path.join(self.output_dir, base_name)

        with open(output_path, "w") as f:
            json.dump(output_record, f, indent=2)


# --- UI Widgets ---
image_widget = widgets.Image(format="jpeg", width=512)
question_widget = widgets.HTML(value="<b>Question:</b>")
model_answer_widget = widgets.HTML(value="<b>Model's Answer:</b>")

correction_box = widgets.Textarea(
    value="",
    placeholder="If the answer is wrong, provide a correction here. Otherwise, leave blank.",
    description="Correction:",
    layout=widgets.Layout(width="90%", height="100px"),
)

submit_button = widgets.Button(
    description="Accept & Next", button_style="success", icon="check"
)

status_label = widgets.Label(value="Welcome! Loading the first sample.")


# --- App Logic ---
def update_ui(data):
    if data is None:
        clear_output(wait=True)
        display(
            widgets.HTML(
                "<h1>Curation Complete!</h1><p>All samples have been processed.</p>"
            )
        )
        return

    # Load and display image
    image_path = os.path.join(state.image_dir, data.get("image"))
    try:
        with open(image_path, "rb") as f:
            image_widget.value = f.read()
    except FileNotFoundError:
        # In a real scenario, handle this more gracefully
        image_widget.value = b""
        status_label.value = f"Error: Image not found at {image_path}"
        return

    # Display Q&A
    question_text = data["conversations"][0]["value"].replace("<image>\n", "")
    question_widget.value = f"<b>Question:</b><br>{question_text}"
    model_answer = data["conversations"][1]["value"]
    model_answer_widget.value = f"<b>Model's Answer:</b><br>{model_answer}"

    # Reset input fields
    correction_box.value = ""
    submit_button.description = "Accept & Next"
    submit_button.button_style = "success"
    status_label.value = f"Showing sample {state.current_index}/{len(state.data_files)}"


def on_submit_clicked(b):
    corrected_answer = correction_box.value.strip()
    if not corrected_answer:
        # If no correction, use the original model answer
        corrected_answer = state.current_data["conversations"][1]["value"]

    state.save_curation(corrected_answer)

    # Load and display next sample
    next_data = state.load_next()
    update_ui(next_data)


submit_button.on_click(on_submit_clicked)

# Initialize state and start the app
try:
    json_files = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {INPUT_DATA_DIR}")

    state = CurationState(json_files, IMAGE_DIR, OUTPUT_DATA_DIR)

    # Initial load
    initial_data = state.load_next()
    if initial_data:
        update_ui(initial_data)
        # --- UI Layout ---
        vbox = widgets.VBox(
            [
                status_label,
                widgets.HBox(
                    [image_widget, widgets.VBox([question_widget, model_answer_widget])]
                ),
                correction_box,
                submit_button,
            ]
        )
        display(vbox)
    else:
        display(widgets.HTML("<h1>No data to curate.</h1>"))

except FileNotFoundError as e:
    display(
        widgets.HTML(
            f"<h1>Error</h1><p>Could not start curation app: {e}</p><p>Ensure INPUT_DATA_DIR and IMAGE_DIR environment variables are set correctly and point to valid data.</p>"
        )
    )
