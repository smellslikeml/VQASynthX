import argparse
import os
import json
from pathlib import Path
import google.generativeai as genai
from PIL import Image

# Inspired by the prompt in SOURCE: Agent/OCR_Genmini.py
OCR_PROMPT = """
You are an expert document analysis agent. Your task is to perform a comprehensive OCR and structural analysis of the provided image.
Follow these guidelines precisely:
1.  **Extract all text** and present it, maintaining the original layout and structure as much as possible.
2.  **Identify any graphics** such as charts, tables, or complex diagrams.
3.  For each graphic, provide a detailed **description** of its content, purpose, and key data points.
4.  For any **tables**, extract the data into a structured format (e.g., a list of lists or JSON).
5.  For any **mathematical formulas**, use LaTeX expressions.

Return your final output as a single JSON object with the following keys:
- "full_text": A string containing all extracted text, with newlines to approximate the original layout.
- "graphics": A list of objects, where each object has 'type' (e.g., 'bar_chart', 'table', 'diagram') and 'description' (a detailed explanation).
- "tables": A list of objects, where each object has 'description' and 'data' (the extracted table data as a list of lists).
- "formulas": A list of strings, each containing a LaTeX expression for a detected formula.
"""


def analyze_image(image_path: Path, output_path: Path):
    """Analyzes an image using Gemini and saves the structured output."""
    print(f"Analyzing image: {image_path}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content([OCR_PROMPT, img])
        # Clean up the response from markdown fences if they exist
        cleaned_response = (
            response.text.strip().replace("```json", "").replace("```", "").strip()
        )

        # Validate and save the JSON output
        data = json.loads(cleaned_response)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved OCR analysis to {output_path}")

    except Exception as e:
        print(f"An error occurred during API call or processing: {e}")
        print(
            f"Raw response text: {response.text if 'response' in locals() else 'N/A'}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run multimodal OCR and analysis on an image."
    )
    parser.add_argument(
        "--image-path", type=Path, required=True, help="Path to the input image file."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the output JSON file.",
    )
    args = parser.parse_args()

    analyze_image(image_path=args.image_path, output_path=args.output - path)


if __name__ == "__main__":
    main()
