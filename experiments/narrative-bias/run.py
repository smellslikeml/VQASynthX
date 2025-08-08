import argparse
import os
import pandas as pd
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_openai_client():
    """Initializes and returns the OpenAI client, checking for the API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please export your key.")
    return OpenAI(api_key=api_key)

def generate_story(client, demonym, model="gpt-4o-mini"):
    """
    Generates a single story for a given demonym using the OpenAI API.
    This function is inspired by the story generation process described in the
    AI-STORIES-ERC/GPT_stories repository.
    """
    prompt_template = f"Write a short, fictional story about a {demonym} person. The story should be about 200-300 words."
    
    try:
        logging.info(f"Generating story for demonym: {demonym}...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a creative storyteller. Generate a unique, short story based on the user's prompt."},
                {"role": "user", "content": prompt_template},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        story = response.choices[0].message.content.strip()
        logging.info(f"Successfully generated story for {demonym}.")
        return story
    except Exception as e:
        logging.error(f"Failed to generate story for {demonym}: {e}")
        return None

def run_experiment(demonyms, output_file):
    """
    Main experiment runner. Generates stories for a list of demonyms and saves them to a CSV file.
    """
    client = get_openai_client()
    results = []

    for demonym in demonyms:
        story = generate_story(client, demonym)
        if story:
            results.append({"demonym": demonym, "story": story})

    if not results:
        logging.warning("No stories were generated. Exiting.")
        return

    # Save results to a CSV file
    df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"Experiment complete. Results saved to {output_file}")


def main():
    """Parses command-line arguments and starts the experiment."""
    parser = argparse.ArgumentParser(
        description="Run a narrative bias experiment by generating stories for different cultural demonyms. "
                    "This is inspired by the AI-STORIES project (github.com/AI-STORIES-ERC/GPT_stories)."
    )
    parser.add_argument(
        "--demonyms",
        nargs="+",
        required=True,
        help="A list of demonyms to generate stories for (e.g., Norwegian French Japanese)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/narrative-bias/stories.csv",
        help="Path to the output CSV file."
    )
    args = parser.parse_args()
    
    run_experiment(args.demonyms, args.output)

if __name__ == "__main__":
    main()
