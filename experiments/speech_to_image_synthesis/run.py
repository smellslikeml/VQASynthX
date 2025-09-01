import os
import argparse
import openai
import requests

# For compatibility with the source repository and its tested environment,
# this script uses the openai<1.0.0 API.
# Ensure the API key is set as an environment variable: OPENAI_API_KEY
try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    raise KeyError(
        "The OPENAI_API_KEY environment variable is not set. Please set it before running the script."
    )


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file using OpenAI's Whisper model.
    """
    print(f"Transcribing audio from: {audio_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    with open(audio_path, "rb") as audio_file:
        try:
            transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file)
            transcribed_text = transcript["text"]
            print(f"Successfully transcribed text: '{transcribed_text}'")
            return transcribed_text
        except openai.error.OpenAIError as e:
            print(f"An error occurred with the OpenAI API during transcription: {e}")
            raise


def generate_image_from_text(prompt: str, output_path: str, size: str = "1024x1024"):
    """
    Generates an image from a text prompt using OpenAI's DALL-E and saves it.
    """
    print(f"Generating image for prompt: '{prompt}'")
    try:
        response = openai.Image.create(prompt=prompt, n=1, size=size)
        image_url = response["data"][0]["url"]
        print(f"Image generated, URL: {image_url}")

        # Download and save the image
        image_response = requests.get(image_url, timeout=60)
        image_response.raise_for_status()  # Raise an exception for bad status codes

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(image_response.content)
        print(f"Image successfully saved to: {output_path}")

    except openai.error.OpenAIError as e:
        print(f"An error occurred with the OpenAI API during image generation: {e}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the image: {e}")
        raise


def main():
    """
    Main function to run the speech-to-image pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Generate an image from an audio file using Whisper and DALL-E."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the input audio file (e.g., .wav, .mp3).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated image file (e.g., /path/to/image.jpg).",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        choices=["256x256", "512x512", "1024x1024"],
        help="The size of the generated image.",
    )

    args = parser.parse_args()

    # Step 1: Transcribe audio to text
    transcribed_text = transcribe_audio(args.audio_path)

    # Step 2: Generate image from transcribed text
    if transcribed_text:
        generate_image_from_text(transcribed_text, args.output_path, args.size)


if __name__ == "__main__":
    main()
