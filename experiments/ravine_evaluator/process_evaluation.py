import argparse
import json
import os
import requests
from io import BytesIO

import torch
import torchaudio
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Mock dataset containing image, audio, question, and expected answer.
# This allows the script to be self-contained for this initial experiment.
MOCK_DATASET = [
    {
        "id": "sample_1",
        "image_url": "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true",
        "audio_url": "https://actions.google.com/sounds/v1/emergency/beeper_confirmation.ogg", # Placeholder forklift backup beep
        "question": "Given the sound of a beeping alarm, is the red forklift likely in operation or about to move?",
        "audio_description_for_vlm": "A repetitive, high-pitched beeping sound, typical of a vehicle's reverse alarm, is audible.",
        "expected_keyword": "yes"
    },
    {
        "id": "sample_2",
        "image_url": "https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_2.jpeg?raw=true",
        "audio_url": "https://actions.google.com/sounds/v1/ambiences/coffee_shop.ogg", # Placeholder ambient noise
        "question": "Considering the quiet ambient sounds, is there any immediate activity involving the man in the red hat?",
        "audio_description_for_vlm": "The ambient sound is low and indistinct, with no specific machinery or alarms detected.",
        "expected_keyword": "no"
    }
]

def process_audio_fisher_inspired(audio_bytes):
    """
    Processes audio bytes into a spectrogram using a method inspired by the FISHER repo.
    This demonstrates the audio-processing part of the proposed pipeline.
    """
    try:
        wav, sr = torchaudio.load(BytesIO(audio_bytes))
        
        # Resample if necessary, e.g., to a standard 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
            sr = 16000

        # Normalize and create spectrogram as per FISHER's inference example
        wav = wav - wav.mean()
        stft_transformer = torchaudio.transforms.Spectrogram(
            n_fft=25 * sr // 1000, # 25ms window
            hop_length=10 * sr // 1000, # 10ms hop
            power=1,
            center=False
        )
        spec = torch.log(torch.abs(stft_transformer(wav)) + 1e-10)
        return spec
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def evaluate_model(model_id):
    """
    Main evaluation loop.
    """
    print(f"Loading model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    results = []
    correct_predictions = 0

    for item in MOCK_DATASET:
        print(f"--- Processing sample: {item['id']} ---")
        
        # 1. Load Image
        image_response = requests.get(item['image_url'])
        image = Image.open(BytesIO(image_response.content)).convert("RGB")

        # 2. Load and Process Audio
        audio_response = requests.get(item['audio_url'])
        spectrogram = process_audio_fisher_inspired(audio_response.content)
        
        if spectrogram is None:
            print(f"Skipping sample {item['id']} due to audio processing error.")
            continue
        
        print(f"Successfully generated spectrogram of shape: {spectrogram.shape}")
        
        # 3. Construct Prompt
        # For this experiment, we use a placeholder description that a model like FISHER
        # would generate from the spectrogram. This tests the VLM's reasoning ability.
        prompt = (
            f"You are an expert audio-visual AI assistant. Analyze the image and the accompanying audio context to answer the question.\n\n"
            f"AUDIO CONTEXT: {item['audio_description_for_vlm']}\n\n"
            f"USER: <image>\n{item['question']}\n"
            f"ASSISTANT:"
        )

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.bfloat16)

        # 4. Generate Response
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        response_text = processor.decode(output[0], skip_special_tokens=True)
        # Extract only the newly generated text (the answer part)
        answer = response_text.split("ASSISTANT:")[1].strip()

        print(f"Question: {item['question']}")
        print(f"Generated Answer: {answer}")

        # 5. Evaluate
        is_correct = item['expected_keyword'].lower() in answer.lower()
        if is_correct:
            correct_predictions += 1
        print(f"Correct: {is_correct} (Expected keyword: '{item['expected_keyword']}')")

        results.append({
            "id": item['id'],
            "question": item['question'],
            "answer": answer,
            "is_correct": is_correct
        })

    # 6. Report Final Score
    accuracy = (correct_predictions / len(MOCK_DATASET)) * 100 if MOCK_DATASET else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"Final Accuracy: {accuracy:.2f}%")

    # Save results to a file
    with open("ravine_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
    print("Results saved to ravine_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ravine Audio-Visual Evaluation.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="remyxai/SpaceThinker-Qwen2.5VL-3B",
        help="The Hugging Face model ID to evaluate."
    )
    args = parser.parse_args()
    evaluate_model(args.model_id)
