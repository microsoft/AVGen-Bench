import torch
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor

def evaluate_video_audio(video_path):
    print(f"Loading model and processing file: {video_path} ...")

    # 1. Initialize predictor
    # The first run may download checkpoints (~1.3GB).
    predictor = initialize_predictor()

    try:
        # 2. Extract audio from MP4
        # torchaudio.load can read audio tracks directly from the video container.
        # Returns: waveform (Tensor), sample_rate (int)
        wav, sr = torchaudio.load(video_path)

        # 3. Build input payload
        # See docs: "Infer from torch tensor".
        # Note: "path" maps to a waveform tensor here, not a file path string.
        input_data = [
            {"path": wav, "sample_rate": sr}
        ]

        # 4. Run inference
        print("Running score inference...")
        predictions = predictor.forward(input_data)

        # 5. Format output
        score = predictions[0]

        print("-" * 30)
        print(f"File: {video_path}")
        print("Detailed scores:")
        print(f"  CE (Content Enjoyment):   {score['CE']:.4f}")
        print(f"  CU (Content Usefulness):  {score['CU']:.4f}")
        print(f"  PC (Production Complexity): {score['PC']:.4f}")
        print(f"  PQ (Production Quality):    {score['PQ']:.4f}")
        print("-" * 30)
        
        return score

    except RuntimeError as e:
        print(f"Error: failed to load audio. Please ensure ffmpeg is installed and the file path is correct.\nDetails: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# --- Example usage ---
if __name__ == "__main__":
    mp4_file = "/path/to/Ukulele__Island_Strumming.mp4"

    import os
    if not os.path.exists(mp4_file):
        print(f"Tip: replace '{mp4_file}' in this script with your actual video path")
    else:
        evaluate_video_audio(mp4_file)
