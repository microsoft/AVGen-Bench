import torch
from q_align import QAlignVideoScorer, load_video

def evaluate_video_with_official_api(video_path):
    print("Initializing scorer...")
    scorer = QAlignVideoScorer(device="cuda")

    print(f"Loading video: {video_path}")
    try:
        video_data = load_video(video_path)

        video_list = [video_data]

        print("Scoring...")
        scores = scorer(video_list)

        score_val = scores.tolist()[0]

        print("-" * 30)
        print(f"Video path: {video_path}")
        print(f"Official API score (Quality): {score_val:.4f}/1.0000")
        return score_val

    except FileNotFoundError:
        print(f"Error: video file not found: {video_path}")
    except Exception as e:
        print(f"Error: {e}")

# --- Example usage ---
my_video_path = "/path/to/video_generation/ovi/ads/Do you know what customers are saying about you_ (Anonymized).mp4"

# If the sample video is unavailable, use your own local file.
# my_video_path = "your_local_video.mp4"

evaluate_video_with_official_api(my_video_path)
