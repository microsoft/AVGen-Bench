import cv2
import csv
import os
import numpy as np
from paddleocr import PaddleOCR
import paddleocr

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)
print("paddleocr package version:", paddleocr.__version__)
print("ocr config:", ocr.config)

def video_ocr_to_csv(video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Start processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")

    with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)

        header = ['Frame_Index', 'Timestamp(s)', 'Text', 'Confidence', 'Box_Points']
        writer.writerow(header)

        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps if fps > 0 else 0

            try:
                result = ocr.predict(input=frame)
            except Exception as e:
                print(f"Frame {frame_idx}: prediction error - {e}")
                frame_idx += 1
                continue

            for res in result:
                texts = []
                scores = []
                boxes = []

                try:
                    if 'rec_texts' in res and 'rec_scores' in res:
                        texts = res['rec_texts']
                        scores = res['rec_scores']
                        boxes = res['dt_polys'] if 'dt_polys' in res else []
                    elif isinstance(res, list):
                        for line in res:
                            boxes.append(line[0])
                            texts.append(line[1][0])
                            scores.append(line[1][1])
                    else:
                        if hasattr(res, 'getitem'):
                            texts = res['rec_texts']
                            scores = res['rec_scores']
                            boxes = res['dt_polys']
                except Exception as parse_err:
                    print(f"Frame {frame_idx}: result parsing error - {parse_err}")
                    continue

                for text, score, box in zip(texts, scores, boxes):
                    box_str = str(box.tolist()) if hasattr(box, 'tolist') else str(box)
                    
                    writer.writerow([
                        frame_idx, 
                        f"{timestamp:.3f}", 
                        text, 
                        f"{score:.4f}", 
                        box_str
                    ])

            if frame_idx % 10 == 0:
                print(f"Processed frames: {frame_idx}/{total_frames}")

            frame_idx += 1

    cap.release()
    print(f"Done. Results saved to: {output_csv_path}")

if __name__ == "__main__":
    input_video = "/path/to/Stealth_Action__The_Distraction.mp4"
    output_csv = "/path/to/AVGen-Bench/eval/Ocr/ocr_results.csv"

    if not os.path.exists(input_video):
        print(f"Please update the input_video path in code. File not found: {input_video}")
    else:
        video_ocr_to_csv(input_video, output_csv)
