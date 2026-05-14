import cv2
import csv
import os
import numpy as np
import paddle
from paddleocr import PaddleOCR
import paddleocr

OCR_DEVICE = os.getenv(
    "OCR_DEVICE",
    "gpu:0" if paddle.is_compiled_with_cuda() else "cpu",
).strip()
try:
    paddle.device.set_device(OCR_DEVICE)
except Exception:
    OCR_DEVICE = "cpu"
    paddle.device.set_device(OCR_DEVICE)

OCR_BATCH_SIZE = max(
    1,
    int(os.getenv("OCR_BATCH_SIZE", "16" if OCR_DEVICE.startswith("gpu") else "1")),
)
OCR_TEXT_RECOGNITION_BATCH_SIZE = max(
    1,
    int(os.getenv("OCR_TEXT_RECOGNITION_BATCH_SIZE", str(OCR_BATCH_SIZE))),
)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_recognition_batch_size=OCR_TEXT_RECOGNITION_BATCH_SIZE,
    device=OCR_DEVICE)
try:
    ocr._merged_paddlex_config["batch_size"] = OCR_BATCH_SIZE
    ocr.paddlex_pipeline.batch_sampler.batch_size = OCR_BATCH_SIZE
except Exception:
    pass
print("paddleocr package version:", paddleocr.__version__)
print("ocr config:", ocr.config)


def parse_ocr_result(res):
    texts = []
    scores = []
    boxes = []

    if isinstance(res, dict) and "rec_texts" in res and "rec_scores" in res:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("dt_polys", []) or []
    elif isinstance(res, list):
        for line in res:
            boxes.append(line[0])
            texts.append(line[1][0])
            scores.append(line[1][1])
    elif hasattr(res, "getitem"):
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        boxes = res["dt_polys"]

    return texts, scores, boxes


def flush_batch(writer, frames_meta, frames):
    if not frames:
        return

    try:
        results = ocr.predict(input=frames)
    except Exception as e:
        print(f"Batch prediction error on frames {frames_meta[0][0]}-{frames_meta[-1][0]}: {e}")
        results = []
        for _, _, frame in frames_meta:
            try:
                per_frame_result = ocr.predict(input=[frame])
                results.append(per_frame_result[0] if per_frame_result else [])
            except Exception as frame_err:
                print(f"Per-frame fallback error: {frame_err}")
                results.append([])

    for (frame_idx, timestamp, _), res in zip(frames_meta, results):
        try:
            texts, scores, boxes = parse_ocr_result(res)
        except Exception as parse_err:
            print(f"Frame {frame_idx}: result parsing error - {parse_err}")
            continue

        for text, score, box in zip(texts, scores, boxes):
            box_str = str(box.tolist()) if hasattr(box, "tolist") else str(box)
            writer.writerow([
                frame_idx,
                f"{timestamp:.3f}",
                text,
                f"{score:.4f}",
                box_str
            ])

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
        pending_frames = []
        pending_meta = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps if fps > 0 else 0
            pending_frames.append(frame)
            pending_meta.append((frame_idx, timestamp, frame))
            if len(pending_frames) >= OCR_BATCH_SIZE:
                flush_batch(writer, pending_meta, pending_frames)
                pending_frames = []
                pending_meta = []

            if frame_idx % 10 == 0:
                print(f"Processed frames: {frame_idx}/{total_frames}")

            frame_idx += 1

        flush_batch(writer, pending_meta, pending_frames)

    cap.release()
    print(f"Done. Results saved to: {output_csv_path}")

if __name__ == "__main__":
    input_video = "/path/to/Stealth_Action__The_Distraction.mp4"
    output_csv = "/path/to/AVGen-Bench/eval/Ocr/ocr_results.csv"

    if not os.path.exists(input_video):
        print(f"Please update the input_video path in code. File not found: {input_video}")
    else:
        video_ocr_to_csv(input_video, output_csv)
