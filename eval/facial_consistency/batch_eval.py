# batch_eval.py
# Evaluate face-consistency vs Gemini-labeled primary face ID range, aggregated by category + overall.
#
# Expected input layout:
#   prompts_dir/
#     ads.expected_faces.json
#     news.expected_faces.json
#     ...
#   root_videos/
#     ads/<safe_filename(content)>.mp4
#     news/<safe_filename(content)>.mp4
#     ...
#
# Notes:
# - Only evaluates PRIMARY characters (per Gemini labels: min_ids/max_ids).
# - If Gemini says no face (max_ids == 0), skip that item (not counted in averages).
# - Uses your MultiFaceConsistencyV2 pipeline to detect/track/cluster faces, then derives "primary clusters".
# - Outputs per-category mean score and overall mean score; also dumps per-video results JSON.

import os
import re
import json
import math
import argparse
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN


# ----------------------------
# safe_filename (given)
# ----------------------------
def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name).strip()
    name = re.sub(r"[/\:*?\"<>|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


# ----------------------------
# Utility
# ----------------------------
def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def cosine_sim(a, b):
    return float(np.dot(a, b))

def bbox_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    if inter <= 0:
        return 0.0
    areaA = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    areaB = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    denom = areaA + areaB - inter
    return float(inter / denom) if denom > 0 else 0.0

def weighted_mean(embs, weights):
    embs = np.asarray(embs)
    w = np.asarray(weights).reshape(-1, 1)
    s = (embs * w).sum(axis=0) / max(w.sum(), 1e-12)
    return s

def percentile(arr, q):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))

def tracklet_internal_stability(tracklet_rec, all_faces, use_weights=True):
    idxs = tracklet_rec['indices']
    mu = tracklet_rec['embedding']  # already l2-normalized
    sims = []
    ws = []

    for j, i in enumerate(idxs):
        e = all_faces[i]['embedding']  # already l2-normalized
        sims.append(cosine_sim(e, mu))
        if use_weights:
            f = all_faces[i]
            ws.append(f['det_score'] * math.sqrt(max(f['rel_area'], 1e-12)))
        else:
            ws.append(1.0)

    sims = np.asarray(sims, dtype=np.float64)
    ws = np.asarray(ws, dtype=np.float64)
    wsum = float(ws.sum()) if ws.size else 0.0

    if sims.size == 0:
        return {
            'tracklet_id': int(tracklet_rec['tracklet_id']),
            'n': 0,
            'mean_sim': 0.0,
            'std_sim': 0.0,
            'var_sim': 0.0,
            'min_sim': 0.0,
            'p05_sim': 0.0,
            'p50_sim': 0.0,
            'p95_sim': 0.0,
        }

    if wsum > 1e-12:
        mean = float((sims * ws).sum() / wsum)
        var = float((ws * (sims - mean) ** 2).sum() / wsum)
    else:
        mean = float(np.mean(sims))
        var = float(np.var(sims))

    return {
        'tracklet_id': int(tracklet_rec['tracklet_id']),
        'n': int(sims.size),
        'mean_sim': float(mean),
        'std_sim': float(math.sqrt(max(var, 0.0))),
        'var_sim': float(var),
        'min_sim': float(np.min(sims)),
        'p05_sim': percentile(sims, 0.05),
        'p50_sim': percentile(sims, 0.50),
        'p95_sim': percentile(sims, 0.95),
    }

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ----------------------------
# Face analyzer (your pipeline)
# ----------------------------
class MultiFaceConsistencyV2:
    def __init__(self, model_name='buffalo_l', ctx_id=0, det_size=(640, 640)):
        import onnxruntime as ort
        print("ORT providers (python):", ort.get_available_providers())

        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def process_video(self,
                      video_path,
                      sample_rate=1,
                      min_face_px=30,
                      min_det_score=0.6,
                      min_rel_area=0.0005):
        cap = cv2.VideoCapture(video_path)
        all_faces = []

        frame_idx = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_area = max(1, width * height)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                faces = self.app.get(frame)

                for face in faces:
                    b = face.bbox.astype(float).tolist()
                    w = b[2] - b[0]
                    h = b[3] - b[1]
                    if w < min_face_px or h < min_face_px:
                        continue
                    if float(face.det_score) < float(min_det_score):
                        continue
                    rel_area = (w * h) / frame_area
                    if rel_area < min_rel_area:
                        continue

                    emb = np.asarray(face.embedding, dtype=np.float32)
                    emb = l2norm(emb)

                    all_faces.append({
                        'frame_idx': frame_idx,
                        'embedding': emb,
                        'bbox': b,
                        'det_score': float(face.det_score),
                        'rel_area': float(rel_area),
                    })

            frame_idx += 1

        cap.release()
        return all_faces

    def build_tracklets(self,
                        all_faces,
                        max_gap_frames=3,
                        min_iou=0.1,
                        min_cos=0.35,
                        w_cos=0.7,
                        w_iou=0.3):
        if not all_faces:
            return []

        by_frame = defaultdict(list)
        for i, f in enumerate(all_faces):
            by_frame[f['frame_idx']].append(i)

        frames_sorted = sorted(by_frame.keys())

        tracklets = []
        active = {}  # tid -> state

        def new_track(idx):
            tid = len(tracklets)
            tracklets.append({'tracklet_id': tid, 'indices': [idx]})
            active[tid] = {
                'last_frame': all_faces[idx]['frame_idx'],
                'last_bbox': all_faces[idx]['bbox'],
                'last_emb': all_faces[idx]['embedding'],
                'indices': [idx]
            }
            all_faces[idx]['tracklet_id'] = tid

        for idx in by_frame[frames_sorted[0]]:
            new_track(idx)

        for fno in frames_sorted[1:]:
            to_del = []
            for tid, st in active.items():
                if fno - st['last_frame'] > max_gap_frames:
                    tracklets[tid]['indices'] = st['indices']
                    to_del.append(tid)
            for tid in to_del:
                del active[tid]

            det_indices = by_frame[fno]
            if not det_indices:
                continue

            candidates = []
            for di in det_indices:
                d = all_faces[di]
                best = None
                for tid, st in active.items():
                    iou = bbox_iou(d['bbox'], st['last_bbox'])
                    if iou < min_iou:
                        continue
                    cos = cosine_sim(d['embedding'], st['last_emb'])
                    if cos < min_cos:
                        continue
                    score = w_cos * cos + w_iou * iou
                    if (best is None) or (score > best[0]):
                        best = (score, tid, cos, iou)
                if best is not None:
                    candidates.append((best[0], di, best[1], best[2], best[3]))

            candidates.sort(reverse=True, key=lambda x: x[0])
            used_dets = set()
            used_tracks = set()

            for score, di, tid, cos, iou in candidates:
                if di in used_dets or tid in used_tracks:
                    continue
                used_dets.add(di); used_tracks.add(tid)
                st = active[tid]
                st['last_frame'] = fno
                st['last_bbox'] = all_faces[di]['bbox']
                st['last_emb'] = all_faces[di]['embedding']
                st['indices'].append(di)
                all_faces[di]['tracklet_id'] = tid

            for di in det_indices:
                if di not in used_dets:
                    new_track(di)

        for tid, st in active.items():
            tracklets[tid]['indices'] = st['indices']

        return tracklets

    def tracklet_embeddings(self, all_faces, tracklets, min_len=3):
        recs = []
        for tr in tracklets:
            idxs = tr['indices']
            if len(idxs) < min_len:
                continue

            embs = [all_faces[i]['embedding'] for i in idxs]
            ws = []
            for i in idxs:
                f = all_faces[i]
                ws.append(f['det_score'] * math.sqrt(max(f['rel_area'], 1e-12)))

            mean = weighted_mean(embs, ws)
            mean = l2norm(mean)

            frames = [all_faces[i]['frame_idx'] for i in idxs]
            recs.append({
                'tracklet_id': tr['tracklet_id'],
                'n': len(idxs),
                'start_frame': int(min(frames)),
                'end_frame': int(max(frames)),
                'embedding': mean,
                'indices': idxs,
                'weights': ws,
                'mean_det_score': float(np.mean([all_faces[i]['det_score'] for i in idxs])),
            })
        return recs

    def cluster_tracklets(self,
                         tracklet_recs,
                         eps=0.55,
                         min_samples=1):
        if not tracklet_recs:
            return None

        X = np.stack([r['embedding'] for r in tracklet_recs], axis=0)
        X = l2norm(X)

        clustering = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric='euclidean').fit(X)
        labels = clustering.labels_

        identities = defaultdict(list)
        for i, lab in enumerate(labels):
            tracklet_recs[i]['cluster_id'] = int(lab)
            if lab == -1:
                continue
            identities[int(lab)].append(tracklet_recs[i])

        return identities, tracklet_recs, labels, float(eps)

    def propagate_cluster_to_faces(self, all_faces, tracklet_recs):
        tmap = {r['tracklet_id']: r.get('cluster_id', -1) for r in tracklet_recs}
        for f in all_faces:
            tid = f.get('tracklet_id', None)
            if tid is None:
                f['cluster_id'] = -1
            else:
                f['cluster_id'] = int(tmap.get(tid, -1))

    def consistency_metrics(self, all_faces, tracklet_recs):
        if not all_faces:
            return {}

        # cluster length by tracklet n
        cluster_len = defaultdict(int)
        noise_len = 0
        for r in tracklet_recs:
            lab = int(r.get('cluster_id', -1))
            if lab == -1:
                noise_len += r['n']
            else:
                cluster_len[lab] += r['n']

        total = sum(cluster_len.values()) + noise_len
        if total == 0:
            return {}

        sorted_clusters = sorted(cluster_len.items(), key=lambda x: x[1], reverse=True)
        main_cluster, main_len = (sorted_clusters[0][0], sorted_clusters[0][1]) if sorted_clusters else (-1, 0)
        main_ratio = main_len / total if total else 0.0
        noise_ratio = noise_len / total if total else 0.0

        # tracklet internal stats
        tracklet_internal = []
        for r in tracklet_recs:
            if int(r.get('cluster_id', -1)) == -1:
                continue
            tracklet_internal.append(tracklet_internal_stability(r, all_faces, use_weights=True))

        # cluster-level from tracklets (weighted by n)
        # weighted mean of tracklet mean_sim, p05_sim, p50_sim
        by_cluster_items = defaultdict(list)
        for st in tracklet_internal:
            tid = st['tracklet_id']
            rec = next((r for r in tracklet_recs if int(r['tracklet_id']) == int(tid)), None)
            if rec is None:
                continue
            cid = int(rec.get('cluster_id', -1))
            if cid == -1:
                continue
            by_cluster_items[cid].append((st, int(rec['n'])))

        cluster_internal_from_tracklets = {}
        for cid, items in by_cluster_items.items():
            weights = np.asarray([w for _, w in items], dtype=np.float64)
            wsum = float(weights.sum()) if weights.size else 0.0

            mean_sims = np.asarray([st['mean_sim'] for st, _ in items], dtype=np.float64)
            p05_sims = np.asarray([st['p05_sim'] for st, _ in items], dtype=np.float64)
            p50_sims = np.asarray([st['p50_sim'] for st, _ in items], dtype=np.float64)

            if wsum > 1e-12:
                w_mean = float((mean_sims * weights).sum() / wsum)
                w_p05 = float((p05_sims * weights).sum() / wsum)
                w_p50 = float((p50_sims * weights).sum() / wsum)
            else:
                w_mean = float(np.mean(mean_sims)) if mean_sims.size else 0.0
                w_p05 = float(np.mean(p05_sims)) if p05_sims.size else 0.0
                w_p50 = float(np.mean(p50_sims)) if p50_sims.size else 0.0

            cluster_internal_from_tracklets[cid] = {
                'weighted_mean_of_tracklet_mean_sim': w_mean,
                'weighted_mean_of_tracklet_p05_sim': w_p05,
                'weighted_mean_of_tracklet_p50_sim': w_p50,
            }

        return {
            'total_face_dets': int(len(all_faces)),
            'total_tracklets': int(len(tracklet_recs)),
            'clusters': {int(k): int(v) for k, v in cluster_len.items()},
            'main_cluster_id': int(main_cluster),
            'main_cluster_ratio': float(main_ratio),
            'noise_ratio': float(noise_ratio),
            'tracklet_internal_stability': tracklet_internal,
            'cluster_internal_stability_from_tracklets': cluster_internal_from_tracklets,
        }


# ----------------------------
# Scoring helpers (primary-only)
# ----------------------------
def pick_primary_clusters(cluster_len: Dict[int, int],
                         total_face_dets: int,
                         primary_ratio_thresh: float = 0.10,
                         min_cluster_dets: int = 30) -> List[int]:
    """
    Heuristic: clusters that appear frequently are "primary".
    """
    if total_face_dets <= 0:
        return []
    prim = []
    for cid, n in cluster_len.items():
        if n >= min_cluster_dets and (n / total_face_dets) >= primary_ratio_thresh:
            prim.append(int(cid))
    prim.sort(key=lambda c: cluster_len.get(c, 0), reverse=True)
    return prim

def score_count_range(k_pred: int, k_min: int, k_max: int) -> float:
    """
    Returns [0,1] score for predicted primary IDs falling within expected range.
    Inside range => 1
    Outside => linear decay by distance relative to range width/center
    """
    k_min = int(k_min); k_max = int(k_max)
    if k_min < 0: k_min = 0
    if k_max < k_min: k_max = k_min
    k_pred = int(k_pred)

    if k_min == 0 and k_max == 0:
        return 1.0 if k_pred == 0 else 0.0

    if k_min <= k_pred <= k_max:
        return 1.0

    # distance to nearest bound
    if k_pred < k_min:
        dist = k_min - k_pred
    else:
        dist = k_pred - k_max

    # scale: allow small deviation; wider ranges are more tolerant
    width = max(1, k_max - k_min)
    denom = max(1, width)  # linear
    return clamp(1.0 - dist / denom, 0.0, 1.0)

def score_stability_primary(primary_cids: List[int],
                            cluster_len: Dict[int, int],
                            cluster_stab: Dict[int, Dict[str, float]],
                            total_face_dets: int,
                            p50_good: float = 0.80,
                            p50_excellent: float = 0.92) -> float:
    """
    Weighted by cluster length. Uses weighted_mean_of_tracklet_p50_sim (median) stability.
    Maps p50 in [p50_good..p50_excellent] -> [0..1], below good -> 0, above excellent -> 1.
    """
    if not primary_cids:
        return 0.0
    weights = []
    vals = []
    for cid in primary_cids:
        n = float(cluster_len.get(cid, 0))
        stab = cluster_stab.get(cid, {})
        p50 = float(stab.get('weighted_mean_of_tracklet_p50_sim', 0.0))
        s = clamp((p50 - p50_good) / max(1e-6, (p50_excellent - p50_good)), 0.0, 1.0)
        weights.append(n)
        vals.append(s)
    wsum = sum(weights)
    if wsum <= 1e-9:
        return float(np.mean(vals)) if vals else 0.0
    return float(sum(w*v for w, v in zip(weights, vals)) / wsum)

def compute_final_score(expected_min: int,
                        expected_max: int,
                        metrics: Dict[str, Any],
                        primary_ratio_thresh: float,
                        min_cluster_dets: int) -> Dict[str, Any]:
    """
    Returns dict with final_score and breakdown.
    """
    cluster_len = metrics.get("clusters", {}) or {}
    total_face_dets = int(metrics.get("total_face_dets", 0) or 0)
    cluster_stab = metrics.get("cluster_internal_stability_from_tracklets", {}) or {}

    primary_cids = pick_primary_clusters(
        cluster_len=cluster_len,
        total_face_dets=total_face_dets,
        primary_ratio_thresh=primary_ratio_thresh,
        min_cluster_dets=min_cluster_dets
    )
    k_primary_pred = len(primary_cids)

    # count match: 40
    s_count = score_count_range(k_primary_pred, expected_min, expected_max)
    score_count = 40.0 * s_count

    # stability: 60
    s_stab = score_stability_primary(primary_cids, cluster_len, cluster_stab, total_face_dets)
    score_stab = 60.0 * s_stab

    final = score_count + score_stab
    return {
        "final_score": float(final),
        "breakdown": {
            "score_count_40": float(score_count),
            "score_stability_60": float(score_stab),
            "s_count": float(s_count),
            "s_stability": float(s_stab),
        },
        "observed": {
            "primary_cluster_ids": primary_cids,
            "k_primary_pred": int(k_primary_pred),
            "cluster_len": {int(k): int(v) for k, v in cluster_len.items()},
        }
    }


# ----------------------------
# IO: load Gemini labels
# ----------------------------
def load_expected_faces_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def glob_glob(pat: str) -> List[str]:
    import glob
    return glob.glob(pat)

def extract_expected_minmax(item: Dict[str, Any]) -> Tuple[int, int]:
    pf = item.get("primary_faces", {}) or {}
    mn = int(pf.get("min_ids", 0) or 0)
    mx = int(pf.get("max_ids", mn) or mn)
    if mx < mn:
        mx = mn
    return mn, mx


# ----------------------------
# Main batch eval
# ----------------------------
def evaluate_category(
    category: str,
    label_items: List[Dict[str, Any]],
    root_videos: str,
    analyzer: MultiFaceConsistencyV2,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (per_video_results, summary)
    """
    results = []
    counted_scores = []

    skipped_no_face_label = 0
    skipped_missing_video = 0
    skipped_no_faces_detected = 0
    failed = 0

    for idx, item in enumerate(label_items, 1):
        content = item.get("content", "")
        prompt = item.get("prompt", "")
        mn, mx = extract_expected_minmax(item)

        # If Gemini thinks no human faces, skip (per your requirement)
        if mx == 0:
            skipped_no_face_label += 1
            continue

        title = safe_filename(content)
        print(f"[{category}] {idx}/{len(label_items)}  {title}", flush=True)
        video_path = os.path.join(root_videos, category, f"{title}.mp4")
        if not os.path.exists(video_path):
            skipped_missing_video += 1
            results.append({
                "category": category,
                "content": content,
                "video_path": video_path,
                "status": "missing_video",
                "expected_primary_faces": {"min_ids": mn, "max_ids": mx},
            })
            continue

        try:
            all_faces = analyzer.process_video(
                video_path,
                sample_rate=cfg["sample_rate"],
                min_face_px=cfg["min_face_px"],
                min_det_score=cfg["min_det_score"],
                min_rel_area=cfg["min_rel_area"],
            )

            if len(all_faces) == 0:
                skipped_no_faces_detected += 1
                results.append({
                    "category": category,
                    "content": content,
                    "video_path": video_path,
                    "status": "no_faces_detected",
                    "expected_primary_faces": {"min_ids": mn, "max_ids": mx},
                })
                continue

            tracklets = analyzer.build_tracklets(
                all_faces,
                max_gap_frames=cfg["max_gap_frames"],
                min_iou=cfg["min_iou"],
                min_cos=cfg["min_cos"],
            )
            tracklet_recs = analyzer.tracklet_embeddings(
                all_faces,
                tracklets,
                min_len=cfg["min_tracklet_len"],
            )
            if len(tracklet_recs) == 0:
                skipped_no_faces_detected += 1
                results.append({
                    "category": category,
                    "content": content,
                    "video_path": video_path,
                    "status": "no_tracklets",
                    "expected_primary_faces": {"min_ids": mn, "max_ids": mx},
                })
                continue

            out = analyzer.cluster_tracklets(
                tracklet_recs,
                eps=cfg["dbscan_eps"],
                min_samples=cfg["dbscan_min_samples"],
            )
            if out is None:
                skipped_no_faces_detected += 1
                results.append({
                    "category": category,
                    "content": content,
                    "video_path": video_path,
                    "status": "no_clusters",
                    "expected_primary_faces": {"min_ids": mn, "max_ids": mx},
                })
                continue

            identities, tracklet_recs, labels, used_eps = out
            analyzer.propagate_cluster_to_faces(all_faces, tracklet_recs)
            metrics = analyzer.consistency_metrics(all_faces, tracklet_recs)

            scored = compute_final_score(
                expected_min=mn,
                expected_max=mx,
                metrics=metrics,
                primary_ratio_thresh=cfg["primary_ratio_thresh"],
                min_cluster_dets=cfg["min_primary_cluster_dets"],
            )

            rec = {
                "category": category,
                "content": content,
                "prompt": prompt,
                "video_path": video_path,
                "status": "ok",
                "expected_primary_faces": {"min_ids": mn, "max_ids": mx},
                "dbscan": {"eps": float(used_eps), "min_samples": int(cfg["dbscan_min_samples"])},
                "metrics": metrics,
                **scored
            }
            results.append(rec)
            counted_scores.append(float(scored["final_score"]))
        except Exception as e:
            failed += 1
            results.append({
                "category": category,
                "content": content,
                "prompt": prompt,
                "video_path": video_path,
                "status": "failed",
                "error": str(e),
                "expected_primary_faces": {"min_ids": mn, "max_ids": mx},
            })

    summary = {
        "category": category,
        "count_total_labels": int(len(label_items)),
        "count_scored": int(len(counted_scores)),
        "mean_score": float(sum(counted_scores) / len(counted_scores)) if counted_scores else None,
        "skipped_no_face_label": int(skipped_no_face_label),
        "skipped_missing_video": int(skipped_missing_video),
        "skipped_no_faces_detected": int(skipped_no_faces_detected),
        "failed": int(failed),
    }
    return results, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_dir", type=str, required=True,
                    help="Directory containing Gemini-labeled json files (e.g., *.expected_faces.json)")
    ap.add_argument("--root_videos", type=str, required=True,
                    help="Root dir containing per-category video folders")
    ap.add_argument("--out_json", type=str, default="eval_results.json",
                    help="Where to write per-video results + summaries")
    ap.add_argument("--ctx_id", type=int, default=0)
    ap.add_argument("--det_size", type=str, default="640,640")

    # pipeline cfg
    ap.add_argument("--sample_rate", type=int, default=1)
    ap.add_argument("--min_face_px", type=int, default=30)
    ap.add_argument("--min_det_score", type=float, default=0.6)
    ap.add_argument("--min_rel_area", type=float, default=0.0005)

    ap.add_argument("--max_gap_frames", type=int, default=3)
    ap.add_argument("--min_iou", type=float, default=0.1)
    ap.add_argument("--min_cos", type=float, default=0.35)
    ap.add_argument("--min_tracklet_len", type=int, default=3)

    ap.add_argument("--dbscan_eps", type=float, default=0.65)
    ap.add_argument("--dbscan_min_samples", type=int, default=1)

    # primary selection + scoring
    ap.add_argument("--primary_ratio_thresh", type=float, default=0.10,
                    help="Cluster det ratio threshold to be considered primary")
    ap.add_argument("--min_primary_cluster_dets", type=int, default=30,
                    help="Min detections in a cluster to be considered primary")

    args = ap.parse_args()

    det_w, det_h = [int(x) for x in args.det_size.split(",")]
    analyzer = MultiFaceConsistencyV2(model_name="buffalo_l", ctx_id=args.ctx_id, det_size=(det_w, det_h))

    cfg = {
        "sample_rate": args.sample_rate,
        "min_face_px": args.min_face_px,
        "min_det_score": args.min_det_score,
        "min_rel_area": args.min_rel_area,
        "max_gap_frames": args.max_gap_frames,
        "min_iou": args.min_iou,
        "min_cos": args.min_cos,
        "min_tracklet_len": args.min_tracklet_len,
        "dbscan_eps": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
        "primary_ratio_thresh": args.primary_ratio_thresh,
        "min_primary_cluster_dets": args.min_primary_cluster_dets,
    }

    label_files = sorted(glob_glob(os.path.join(args.prompts_dir, "*.json")))
    if not label_files:
        raise SystemExit(f"No label json files found in {args.prompts_dir}")

    all_results = []
    summaries = []
    overall_scores = []

    for lf in label_files:
        category = os.path.basename(lf).split(".", 1)[0]  # e.g. ads.expected_faces.json -> ads
        label_items = load_expected_faces_file(lf)

        per_video, summary = evaluate_category(
            category=category,
            label_items=label_items,
            root_videos=args.root_videos,
            analyzer=analyzer,
            cfg=cfg
        )

        all_results.extend(per_video)
        summaries.append(summary)
        if summary["mean_score"] is not None:
            for r in per_video:
                if r.get("status") == "ok":
                    overall_scores.append(float(r.get("final_score", 0.0)))

    overall = {
        "count_scored_total": int(len(overall_scores)),
        "mean_score_total": float(sum(overall_scores) / len(overall_scores)) if overall_scores else None
    }

    out = {
        "config": cfg,
        "summaries_by_category": summaries,
        "overall": overall,
        "results": all_results
    }
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # print concise report
    print("=== Category Means ===")
    for s in summaries:
        print(f"{s['category']:>20s}  scored={s['count_scored']:4d}  mean={s['mean_score']}")
    print("=== Overall ===")
    print(f"scored_total={overall['count_scored_total']}  mean_total={overall['mean_score_total']}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
