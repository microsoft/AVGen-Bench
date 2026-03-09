import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from collections import defaultdict
from tqdm import tqdm
import math

# ----------------------------
# Utility
# ----------------------------
def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def cosine_sim(a, b):
    # assumes already l2-normalized
    return float(np.dot(a, b))

def bbox_iou(a, b):
    # a,b: [x1,y1,x2,y2]
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

def bbox_area(b):
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def weighted_mean(embs, weights):
    embs = np.asarray(embs)
    w = np.asarray(weights).reshape(-1, 1)
    s = (embs * w).sum(axis=0) / max(w.sum(), 1e-12)
    return s

def robust_eps_from_knn_dist(X, k=5, q=0.9):
    """
    Simple automatic eps: for each point, compute distance to the k-th nearest
    neighbor and use quantile q as eps.
    X: (N,D) l2-normalized embeddings, metric=euclidean
    Note: this is a simplified O(N^2) version; optimize it for large N.
    """
    X = np.asarray(X)
    N = X.shape[0]
    if N <= k:
        return 0.7
    sims = np.clip(X @ X.T, -1.0, 1.0)
    dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sims))
    knn = np.sort(dists, axis=1)[:, k]
    return float(np.quantile(knn, q))

def percentile(arr, q):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))

def tracklet_internal_stability(tracklet_rec, all_faces, use_weights=True):
    """
    Tracklet internal stability:
    - Use tracklet mean embedding as center mu.
    - Compute cosine similarities between each detection embedding and mu.
    - Return mean/std/var/min/p05, where higher values indicate better stability.
    """
    idxs = tracklet_rec['indices']
    mu = tracklet_rec['embedding']  # already l2-normalized
    sims = []
    ws = []

    for j, i in enumerate(idxs):
        e = all_faces[i]['embedding']  # already l2-normalized
        sims.append(cosine_sim(e, mu))
        if use_weights:
            # Keep the same weighting as tracklet mean: det_score * sqrt(rel_area)
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

    # weighted mean/var
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


# ----------------------------
# Main
# ----------------------------
class MultiFaceConsistencyV2:
    def __init__(self, model_name='buffalo_l', ctx_id=0, det_size=(640, 640)):
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def process_video(self,
                      video_path,
                      sample_rate=1,
                      min_face_px=30,
                      min_det_score=0.6,
                      min_rel_area=0.0005):
        """
        Detect faces frame by frame and output raw detections
        (before tracking/clustering).
        """
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
        """
        A: Greedy tracking that links raw faces into tracklets.
        Output: tracklets as list[dict], and writes tracklet_id to each face.
        """
        if not all_faces:
            return []

        by_frame = defaultdict(list)
        for i, f in enumerate(all_faces):
            by_frame[f['frame_idx']].append(i)

        frames_sorted = sorted(by_frame.keys())

        tracklets = []
        active = {}  # tid -> {last_frame, last_bbox, last_emb, indices}

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
        """
        B: Compute weighted mean embedding for each tracklet
        (weights: det_score * sqrt(rel_area)).
        Returns tracklet_recs for clustering.
        """
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
                         min_samples=2,
                         auto_eps=False,
                         auto_eps_k=5,
                         auto_eps_q=0.9):
        """
        C: Cluster tracklet embeddings (more stable than per-frame points).
        """
        if not tracklet_recs:
            return None

        X = np.stack([r['embedding'] for r in tracklet_recs], axis=0)
        X = l2norm(X)

        if auto_eps:
            eps = robust_eps_from_knn_dist(X, k=auto_eps_k, q=auto_eps_q)

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
        """
        Propagate tracklet cluster_id back to each face detection.
        """
        tmap = {r['tracklet_id']: r.get('cluster_id', -1) for r in tracklet_recs}
        for f in all_faces:
            tid = f.get('tracklet_id', None)
            if tid is None:
                f['cluster_id'] = -1
            else:
                f['cluster_id'] = int(tmap.get(tid, -1))

    def consistency_metrics(self, all_faces, tracklet_recs):
        """
        D: Output identity consistency + stability metrics.
        Added:
        - tracklet_internal_stability: frame-level embedding variation inside each tracklet
        - cluster_internal_stability_from_tracklets: weighted aggregation of the above per cluster
        """
        if not all_faces:
            return {}

        # 1) coverage (placeholder)
        frames_with_face = {f['frame_idx'] for f in all_faces}
        sampled_frames = sorted(frames_with_face)
        _coverage_placeholder = 1.0 if len(sampled_frames) > 0 else 0.0

        # 2) cluster length statistics (accumulate tracklet n)
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

        # 3) ID switch (coarse, tracklet-level)
        seq = []
        for r in tracklet_recs:
            if r.get('cluster_id', -1) == -1:
                continue
            mid = 0.5 * (r['start_frame'] + r['end_frame'])
            seq.append((mid, int(r['cluster_id'])))
        seq.sort()

        switches = 0
        for i in range(1, len(seq)):
            if seq[i][1] != seq[i-1][1]:
                switches += 1

        # 4) intra-cluster stability (tracklet embedding to cluster center)
        by_c = defaultdict(list)
        for r in tracklet_recs:
            cid = int(r.get('cluster_id', -1))
            if cid != -1:
                by_c[cid].append(r)

        center = {}
        for cid, trs in by_c.items():
            X = np.stack([t['embedding'] for t in trs], axis=0)
            mu = l2norm(X.mean(axis=0))
            center[cid] = mu

        intra_cluster_stability = {}
        for cid, trs in by_c.items():
            mu = center[cid]
            sims = [cosine_sim(t['embedding'], mu) for t in trs]
            intra_cluster_stability[cid] = {
                'n_tracklets': len(trs),
                'mean_sim': float(np.mean(sims)) if sims else 0.0,
                'var_sim': float(np.var(sims)) if sims else 0.0,
            }

        # 5) tracklet internal stability (frame-level)
        tracklet_internal = []
        for r in tracklet_recs:
            # Remove this condition if you also want to include noise tracklets.
            if int(r.get('cluster_id', -1)) == -1:
                continue
            tracklet_internal.append(tracklet_internal_stability(r, all_faces, use_weights=True))

        # 6) aggregate tracklet internal stability by cluster (weight=tracklet length n)
        # We aggregate weighted mean_sim; std/var are not strictly merged here.
        # Expose weighted mean_sim + weighted p05_sim as conservative stability indicators.
        by_cluster_tracklet_stats = defaultdict(list)
        for st in tracklet_internal:
            tid = st['tracklet_id']
            # Find corresponding cluster and n.
            rec = next((r for r in tracklet_recs if int(r['tracklet_id']) == int(tid)), None)
            if rec is None:
                continue
            cid = int(rec.get('cluster_id', -1))
            if cid == -1:
                continue
            by_cluster_tracklet_stats[cid].append((st, int(rec['n'])))

        cluster_internal_from_tracklets = {}
        for cid, items in by_cluster_tracklet_stats.items():
            if not items:
                continue
            weights = np.asarray([w for _, w in items], dtype=np.float64)
            wsum = float(weights.sum())
            mean_sims = np.asarray([st['mean_sim'] for st, _ in items], dtype=np.float64)
            p05_sims = np.asarray([st['p05_sim'] for st, _ in items], dtype=np.float64)

            if wsum > 1e-12:
                w_mean = float((mean_sims * weights).sum() / wsum)
                w_p05 = float((p05_sims * weights).sum() / wsum)
            else:
                w_mean = float(np.mean(mean_sims))
                w_p05 = float(np.mean(p05_sims))

            cluster_internal_from_tracklets[cid] = {
                'n_tracklets': int(len(items)),
                'weighted_mean_of_tracklet_mean_sim': w_mean,
                'weighted_mean_of_tracklet_p05_sim': w_p05,
            }

        return {
            'total_face_dets': int(len(all_faces)),
            'total_tracklets': int(len(tracklet_recs)),
            'clusters': {int(k): int(v) for k, v in cluster_len.items()},
            'main_cluster_id': int(main_cluster),
            'main_cluster_ratio': float(main_ratio),
            'noise_ratio': float(noise_ratio),
            'id_switch_count_tracklet_level': int(switches),

            # Existing stability metric (tracklet to cluster center).
            'intra_cluster_stability': intra_cluster_stability,

            # Added: tracklet internal frame-level stability.
            'tracklet_internal_stability': tracklet_internal,

            # Added: cluster-level aggregation of tracklet internal stability.
            'cluster_internal_stability_from_tracklets': cluster_internal_from_tracklets,

            'coverage_note': 'coverage needs total sampled frame count; current is placeholder',
        }


# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    video_path = "/path/to/video_generation/sora2_generated/ads/Unskippable Elevator _ Insurance Ad Parody (Anonymized).mp4"

    analyzer = MultiFaceConsistencyV2(model_name='buffalo_l', ctx_id=0, det_size=(640, 640))

    print("Step 1: Detect faces (filtered by size & det_score)...")
    all_faces = analyzer.process_video(
        video_path,
        sample_rate=1,
        min_face_px=30,
        min_det_score=0.6,
        min_rel_area=0.0005
    )

    print(f"Detections: {len(all_faces)}")

    print("Step 2: Build tracklets (IoU + cosine)...")
    tracklets = analyzer.build_tracklets(
        all_faces,
        max_gap_frames=3,
        min_iou=0.1,
        min_cos=0.35
    )
    print(f"Tracklets raw: {len(tracklets)}")

    print("Step 3: Tracklet embeddings (weighted mean)...")
    tracklet_recs = analyzer.tracklet_embeddings(all_faces, tracklets, min_len=3)
    print(f"Tracklets used: {len(tracklet_recs)}")

    print("Step 4: Cluster tracklets (DBSCAN)...")
    out = analyzer.cluster_tracklets(
        tracklet_recs,
        eps=0.55,
        min_samples=1,
        auto_eps=False
    )

    if out is None:
        print("No tracklets to cluster.")
        raise SystemExit(0)

    identities, tracklet_recs, labels, used_eps = out
    print(f"DBSCAN eps={used_eps:.3f} -> identities: {list(identities.keys())}, noise_tracklets={(labels==-1).sum()}")

    print("Step 5: Propagate cluster_id back to detections...")
    analyzer.propagate_cluster_to_faces(all_faces, tracklet_recs)

    print("Step 6: Metrics...")
    metrics = analyzer.consistency_metrics(all_faces, tracklet_recs)
    print(metrics)
