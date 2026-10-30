import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import cv2
from quantum.benchmark import make_benchmark_pair, ALL_CONDITIONS

# SIFT keypoint extractor + Mutual NN matching without Lowe ratio test
sift = cv2.SIFT_create(nfeatures=200)

conditions = ALL_CONDITIONS
n_seeds = 10

print("Measuring raw candidate matches and positive fraction at 3px across 12 conditions x 10 seeds...")

results_by_cond = {}

for cond in conditions:
    counts = []
    pos_fractions = []
    
    for s in range(n_seeds):
        pair = make_benchmark_pair(condition=cond, seed=s)
        src_img = pair["src"]
        ref_img = pair["ref"]
        H_gt = pair["H_true"]
        
        # SIFT keypoints & descriptors
        kp1, des1 = sift.detectAndCompute(src_img, None)
        kp2, des2 = sift.detectAndCompute(ref_img, None)
        
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            counts.append(0)
            pos_fractions.append(0.0)
            continue
            
        # Mutual Nearest Neighbor without ratio test
        # Descriptors are float32
        dists = np.linalg.norm(des1[:, None, :] - des2[None, :, :], axis=2)
        nn_12 = np.argmin(dists, axis=1)
        nn_21 = np.argmin(dists, axis=0)
        
        mutual_matches = []
        for i, j in enumerate(nn_12):
            if nn_21[j] == i:
                mutual_matches.append((i, j))
                
        n_cand = len(mutual_matches)
        counts.append(n_cand)
        
        if n_cand > 0:
            pts1 = np.array([kp1[i].pt for i, j in mutual_matches], dtype=np.float64)
            pts2 = np.array([kp2[j].pt for i, j in mutual_matches], dtype=np.float64)
            
            # Reprojection error under H_gt: H_gt @ [pts1, 1]
            pts1_h = np.hstack([pts1, np.ones((n_cand, 1))])
            pred_h = (H_gt @ pts1_h.T).T
            pred_pts2 = pred_h[:, :2] / (pred_h[:, 2:3] + 1e-12)
            
            reproj_err = np.linalg.norm(pts2 - pred_pts2, axis=1)
            pos_count = int(np.sum(reproj_err <= 3.0))
            pos_fractions.append(pos_count / n_cand)
        else:
            pos_fractions.append(0.0)
            
    results_by_cond[cond] = {
        "mean_count": float(np.mean(counts)),
        "std_count": float(np.std(counts)),
        "mean_pos_frac": float(np.mean(pos_fractions)),
        "std_pos_frac": float(np.std(pos_fractions)),
    }

print(f"\n{'Condition':>20} | {'Raw Candidates (mean ± std)':>30} | {'Pos Fraction @ 3px':>22}")
print("-" * 78)
for cond, r in results_by_cond.items():
    print(f"{cond:>20} | {r['mean_count']:6.1f} ± {r['std_count']:4.1f} matches        | {r['mean_pos_frac']*100:6.1f}% ± {r['std_pos_frac']*100:4.1f}%")
