import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import cv2
from quantum.benchmark import make_benchmark_pair, ALL_CONDITIONS
from src.matching.sift import SIFTMatcher
from src.matching.rift import RIFT2Matcher

sift_matcher = SIFTMatcher()
rift_matcher = RIFT2Matcher()

conditions = ALL_CONDITIONS
n_seeds = 10

winners_count = {"sift": 0, "rift2": 0, "tie": 0, "both_failed": 0}
condition_breakdown = {}

print("Measuring class balance of best-matcher labels (SIFT vs RIFT2) across 12 conditions x 10 seeds...")

for cond in conditions:
    cond_wins = {"sift": 0, "rift2": 0, "tie": 0, "both_failed": 0}
    for s in range(n_seeds):
        pair = make_benchmark_pair(condition=cond, seed=s)
        src = pair["src"]
        ref = pair["ref"]
        H_gt = pair["H_true"]
        
        # 1. Run SIFT
        res_sift = sift_matcher.match(src, ref)
        s_inliers = 0
        s_rmse = float("inf")
        if res_sift.count >= 4:
            pts1 = res_sift.src_xy
            pts2 = res_sift.ref_xy
            pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
            pred_h = (H_gt @ pts1_h.T).T
            pred_pts2 = pred_h[:, :2] / (pred_h[:, 2:3] + 1e-12)
            err = np.linalg.norm(pts2 - pred_pts2, axis=1)
            inl_idx = np.where(err <= 3.0)[0]
            s_inliers = len(inl_idx)
            if s_inliers >= 4:
                s_rmse = float(np.sqrt(np.mean(err[inl_idx] ** 2)))
                
        # 2. Run RIFT2
        res_rift = rift_matcher.match(src, ref)
        r_inliers = 0
        r_rmse = float("inf")
        if res_rift.count >= 4:
            pts1 = res_rift.src_xy
            pts2 = res_rift.ref_xy
            pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
            pred_h = (H_gt @ pts1_h.T).T
            pred_pts2 = pred_h[:, :2] / (pred_h[:, 2:3] + 1e-12)
            err = np.linalg.norm(pts2 - pred_pts2, axis=1)
            inl_idx = np.where(err <= 3.0)[0]
            r_inliers = len(inl_idx)
            if r_inliers >= 4:
                r_rmse = float(np.sqrt(np.mean(err[inl_idx] ** 2)))
                
        # Determine winner:
        # Matcher with more inliers wins; if inliers close (within 10%), lower RMSE wins
        if s_inliers < 4 and r_inliers < 4:
            winner = "both_failed"
        elif s_inliers >= 4 and r_inliers < 4:
            winner = "sift"
        elif r_inliers >= 4 and s_inliers < 4:
            winner = "rift2"
        else:
            if s_inliers > r_inliers:
                winner = "sift"
            elif r_inliers > s_inliers:
                winner = "rift2"
            else:
                if s_rmse < r_rmse:
                    winner = "sift"
                elif r_rmse < s_rmse:
                    winner = "rift2"
                else:
                    winner = "tie"
                    
        winners_count[winner] += 1
        cond_wins[winner] += 1
        
    condition_breakdown[cond] = cond_wins

print(f"\nOverall Winner Distribution (120 scenes):")
for k, v in winners_count.items():
    print(f"  {k}: {v} ({v / 120 * 100:.1f}%)")

print(f"\nPer-Condition Breakdown:")
print(f"{'Condition':>20} | {'SIFT':>6} | {'RIFT2':>6} | {'Failed':>7}")
print("-" * 46)
for cond, counts in condition_breakdown.items():
    print(f"{cond:>20} | {counts['sift']:6d} | {counts['rift2']:6d} | {counts['both_failed']:7d}")
