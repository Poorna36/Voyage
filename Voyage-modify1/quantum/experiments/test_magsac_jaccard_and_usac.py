import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import cv2

# Set up test data with noise and outliers
rng = np.random.default_rng(12345)
N = 60
s_xy = rng.uniform(50, 450, size=(N, 2)).astype(np.float32)
r_xy = s_xy + np.array([15.0, -10.0], dtype=np.float32) + rng.normal(0, 1.2, size=(N, 2)).astype(np.float32)
# 25 outliers with gross displacement
r_xy[35:] = rng.uniform(0, 500, size=(25, 2)).astype(np.float32)

def jaccard(set_a, set_b):
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 1.0

# 1. Test cv2.setRNGSeed across 10 seeds
masks_seeds = []
for s in [1, 42, 100, 2026, 9999, 12345, 54321, 777, 888, 999]:
    cv2.setRNGSeed(s)
    H, mask = cv2.findHomography(s_xy, r_xy, cv2.USAC_MAGSAC, 3.0)
    inliers = set(np.where(mask.ravel() == 1)[0])
    masks_seeds.append(inliers)

jaccards_seed = []
for i in range(len(masks_seeds)):
    for j in range(i + 1, len(masks_seeds)):
        jaccards_seed.append(jaccard(masks_seeds[i], masks_seeds[j]))

print(f"--- 1. Varying cv2.setRNGSeed (10 seeds) ---")
print(f"Mean Jaccard: {np.mean(jaccards_seed):.6f}, Min Jaccard: {np.min(jaccards_seed):.6f}, Max Jaccard: {np.max(jaccards_seed):.6f}")
print(f"All 10 seeds bit-for-bit identical masks? {all(j == 1.0 for j in jaccards_seed)}")

# 2. Test row permutations across 10 permutations
masks_shuff = []
for p_seed in range(10):
    perm = np.random.default_rng(p_seed * 111 + 7).permutation(N)
    s_perm = s_xy[perm]
    r_perm = r_xy[perm]
    cv2.setRNGSeed(42)
    H, mask = cv2.findHomography(s_perm, r_perm, cv2.USAC_MAGSAC, 3.0)
    # Map back to original indices
    inliers_orig = set(perm[np.where(mask.ravel() == 1)[0]])
    masks_shuff.append(inliers_orig)

jaccards_shuff = []
for i in range(len(masks_shuff)):
    for j in range(i + 1, len(masks_shuff)):
        jaccards_shuff.append(jaccard(masks_shuff[i], masks_shuff[j]))

print(f"\n--- 2. Row Permutations (10 permutations) ---")
print(f"Mean Jaccard: {np.mean(jaccards_shuff):.6f}, Min Jaccard: {np.min(jaccards_shuff):.6f}, Max Jaccard: {np.max(jaccards_shuff):.6f}")
print(f"Inlier set sizes across permutations: {[len(m) for m in masks_shuff]}")

# 3. Test cv2.UsacParams
p = cv2.UsacParams()
print(f"\n--- 3. cv2.UsacParams Exploration ---")
attrs = [a for a in dir(p) if not a.startswith("__")]
print("UsacParams fields:", attrs)
print("Default randomGeneratorState:", p.randomGeneratorState)

# Test whether randomGeneratorState controls seeding
usac_masks_by_state = []
for state_val in [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]:
    params = cv2.UsacParams()
    params.randomGeneratorState = state_val
    params.threshold = 3.0
    H, mask = cv2.findHomography(s_xy, r_xy, params)
    inliers = set(np.where(mask.ravel() == 1)[0])
    usac_masks_by_state.append(inliers)

jaccards_usac_state = []
for i in range(len(usac_masks_by_state)):
    for j in range(i + 1, len(usac_masks_by_state)):
        jaccards_usac_state.append(jaccard(usac_masks_by_state[i], usac_masks_by_state[j]))

print(f"UsacParams varying randomGeneratorState (10 states):")
print(f"Mean Jaccard: {np.mean(jaccards_usac_state):.6f}, Min: {np.min(jaccards_usac_state):.6f}, Max: {np.max(jaccards_usac_state):.6f}")
print(f"Are masks different across different randomGeneratorState? {any(j < 1.0 for j in jaccards_usac_state)}")

# Test reproducibility: same state twice
p1 = cv2.UsacParams()
p1.randomGeneratorState = 42
p1.threshold = 3.0
_, m1 = cv2.findHomography(s_xy, r_xy, p1)

p2 = cv2.UsacParams()
p2.randomGeneratorState = 42
p2.threshold = 3.0
_, m2 = cv2.findHomography(s_xy, r_xy, p2)

print(f"Reproducibility with fixed randomGeneratorState=42 across 2 runs: identical? {np.array_equal(m1, m2)}")
