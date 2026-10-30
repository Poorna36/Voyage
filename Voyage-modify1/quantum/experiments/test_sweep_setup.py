import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import cv2
import time

# Let's test the sweep
canvases = [100, 200, 400, 800]
s_values = [5, 10, 20, 30, 50, 100]
outlier_models = [f"near-miss_s={s}" for s in s_values] + ["gross"]

print("Canvases:", canvases)
print("Outlier models:", outlier_models)
