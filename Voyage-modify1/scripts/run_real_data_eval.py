#!/usr/bin/env python3
"""
scripts/run_real_data_eval.py
==============================
End-to-End Real Data Accuracy Evaluation (SIH 2026 PS-26166 Presentation Metrics)

Runs the full OHRC + TMC-2 pipeline against the two downloaded Chandrayaan-2 granules:
  - OHR (OHRC):  ch2_ohr_ncp_20211228T2209123959_d_img_d18   (0.32 m/px)
  - TMC-2 NCF:   ch2_tmc_ncf_20220613T1623247403_d_img_d32   (5 m/px)

Pipeline stages:
  S1  PDS4 label ingestion & metadata extraction
  S2  L1 Radiometric preprocessing
  S3  Cross-sensor pair formation (OHR -> TMC)
  S4  Multi-matcher correspondence (SIFT | RIFT2 | LightGlue)
  S5  Geometric ladder + MAGSAC++
  S6  Sub-pixel residual evaluation & RMSE
  S7  GT warp accuracy benchmark

Output: results/real_eval/metrics_report.json
        results/real_eval/presentation_metrics.txt
        results/real_eval/*.png  (QC visualizations)

Usage:
    python scripts/run_real_data_eval.py
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
import cv2, numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATASET_BASE = Path(r"D:\neo\hachathon\SIH 2026\yolo\real_dataset")
OHR_DIR = DATASET_BASE / "ohrc" / "ch2_ohr_ncp_20211228T2209123959_d_img_d18"
TMC_DIR = DATASET_BASE / "tmc" / "ch2_tmc_ncf_20220613T1623247403_d_img_d32"

OHR_XML = OHR_DIR / "data/calibrated/20211228/ch2_ohr_ncp_20211228T2209123959_d_img_d18.xml"
OHR_BRW = OHR_DIR / "browse/calibrated/20211228/ch2_ohr_ncp_20211228T2209123959_b_brw_d18.png"

TMC_XML = TMC_DIR / "data/calibrated/20220613/ch2_tmc_ncf_20220613T1623247403_d_img_d32.xml"
TMC_BRW = TMC_DIR / "browse/calibrated/20220613/ch2_tmc_ncf_20220613T1623247403_b_brw_d32.png"

OUT_DIR = _ROOT / "results" / "real_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OHR_GSD_M = 0.32
TMC_GSD_M = 5.0


def section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def stage_s1_ingest():
    section("S1: PDS4 LABEL INGESTION")
    from src.ingest.label_parser import parse_pds4_label
    results = {}
    for name, xml_path in [("OHR", OHR_XML), ("TMC_NCF", TMC_XML)]:
        if not xml_path.exists():
            print(f"  [SKIP] {name}: XML not found at {xml_path}")
            results[name] = {"status": "XML_NOT_FOUND"}
            continue
        try:
            meta = parse_pds4_label(str(xml_path))
            lats = [c[1] for c in meta.footprint_ll]
            lons = [c[0] for c in meta.footprint_ll]
            gates = {
                "footprint_ok": len(meta.footprint_ll) >= 3,
                "solar_ok": meta.solar_incidence_deg is not None and meta.solar_incidence_deg > 0,
                "gsd_ok": meta.gsd_m is not None and meta.gsd_m > 0,
                "utc_ok": bool(meta.utc),
            }
            r = {
                "status": "OK",
                "product_id": meta.product_id,
                "sensor": meta.sensor,
                "utc": meta.utc,
                "gsd_m": meta.gsd_m,
                "solar_incidence_deg": meta.solar_incidence_deg,
                "solar_azimuth_deg": meta.solar_azimuth_deg,
                "footprint_shape": list(meta.footprint_shape),
                "lat_range": [round(min(lats),4), round(max(lats),4)],
                "lon_range": [round(min(lons),4), round(max(lons),4)],
                "gates": gates,
                "all_gates_passed": all(gates.values()),
            }
            results[name] = r
            print(f"  [{name}] {meta.product_id}")
            print(f"    GSD: {meta.gsd_m:.4f} m/px | Solar: {meta.solar_incidence_deg:.2f} | UTC: {meta.utc}")
            print(f"    Footprint: {meta.footprint_shape[0]}x{meta.footprint_shape[1]} | S1 Gates: {'ALL PASSED' if r['all_gates_passed'] else 'SOME FAILED'}")
        except Exception as e:
            results[name] = {"status": "ERROR", "error": str(e)}
            print(f"  [ERROR] {name}: {e}")
    return results


def stage_s2_preprocess():
    section("S2: L1 RADIOMETRIC PREPROCESSING")
    from src.preprocessing.masks import shadow_mask
    from src.preprocessing.normalize import percentile_clip
    from src.preprocessing.branches import apply_tmc_wac
    from src.preprocessing.stats import compute_texture_contrast, compute_mean_gradient
    results = {}
    for name, brw_path, gsd in [("OHR", OHR_BRW, OHR_GSD_M), ("TMC_NCF", TMC_BRW, TMC_GSD_M)]:
        if not brw_path.exists():
            print(f"  [SKIP] {name}: Browse PNG not at {brw_path}")
            results[name] = {"status": "NOT_FOUND"}
            continue
        img_raw = cv2.imread(str(brw_path), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            results[name] = {"status": "READ_ERROR"}
            continue
        h, w = img_raw.shape
        y0 = max(0, h//2 - 512)
        patch = img_raw[y0:y0+1024, :]
        ph, pw = patch.shape
        smask = shadow_mask(patch, solar_incidence_deg=45.0)
        valid_frac = float(np.mean(~smask)) * 100.0
        norm_img = percentile_clip(patch, lo=2.0, hi=98.0)
        contrast_raw = compute_texture_contrast(patch)
        grad_raw = compute_mean_gradient(patch)
        processed = apply_tmc_wac(norm_img, ref=None, config={"clahe_clip_limit": 2.0, "clahe_tile_grid": [8, 8], "histogram_match": False})
        contrast_proc = compute_texture_contrast((processed * 255).astype(np.uint8))
        grad_proc = compute_mean_gradient((processed * 255).astype(np.uint8))
        cv2.imwrite(str(OUT_DIR / f"{name.lower()}_raw_patch.png"), patch)
        cv2.imwrite(str(OUT_DIR / f"{name.lower()}_shadow_mask.png"), smask.astype(np.uint8)*255)
        cv2.imwrite(str(OUT_DIR / f"{name.lower()}_processed.png"), (processed*255).astype(np.uint8))
        r = {
            "status": "OK",
            "image_size": [w, h],
            "gsd_m": gsd,
            "valid_terrain_pct": round(valid_frac, 2),
            "texture_contrast_raw": round(float(contrast_raw), 4),
            "texture_contrast_processed": round(float(contrast_proc), 4),
            "mean_gradient_raw": round(float(grad_raw), 4),
            "mean_gradient_processed": round(float(grad_proc), 4),
            "contrast_improvement_pct": round((contrast_proc - contrast_raw) / max(contrast_raw, 1e-6) * 100, 2),
        }
        results[name] = r
        print(f"  [{name}] {w}x{h} px | Valid: {valid_frac:.1f}% | Contrast: {contrast_raw:.2f}->{contrast_proc:.2f} ({r['contrast_improvement_pct']:+.1f}%) | Grad: {grad_raw:.4f}->{grad_proc:.4f}")
    return results


def stage_s3s5_matching():
    section("S3-S5: CROSS-SENSOR MATCHING (OHR <-> TMC-2)")
    from src.matching.sift import SIFTMatcher
    from src.matching.rift import RIFT2Matcher
    from src.matching.lightglue import LightGlueMatcher
    from src.registration.ladder import model_ladder
    from src.preprocessing.normalize import percentile_clip
    ohr_full = cv2.imread(str(OHR_BRW), cv2.IMREAD_GRAYSCALE) if OHR_BRW.exists() else None
    tmc_full = cv2.imread(str(TMC_BRW), cv2.IMREAD_GRAYSCALE) if TMC_BRW.exists() else None
    if ohr_full is None or tmc_full is None:
        print("  [SKIP] Browse images not available.")
        return {"status": "SKIPPED", "reason": "browse_images_not_found"}
    scale = OHR_GSD_M / TMC_GSD_M
    ohr_h, ohr_w = ohr_full.shape
    ohr_rs = cv2.resize(ohr_full, (max(1, int(ohr_w*scale)), max(1, int(ohr_h*scale))), interpolation=cv2.INTER_AREA)
    def center_crop(img, size=512):
        h, w = img.shape
        y0, x0 = max(0, h//2-size//2), max(0, w//2-size//2)
        return img[y0:y0+min(size,h), x0:x0+min(size,w)]
    patch_size = min(512, ohr_rs.shape[0], ohr_rs.shape[1], tmc_full.shape[0], tmc_full.shape[1])
    src_img = (percentile_clip(center_crop(ohr_rs, patch_size), 2., 98.) * 255).astype(np.uint8)
    ref_img = (percentile_clip(center_crop(tmc_full, patch_size), 2., 98.) * 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "cross_ohr_patch.png"), src_img)
    cv2.imwrite(str(OUT_DIR / "cross_tmc_patch.png"), ref_img)
    print(f"  OHR patch {src_img.shape[1]}x{src_img.shape[0]} (scale={scale:.4f}) vs TMC patch {ref_img.shape[1]}x{ref_img.shape[0]}")
    ph, pw = src_img.shape
    matchers = {
        "SIFT":      SIFTMatcher({"num_keypoints": 4096, "ratio_thresh": 0.80}),
        "RIFT2":     RIFT2Matcher({"num_keypoints": 1024, "n_scale": 4, "n_orient": 6}),
        "LightGlue": LightGlueMatcher({"max_keypoints": 2048, "cpu_fallback": True}),
    }
    benchmarks = {}
    for mname, matcher in matchers.items():
        print(f"\n  -- {mname} --")
        t0 = time.time()
        try:
            mr = matcher.match(src_img, ref_img, gsd_ratio=scale)
            runtime = time.time() - t0
            n_raw = len(mr.src_xy) if mr.src_xy is not None else 0
            print(f"    Raw matches: {n_raw}  ({runtime:.3f}s)")
            if n_raw >= 4:
                conf = mr.confidence if mr.confidence is not None else np.ones(n_raw, dtype=np.float32)
                res = model_ladder(mr.src_xy, mr.ref_xy, confidence=conf,
                                   src_shape=(ph,pw), ref_shape=(ph,pw),
                                   src_gsd_m=OHR_GSD_M/scale, ref_gsd_m=TMC_GSD_M,
                                   latitude_center_deg=0.0, stop_on_rmse_below=2.0)
                rmse_m = res.rmse_px * TMC_GSD_M
                status = "PASS" if res.inlier_count >= 8 and res.rmse_px < 3.0 else "SUBOPTIMAL"
                print(f"    Model={res.model_type.upper()} Inliers={res.inlier_count}/{n_raw} ({res.inlier_ratio*100:.1f}%) RMSE={res.rmse_px:.4f}px/{rmse_m:.4f}m [{status}]")
                benchmarks[mname] = {
                    "raw_matches": n_raw, "inliers": res.inlier_count,
                    "inlier_ratio": round(res.inlier_ratio,4), "best_model": res.model_type,
                    "rmse_px": round(res.rmse_px,4), "rmse_m": round(rmse_m,4),
                    "runtime_s": round(runtime,3), "status": status,
                    "_model_matrix": res.model_matrix.tolist() if res.model_matrix is not None else None,
                    "_inlier_mask": res.inlier_mask.tolist() if res.inlier_mask is not None else None,
                    "_src_xy": mr.src_xy.tolist() if mr.src_xy is not None else [],
                    "_ref_xy": mr.ref_xy.tolist() if mr.ref_xy is not None else [],
                }
            else:
                benchmarks[mname] = {"raw_matches": n_raw, "inliers": 0, "inlier_ratio": 0.0,
                                     "runtime_s": round(runtime,3), "status": "INSUFFICIENT_MATCHES"}
        except Exception as e:
            traceback.print_exc()
            benchmarks[mname] = {"status": "ERROR", "error": str(e), "runtime_s": round(time.time()-t0,3)}
    return {"status": "OK", "patch_size_px": patch_size, "ohr_scale_factor": round(scale,4),
            "matchers": benchmarks, "_src_img": src_img, "_ref_img": ref_img}


def stage_s6s7_warp_accuracy():
    section("S6-S7: GT WARP ACCURACY (TMC SELF-REGISTRATION)")
    from src.matching.sift import SIFTMatcher
    from src.matching.rift import RIFT2Matcher
    from src.matching.lightglue import LightGlueMatcher
    from src.registration.ladder import model_ladder
    from src.preprocessing.normalize import percentile_clip
    img_src = cv2.imread(str(TMC_BRW), cv2.IMREAD_GRAYSCALE) if TMC_BRW.exists() else None
    if img_src is None:
        img_src = cv2.imread(str(OHR_BRW), cv2.IMREAD_GRAYSCALE) if OHR_BRW.exists() else None
    if img_src is None:
        return {"status": "SKIPPED"}
    h, w = img_src.shape
    y0, x0 = max(0, h//2-256), max(0, w//2-256)
    base = img_src[y0:y0+512, x0:x0+512]
    base = (percentile_clip(base, 2., 98.) * 255).astype(np.uint8)
    bh, bw = base.shape
    # GT affine perturbation
    angle_deg, scale_f, dx, dy = -3.5, 1.03, 15.0, -10.0
    M_gt = cv2.getRotationMatrix2D((bw/2., bh/2.), angle_deg, scale_f)
    M_gt[0,2] += dx; M_gt[1,2] += dy
    ref_img = cv2.warpAffine(base, M_gt, (bw,bh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    _, xs = np.mgrid[0:bh, 0:bw]
    ref_img = np.clip(ref_img.astype(np.float32) * (0.85 + 0.30*(xs/bw)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "warp_src.png"), base)
    cv2.imwrite(str(OUT_DIR / "warp_ref_gt.png"), ref_img)
    matchers = {
        "SIFT":      SIFTMatcher({"num_keypoints": 4096, "ratio_thresh": 0.80}),
        "RIFT2":     RIFT2Matcher({"num_keypoints": 1024, "n_scale": 4, "n_orient": 6}),
        "LightGlue": LightGlueMatcher({"max_keypoints": 2048, "cpu_fallback": True}),
    }
    results = {}
    for mname, matcher in matchers.items():
        print(f"\n  -- {mname} (GT warp) --")
        t0 = time.time()
        try:
            mr = matcher.match(base, ref_img, gsd_ratio=1.0)
            runtime = time.time()-t0
            n_raw = len(mr.src_xy) if mr.src_xy is not None else 0
            print(f"    Raw matches: {n_raw}  ({runtime:.3f}s)")
            if n_raw >= 4:
                conf = mr.confidence if mr.confidence is not None else np.ones(n_raw, dtype=np.float32)
                res = model_ladder(mr.src_xy, mr.ref_xy, confidence=conf,
                                   src_shape=(bh,bw), ref_shape=(bh,bw),
                                   src_gsd_m=TMC_GSD_M, ref_gsd_m=TMC_GSD_M,
                                   latitude_center_deg=0.0, stop_on_rmse_below=1.5)
                rmse_m = res.rmse_px * TMC_GSD_M
                gt_acc = {}
                if res.model_matrix is not None and res.model_matrix.shape == (2,3):
                    M = res.model_matrix
                    sx = np.sqrt(M[0,0]**2+M[1,0]**2); sy = np.sqrt(M[0,1]**2+M[1,1]**2)
                    est_s = (sx+sy)/2.; est_a = np.degrees(np.arctan2(-M[1,0], M[0,0]))
                    tx_e, ty_e = M[0,2], M[1,2]
                    t_err_px = np.sqrt((tx_e-dx)**2+(ty_e-dy)**2)
                    gt_acc = {
                        "angle_error_deg": round(abs(est_a-angle_deg),4),
                        "scale_error": round(abs(est_s-scale_f),6),
                        "translation_error_px": round(t_err_px,4),
                        "translation_error_m": round(t_err_px*TMC_GSD_M,4),
                    }
                    print(f"    GT angle={angle_deg}deg est={est_a:.2f}deg err={gt_acc['angle_error_deg']:.3f}deg")
                    print(f"    GT tx,ty=({dx},{dy}) est=({tx_e:.2f},{ty_e:.2f}) trans_err={gt_acc['translation_error_px']:.3f}px")
                status = "PASS" if res.inlier_count >= 8 and res.rmse_px < 3.0 else "SUBOPTIMAL"
                print(f"    Model={res.model_type.upper()} Inliers={res.inlier_count}/{n_raw} ({res.inlier_ratio*100:.1f}%) RMSE={res.rmse_px:.4f}px/{rmse_m:.4f}m [{status}]")
                if res.model_matrix is not None:
                    if res.model_matrix.shape == (2,3):
                        warped = cv2.warpAffine(base, res.model_matrix, (bw,bh))
                    else:
                        warped = cv2.warpPerspective(base, res.model_matrix, (bw,bh))
                    tile = 64
                    checker = np.zeros_like(ref_img)
                    for yy in range(0,bh,tile):
                        for xx in range(0,bw,tile):
                            if ((xx//tile)+(yy//tile))%2==0: checker[yy:yy+tile,xx:xx+tile]=warped[yy:yy+tile,xx:xx+tile]
                            else: checker[yy:yy+tile,xx:xx+tile]=ref_img[yy:yy+tile,xx:xx+tile]
                    cv2.imwrite(str(OUT_DIR / f"{mname.lower()}_checker.png"), checker)
                    cv2.imwrite(str(OUT_DIR / f"{mname.lower()}_residuals.png"), cv2.applyColorMap(cv2.absdiff(warped,ref_img), cv2.COLORMAP_JET))
                results[mname] = {
                    "raw_matches": n_raw, "inliers": res.inlier_count,
                    "inlier_ratio": round(res.inlier_ratio,4), "best_model": res.model_type,
                    "rmse_px": round(res.rmse_px,4), "rmse_m": round(rmse_m,4),
                    "runtime_s": round(runtime,3), "status": status, "gt_accuracy": gt_acc,
                }
            else:
                results[mname] = {"raw_matches": n_raw, "inliers": 0, "inlier_ratio": 0.0,
                                  "runtime_s": round(runtime,3), "status": "INSUFFICIENT_MATCHES"}
        except Exception as e:
            traceback.print_exc()
            results[mname] = {"status": "ERROR", "error": str(e)}
    return {"status": "OK", "matchers": results}


def write_report(s1, s2, cross, warp, elapsed):
    section("PRESENTATION METRICS SUMMARY")
    lines = [
        "="*80,
        "  CHANDRAYAAN-2 REAL DATA ACCURACY EVALUATION REPORT",
        f"  SIH 2026 * PS-26166 * Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "="*80, "",
        "DATASETS:",
        f"  OHR (OHRC):   ch2_ohr_ncp_20211228T2209123959_d_img_d18  GSD={OHR_GSD_M} m/px",
        f"  TMC-2 NCF:    ch2_tmc_ncf_20220613T1623247403_d_img_d32  GSD={TMC_GSD_M} m/px",
        "",
        "S1 - PDS4 INGESTION & QUALITY GATES:",
    ]
    for sensor, rec in s1.items():
        if rec.get("status") == "OK":
            g = rec.get("gates", {}); gp = sum(g.values()); gt = len(g)
            lines.append(f"  {sensor:12s} GSD={rec['gsd_m']:.4f}m/px  Solar={rec['solar_incidence_deg']:.1f}deg  S1={gp}/{gt} gates PASSED")
        else:
            lines.append(f"  {sensor:12s} {rec.get('status')}")
    lines += ["", "S2 - RADIOMETRIC PREPROCESSING:"]
    for sensor, rec in s2.items():
        if rec.get("status") == "OK":
            lines.append(f"  {sensor:12s} Valid terrain={rec['valid_terrain_pct']:.1f}%  Contrast improvement={rec['contrast_improvement_pct']:+.1f}%  Gradient={rec['mean_gradient_raw']:.4f}->{rec['mean_gradient_processed']:.4f}")
        else:
            lines.append(f"  {sensor:12s} {rec.get('status')}")
    if cross.get("status") == "OK":
        lines += ["", "S3-S5 - CROSS-SENSOR MATCHING (OHR -> TMC-2):"]
        for mname, mrec in cross.get("matchers", {}).items():
            st = mrec.get("status","?")
            if mrec.get("inliers",0) > 0:
                lines.append(f"  {mname:12s} Raw={mrec['raw_matches']:4d}  Inliers={mrec['inliers']:4d} ({mrec['inlier_ratio']*100:.1f}%)  RMSE={mrec['rmse_px']:.3f}px/{mrec['rmse_m']:.3f}m  [{st}]")
            else:
                lines.append(f"  {mname:12s} Raw={mrec.get('raw_matches',0):4d}  [{st}]")
    if warp.get("status") == "OK":
        lines += ["", "S6-S7 - GT WARP ACCURACY (TMC self-registration, synthetic perturbation):"]
        lines.append("  GT: angle=-3.5deg, scale=1.03x, translation=(15,-10)px")
        for mname, mrec in warp.get("matchers", {}).items():
            st = mrec.get("status","?")
            if mrec.get("inliers",0) > 0:
                gt = mrec.get("gt_accuracy",{})
                lines.append(f"  {mname:12s} Inliers={mrec['inliers']:4d} ({mrec['inlier_ratio']*100:.1f}%)  RMSE={mrec['rmse_px']:.3f}px/{mrec['rmse_m']:.3f}m  [{st}]")
                if gt.get("angle_error_deg") is not None:
                    lines.append(f"               dAngle={gt['angle_error_deg']:.3f}deg  dScale={gt['scale_error']:.5f}  dTrans={gt['translation_error_px']:.3f}px / {gt['translation_error_m']:.3f}m")
            else:
                lines.append(f"  {mname:12s} [{st}]")
    lines += ["", f"Total evaluation time: {elapsed:.2f}s", f"QC artifacts: results/real_eval/", "="*80]
    report = "\n".join(lines)
    print(report)
    txt_path = OUT_DIR / "presentation_metrics.txt"
    txt_path.write_text(report, encoding="utf-8")
    print(f"\n-> Report saved: {txt_path}")
    def clean(d):
        if isinstance(d, dict): return {k: clean(v) for k,v in d.items() if not k.startswith("_")}
        if isinstance(d, list): return [clean(i) for i in d]
        return d
    json_path = OUT_DIR / "metrics_report.json"
    json_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_s": round(elapsed,2),
        "datasets": {"OHR": str(OHR_DIR), "TMC_NCF": str(TMC_DIR)},
        "s1_ingestion": clean(s1), "s2_preprocessing": clean(s2),
        "s3s5_cross_sensor_matching": clean(cross), "s6s7_warp_accuracy": clean(warp),
    }, indent=2), encoding="utf-8")
    print(f"-> JSON report: {json_path}")


def main():
    t0 = time.time()
    section("CHANDRAYAAN-2 REAL DATA ACCURACY EVALUATION")
    print(f"  OHR: {OHR_DIR}")
    print(f"  TMC: {TMC_DIR}")
    print(f"  Out: {OUT_DIR}")
    s1 = stage_s1_ingest()
    s2 = stage_s2_preprocess()
    cross = stage_s3s5_matching()
    # draw best match vis
    if cross.get("status") == "OK":
        src_img = cross.get("_src_img")
        ref_img = cross.get("_ref_img")
        bm = {k:v for k,v in cross.get("matchers",{}).items() if v.get("inliers",0)>0}
        if bm and src_img is not None and ref_img is not None:
            best_name = max(bm, key=lambda k: bm[k].get("inliers",0))
            best = bm[best_name]
            pts_s = np.array(best.get("_src_xy",[]), dtype=np.float32)
            pts_r = np.array(best.get("_ref_xy",[]), dtype=np.float32)
            imask = np.array(best.get("_inlier_mask",[]), dtype=bool)
            ph, pw = src_img.shape
            vis = np.zeros((ph, pw*2+20, 3), dtype=np.uint8)
            vis[:,:pw,:] = src_img[:,:,np.newaxis]; vis[:,pw+20:,:] = ref_img[:,:,np.newaxis]
            for i,(ps,pr) in enumerate(zip(pts_s,pts_r)):
                is_in = imask[i] if i < len(imask) else False
                c = (0,230,0) if is_in else (0,0,220)
                cv2.circle(vis,(int(ps[0]),int(ps[1])),3,c,-1)
                cv2.circle(vis,(int(pr[0])+pw+20,int(pr[1])),3,c,-1)
                cv2.line(vis,(int(ps[0]),int(ps[1])),(int(pr[0])+pw+20,int(pr[1])),c,1,cv2.LINE_AA)
            cv2.imwrite(str(OUT_DIR/f"cross_matches_{best_name.lower()}.png"), vis)
            print(f"\n  Best cross-sensor: {best_name} ({best.get('inliers',0)} inliers) -> cross_matches_{best_name.lower()}.png")
    warp = stage_s6s7_warp_accuracy()
    elapsed = time.time()-t0
    write_report(s1, s2, cross, warp, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
