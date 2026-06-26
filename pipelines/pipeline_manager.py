# pipelines/pipeline_manager.py
import os
from pipelines.pipeline_autosplice import analyze_autosplice
from pipelines.pipeline_dis25k import analyze_dis25k
from PIL import Image

# ================================================
# Unified Pipeline Manager
# ================================================

def analyze_image(image_path, mode="dis25k", show=False, verbose=False):
    """
    Unified interface for forensic analysis.
    Supports:
        mode = "autosplice"  → run AutoSplice model only
        mode = "dis25k"      → run DIS25k model only
        mode = "both"        → run both and compare
    Returns:
        dict with:
            - label
            - confidence
            - triggered_techniques (list)
            - visuals (numpy arrays)
            - reasons (dict)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    mode = mode.lower()
    result = {}

    def extract_triggered(reasons):
        return [k for k, v in reasons.items() if v]

    if mode == "autosplice":
        if verbose: print("[INFO] Running AutoSplice pipeline...")
        result = analyze_autosplice(image_path, show=show)
        result["triggered_techniques"] = extract_triggered(result.get("reasons", {}))

    elif mode == "dis25k":
        if verbose: print("[INFO] Running DIS25k pipeline...")
        result = analyze_dis25k(image_path, show=show)
        result["triggered_techniques"] = extract_triggered(result.get("reasons", {}))

    elif mode == "both":
        if verbose: print("[INFO] Running both pipelines...")
        a_result = analyze_autosplice(image_path, show=show)
        d_result = analyze_dis25k(image_path, show=show)

        a_result["triggered_techniques"] = extract_triggered(a_result.get("reasons", {}))
        d_result["triggered_techniques"] = extract_triggered(d_result.get("reasons", {}))

        comparison = {
            "autosplice_label": a_result["label"],
            "dis25k_label": d_result["label"],
            "autosplice_confidence": a_result["confidence"],
            "dis25k_confidence": d_result["confidence"],
            "agreement": a_result["label"].split()[0] == d_result["label"].split()[0]
        }

        result = {
            "autosplice": a_result,
            "dis25k": d_result,
            "comparison": comparison
        }

    else:
        raise ValueError("Invalid mode. Choose 'autosplice', 'dis25k', or 'both'.")

    return result


# ================================================
# CLI Testing
# ================================================
if __name__ == "__main__":
    test_image = "./Forged_JPEG90/71695_0.jpg"  # update as needed
    print("[INFO] Testing unified pipeline manager...\n")

    out1 = analyze_image(test_image, mode="autosplice", show=False, verbose=True)
    out2 = analyze_image(test_image, mode="dis25k", show=False, verbose=True)
    out3 = analyze_image(test_image, mode="both", show=False, verbose=True)

    print("\n[AutoSplice Result]:", out1["label"], f"({out1['confidence']:.2f})", out1["triggered_techniques"])
    print("[DIS25k Result]:", out2["label"], f"({out2['confidence']:.2f})", out2["triggered_techniques"])
    print("\n[Combined Comparison]:", out3["comparison"])
