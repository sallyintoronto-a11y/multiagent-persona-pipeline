# ============================================================
# Save / Run helpers + Batch
# ============================================================
def save_report(report: Dict[str, Any], path: str = "./outputs/report_langgraph_allinone.json"):
    Path(Path(path).parent).mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

IMG_PATTERNS = ("*.jpg","*.jpeg","*.png","*.webp","*.JPG","*.JPEG","*.PNG","*.WEBP")

def collect_images_from_dir(image_dir: str) -> List[str]:
    paths: List[str] = []
    for pat in IMG_PATTERNS:
        paths.extend(glob.glob(os.path.join(image_dir, pat)))
    return sorted(paths)

def run_batch(
    images: List[str],
    selected_pairs: Optional[List[str]] = None,
    out_dir: str = "./outputs"
) -> List[Dict[str, Any]]:
    """
    - Run workflow on all images
    - Save formats:
      1) Entire batch: out_dir/report_batch.json (list)
      2) Per-image full report: out_dir/by_image/<file>.json
      3) Per-pair cumulative report: out_dir/by_pair/report_<PAIR>.json (list)
      4) Per-pair & per-image single file: out_dir/by_pair/<PAIR>/<file>.json
    """
    wf = create_workflow(selected_pairs=selected_pairs)
    batch_reports: List[Dict[str, Any]] = []

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Per-pair cumulative memory (to reduce frequent file I/O)
    pair_accumulator: Dict[str, List[Dict[str, Any]]] = {}

    for idx, img in enumerate(images, 1):
        init: GraphState = {"image_path": img, "history": [], "pair_results": []}
        state = wf.invoke(init)
        report = state.get("final_report", {})  # {"image_path":..., "pairs":[...], "schemas":...}
        batch_reports.append({"image": img, "report": report})

        file_stem = os.path.splitext(os.path.basename(img))[0]

        # (2) Save per-image full report
        Path(os.path.join(out_dir, "by_image")).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(out_dir, "by_image", f"{file_stem}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # (3)(4) Save per-pair reports — iterate report["pairs"]
        pairs_list = (report or {}).get("pairs", [])
        for pair_result in pairs_list:
            pair_key = pair_result.get("pair", "UNKNOWN")
            pair_value = pair_result  # Include agent_outputs + evaluation

            # Per-pair cumulative memory
            pair_accumulator.setdefault(pair_key, []).append({"image": img, "result": pair_value})

            # Save per-pair & per-image single file
            pair_dir = os.path.join(out_dir, "by_pair", pair_key)
            Path(pair_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(pair_dir, f"{file_stem}.json"), "w", encoding="utf-8") as f:
                json.dump(pair_value, f, ensure_ascii=False, indent=2)

        print(f"[{idx}/{len(images)}] done: {os.path.basename(img)}")

    # (1) Save entire batch
    with open(os.path.join(out_dir, "report_batch.json"), "w", encoding="utf-8") as f:
        json.dump(batch_reports, f, ensure_ascii=False, indent=2)

    # (3) Save per-pair cumulative file
    Path(os.path.join(out_dir, "by_pair")).mkdir(parents=True, exist_ok=True)
    for pair_key, items in pair_accumulator.items():
        with open(os.path.join(out_dir, "by_pair", f"report_{pair_key}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Saved all reports → {out_dir}")
    return batch_reports

if __name__ == "__main__":
    # (Colab) Mount Drive if needed:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        pass

    DO_BATCH = True   # True: iterate a folder; False: single image

    if DO_BATCH:
        image_dir = "/content/drive/MyDrive/Weekly_seminar(July.2025~)/CulturalVLM/images"  # <--- change this
        images = collect_images_from_dir(image_dir)
        run_batch(images, out_dir="./outputs")
    else:
        image_path = "/content/drive/MyDrive/Weekly_seminar(July.2025~)/CulturalVLM/images/some_image.jpg"  # <--- change this
        wf = create_workflow()
        init = {"image_path": image_path, "history": [], "pair_results": []}
        result_state = wf.invoke(init)
        final = result_state.get("final_report", {})
        print(json.dumps(final, ensure_ascii=False, indent=2))
        Path("./outputs").mkdir(parents=True, exist_ok=True)
        save_report(final)
        print("Saved: ./outputs/report_langgraph_allinone.json")
