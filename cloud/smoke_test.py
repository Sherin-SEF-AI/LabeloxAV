"""Cloud smoke test (runs ON the pod). Proves the three heavy models load on the GPU and produce non
empty output on one sample frame. Each model is isolated in try/except so one failure does not mask the
others. Writes /workspace/smoke_test_result.json with a PASS/FAIL verdict.

The SAM 3.1 and Qwen3-VL invocation APIs drift; these calls are best-effort and try a couple of entry
points. On first run, if one errors, the JSON captures the traceback so the call can be adjusted to the
installed package version. No em-dashes.
"""

from __future__ import annotations

import json
import traceback
import urllib.request
from pathlib import Path

SAMPLE = "/workspace/sample.jpg"
CKPT_SAM = "/workspace/ckpts/sam3p1"
CKPT_QWEN = "/workspace/ckpts/qwen3vl-8b"
RESULT = "/workspace/smoke_test_result.json"


def ensure_sample() -> str:
    if not Path(SAMPLE).exists():
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", SAMPLE)
    return SAMPLE


def test_yolo(img: str) -> dict:
    from ultralytics import YOLO

    last = None
    for w in ("yolo26n.pt", "yolo11n.pt"):
        try:
            model = YOLO(w)
            res = model.predict(img, device=0, verbose=False)
            n = int(sum(len(r.boxes) for r in res))
            return {"ok": n > 0, "weights": w, "box_count": n}
        except Exception as exc:  # noqa: BLE001
            last = f"{w}: {exc}"
    return {"ok": False, "error": last}


def test_sam(img: str) -> dict:
    # SAM 3.1 promptable concept segmentation. build_sam3_image_model() fetches the SAM 3.1 weights
    # (HF_TOKEN in env). Inference runs under autocast bf16 (the weights are bf16). "person" is a
    # concept present in the bus.jpg sample, so a working runtime returns masks.
    import sam3.model_builder as _mb
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    prompt = "person"
    # Pass bpe_path explicitly: the default pkg_resources lookup breaks for the editable (-e) sam3
    # install (the top-level sam3 module has __file__=None). The submodule file IS real, so locate
    # the bundled asset relative to it.
    bpe = str(Path(_mb.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe)
    proc = Sam3Processor(model)
    image = Image.open(img).convert("RGB")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(image)
        out = proc.set_text_prompt(state=state, prompt=prompt)
    masks = out["masks"]
    n = int(len(masks)) if masks is not None else 0
    return {"ok": n > 0, "prompt": prompt, "mask_count": n, "box_count": int(len(out["boxes"])),
            "entry": "Sam3Processor.set_text_prompt"}


def test_qwen(img: str) -> dict:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        proc = AutoProcessor.from_pretrained(CKPT_QWEN)
        model = AutoModelForImageTextToText.from_pretrained(CKPT_QWEN, torch_dtype=torch.bfloat16, device_map="cuda")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "List bounding boxes [x1,y1,x2,y2] of all traffic-relevant objects."}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=256)
        text = proc.batch_decode(out, skip_special_tokens=True)[0]
        return {"ok": len(text.strip()) > 0, "chars": len(text), "preview": text[:200]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{exc}", "trace": traceback.format_exc()[-600:]}


_TESTS = {"yolo26": test_yolo, "sam3p1": test_sam, "qwen3vl": test_qwen}


def _run_one(name: str) -> dict:
    img = ensure_sample()
    try:
        return _TESTS[name](img)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "trace": traceback.format_exc()[-600:]}


def main() -> None:
    import subprocess
    import sys

    # Each heavy model runs in its OWN process: isolates import side-effects (ultralytics pollutes
    # module state that SAM's pkg_resources lookup trips on) and keeps peak VRAM to one model at a time.
    results = {}
    for name in _TESTS:
        try:
            out = subprocess.run([sys.executable, __file__, "--only", name], capture_output=True, text=True, timeout=900)
            line = [ln for ln in out.stdout.strip().splitlines() if ln.startswith("{")]
            results[name] = json.loads(line[-1]) if line else {"ok": False, "error": out.stderr[-400:]}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "error": str(exc)}
    verdict = "PASS" if all(r.get("ok") for r in results.values()) else "FAIL"
    payload = {"verdict": verdict, "results": results}
    Path(RESULT).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"SMOKE TEST: {verdict}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        print(json.dumps(_run_one(sys.argv[2])))
    else:
        main()
