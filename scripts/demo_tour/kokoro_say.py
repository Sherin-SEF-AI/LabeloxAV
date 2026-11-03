"""Speak tour narration with Kokoro, as a long-lived helper the recorder talks to over stdin.

Kokoro (Apache-2.0 weights, kokoro-onnx MIT) reads far more naturally than the piper voice the first tour
used, but it needs its own Python environment, so `record.py` starts this once with that interpreter and
sends one JSON request per line: {"text": ..., "path": ...}. The reply is one JSON line with the duration
in seconds, measured from the written file rather than estimated.

    .scratch/demo/kokoro-venv/bin/python scripts/demo_tour/kokoro_say.py --model DIR --voice af_heart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="directory holding kokoro-v1.0.onnx and voices-v1.0.bin")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    import os

    import onnxruntime as ort
    import soundfile as sf
    from kokoro_onnx import Kokoro

    # Left alone, onnxruntime takes every core, which starves the browser being recorded at the same time.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("ORT_NUM_THREADS", "8"))
    session = ort.InferenceSession(str(a.model / "kokoro-v1.0.onnx"), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    tts = Kokoro.from_session(session, str(a.model / "voices-v1.0.bin"))
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        out = Path(req["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        samples, rate = tts.create(req["text"], voice=req.get("voice", a.voice), speed=a.speed, lang="en-us")
        sf.write(str(out), samples, rate, subtype="PCM_16")
        print(json.dumps({"path": str(out), "seconds": len(samples) / float(rate)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
