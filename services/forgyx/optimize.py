"""FORGYX export, quantize, and compile. Each stage runs the real backend when it is installed and raises
CapabilityError when it is not, so a target is never faked. ONNX export uses ultralytics; INT8 PTQ uses ONNX
Runtime static quantization with a Release-drawn calibration set; per-target compile dispatches to TensorRT
(Jetson), LiteRT (Android/SentrixAI), or the Hailo compiler."""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from services.forgyx.capabilities import CapabilityError, require

log = get_logger("forgyx.optimize")


def export_onnx(weights_path: str, out_path: str, imgsz: int = 640, opset: int = 17) -> str:
    """Export a PyTorch/ultralytics model to ONNX with a fixed opset and dynamic batch axis."""
    require("onnx")
    from ultralytics import YOLO

    model = YOLO(weights_path)
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=True)
    src = Path(str(exported))
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    log.info("forgyx.onnx_export", weights=weights_path, out=str(dst), imgsz=imgsz)
    return str(dst)


def quantize_int8(onnx_path: str, calib_images: list, out_path: str, imgsz: int = 640) -> str:
    """Post-training INT8 static quantization using a representative calibration set drawn from a Release."""
    require("onnxruntime")
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self, images):
            self._it = iter(images)

        def get_next(self):
            img = next(self._it, None)
            if img is None:
                return None
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)[None]
            return {"images": arr}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    quantize_static(onnx_path, out_path, _Reader(calib_images), weight_type=QuantType.QInt8)
    log.info("forgyx.quantize", onnx=onnx_path, out=out_path, calib=len(calib_images))
    return out_path


def compile_target(onnx_path: str, target: str, out_path: str, int8: bool = True) -> str:
    """Compile an ONNX model for one edge target. Raises CapabilityError when the target's toolchain is absent
    (so the SentrixAI/Jetson/Hailo paths run only where the compiler and device exist)."""
    require(target)
    if target in ("agx_orin_trt", "orin_nano_trt"):
        return _compile_tensorrt(onnx_path, out_path, int8)
    if target == "sentrixai_litert":
        return _compile_litert(onnx_path, out_path)
    if target == "pi_hailo":
        return _compile_hailo(onnx_path, out_path)
    raise CapabilityError(f"no compiler for target {target}")


def _compile_tensorrt(onnx_path: str, out_path: str, int8: bool) -> str:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("TensorRT ONNX parse failed")
    config = builder.create_builder_config()
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
    engine = builder.build_serialized_network(network, config)
    Path(out_path).write_bytes(bytes(engine))
    return out_path


def _compile_litert(onnx_path: str, out_path: str) -> str:
    # ONNX -> LiteRT for SentrixAI (Android). Requires the ai-edge-litert converter toolchain.
    from ai_edge_litert.convert import convert_onnx  # type: ignore

    Path(out_path).write_bytes(convert_onnx(onnx_path))
    return out_path


def _compile_hailo(onnx_path: str, out_path: str) -> str:
    from hailo_sdk_client import ClientRunner  # type: ignore

    runner = ClientRunner()
    runner.translate_onnx_model(onnx_path)
    runner.save_har(out_path)
    return out_path
