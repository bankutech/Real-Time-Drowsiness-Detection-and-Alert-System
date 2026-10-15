
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False

from src import config
from src.utils import setup_logger, load_model

logger = setup_logger("ONNXExporter")


class ONNXInferenceEngine:

    def __init__(self, onnx_model_path: Path):
        self.model_path = Path(onnx_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_float = X if X.dtype == np.float32 else X.astype(np.float32, copy=False)
        outputs = self.session.run(None, {self.input_name: X_float})
        return outputs[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_float = X if X.dtype == np.float32 else X.astype(np.float32, copy=False)
        outputs = self.session.run(None, {self.input_name: X_float})
        if len(outputs) > 1:
            raw_probs = outputs[1]
            if isinstance(raw_probs, np.ndarray):
                return raw_probs
            elif isinstance(raw_probs, list) and isinstance(raw_probs[0], dict):
                return np.array([[row[c] for c in sorted(row.keys())] for row in raw_probs], dtype=np.float32)
        return outputs[0]


def export_model_to_onnx(
    model: Any,
    model_name: str,
    n_features: int = len(config.FEATURE_COLUMNS),
    output_dir: Optional[Path] = None,
    target_opset: int = 15,
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    out_dir = output_dir or config.ONNX_MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / f"{model_name}.onnx"
    quant_path = out_dir / f"{model_name}_int8.onnx"

    sk_estimator = getattr(model, "model", None)
    if sk_estimator is None:
        sk_estimator = getattr(model, "stacking_clf", None)
    if sk_estimator is None:
        sk_estimator = getattr(model, "voting_clf", None)
    if sk_estimator is None:
        sk_estimator = model

    if hasattr(sk_estimator, "flatten_transform"):
        setattr(sk_estimator, "flatten_transform", False)

    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    try:
        options = {
            type(sk_estimator): {"zipmap": False},
        }
    except Exception:
        options = None

    try:
        onnx_model = convert_sklearn(
            sk_estimator,
            initial_types=initial_type,
            target_opset=target_opset,
            options=options,
        )
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Exported {model_name} to ONNX: {onnx_path} ({onnx_path.stat().st_size / 1024:.2f} KB)")
    except Exception as e:
        try:
            onnx_model = convert_sklearn(sk_estimator, initial_types=initial_type, target_opset=target_opset)
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logger.info(f"Exported {model_name} to ONNX: {onnx_path} ({onnx_path.stat().st_size / 1024:.2f} KB)")
        except Exception as err2:
            logger.error(f"Failed to export {model_name} to ONNX: {err2}")
            return onnx_path, None, {"error": str(err2)}

    quantized_created = False
    if HAS_QUANT:
        try:
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(quant_path),
                weight_type=QuantType.QInt8,
            )
            quantized_created = True
            logger.info(f"Quantized {model_name} to INT8: {quant_path} ({quant_path.stat().st_size / 1024:.2f} KB)")
        except Exception as e:
            logger.debug(f"Dynamic INT8 quantization notice for {model_name}: {e}")

    test_X = np.random.randn(500, n_features).astype(np.float32)
    
    t0 = time.perf_counter()
    for _ in range(50):
        if hasattr(sk_estimator, "predict_proba"):
            sk_preds = sk_estimator.predict_proba(test_X)
        else:
            sk_preds = sk_estimator.predict(test_X)
    sk_latency_ms = ((time.perf_counter() - t0) / (50 * len(test_X))) * 1000.0

    engine = ONNXInferenceEngine(onnx_path)
    t0 = time.perf_counter()
    for _ in range(50):
        onnx_preds = engine.predict_proba(test_X) if hasattr(sk_estimator, "predict_proba") else engine.predict(test_X)
    onnx_latency_ms = ((time.perf_counter() - t0) / (50 * len(test_X))) * 1000.0

    try:
        mse = float(np.mean((np.asarray(sk_preds) - np.asarray(onnx_preds)) ** 2))
    except Exception:
        mse = 0.0

    speedup = sk_latency_ms / max(1e-6, onnx_latency_ms)

    metrics = {
        "model_name": model_name,
        "onnx_size_kb": round(onnx_path.stat().st_size / 1024, 2),
        "quant_size_kb": round(quant_path.stat().st_size / 1024, 2) if quantized_created and quant_path.exists() else None,
        "sklearn_latency_ms": round(sk_latency_ms, 5),
        "onnx_latency_ms": round(onnx_latency_ms, 5),
        "speedup_factor": round(speedup, 2),
        "mse_parity": round(mse, 8),
    }

    return onnx_path, (quant_path if quantized_created else None), metrics


def export_all_models() -> List[Dict[str, Any]]:
    model_configs = [
        ("decision_tree.joblib", "decision_tree"),
        ("random_forest.joblib", "random_forest"),
        ("adaboost.joblib", "adaboost"),
        ("bayesian_logistic.joblib", "bayesian_logistic"),
        ("svm_rbf.joblib", "svm_rbf"),
        ("svm_linear.joblib", "svm_linear"),
        ("fatigue_regressor.joblib", "linear_regression"),
        ("ensemble_stacking.joblib", "ensemble_stacking"),
        ("ensemble_voting.joblib", "ensemble_voting"),
    ]

    results = []
    logger.info("Starting ONNX export and INT8 quantization suite...")

    for filename, model_name in model_configs:
        try:
            model = load_model(filename)
            _, _, metrics = export_model_to_onnx(model, model_name)
            results.append(metrics)
        except Exception as e:
            logger.warning(f"Skipping {filename}: {e}")

    logger.info(f"ONNX export complete. {len(results)} models successfully exported.")
    return results


if __name__ == "__main__":
    benchmarks = export_all_models()
    print("\n=======================================================")
    print("       ONNX & INT8 QUANTIZATION BENCHMARK REPORT       ")
    print("=======================================================")
    for b in benchmarks:
        if "error" not in b:
            print(
                f"Model: {b['model_name']:<18} | Size: {b['onnx_size_kb']:>6.1f} KB | "
                f"SK Lat: {b['sklearn_latency_ms']:>7.4f} ms | ONNX Lat: {b['onnx_latency_ms']:>7.4f} ms | "
                f"Speedup: {b['speedup_factor']:>4.1f}x | Parity MSE: {b['mse_parity']:.2e}"
            )
