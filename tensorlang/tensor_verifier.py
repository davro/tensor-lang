"""
Tensor Integrity Verification Module
Validates computed tensors against cached .npy references
Captures metadata: shape, dtype, statistics, hash, operation type
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


class TensorMetadata:
    """Container for tensor metadata and verification results."""

    def __init__(self, name: str, array: np.ndarray, op_type: str = "unknown"):
        self.name = name
        self.array = array
        self.op_type = op_type
        self.shape = array.shape
        self.dtype = str(array.dtype)
        self.size = array.size

    def compute_stats(self) -> Dict[str, Any]:
        """Compute statistical properties."""
        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "size": self.size,
            "min": float(np.min(self.array)) if self.size > 0 else None,
            "max": float(np.max(self.array)) if self.size > 0 else None,
            "mean": float(np.mean(self.array)) if self.size > 0 else None,
            "std": float(np.std(self.array)) if self.size > 0 else None,
            "nan_count": int(np.isnan(self.array).sum()),
            "inf_count": int(np.isinf(self.array).sum()),
        }

    def compute_hash(self) -> str:
        """Compute SHA256 hash of tensor data."""
        return hashlib.sha256(self.array.tobytes()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dict."""
        return {
            "name": self.name,
            "op_type": self.op_type,
            "stats": self.compute_stats(),
            "hash": self.compute_hash(),
        }


class TensorVerifier:
    """Verifies computed tensors against expected results and .npy cache."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        self.rtol = rtol
        self.atol = atol
        self.verification_log: Dict[str, Any] = {}

    def load_npy_file(self, npy_path: str) -> Optional[np.ndarray]:
        """Load .npy file, return None if not found."""
        try:
            return np.load(npy_path)
        except FileNotFoundError:
            return None
        except Exception as e:
            return None

    def compare_arrays(
        self,
        computed: np.ndarray,
        expected: np.ndarray,
        name: str
    ) -> Tuple[bool, str]:
        """
        Compare two arrays with tolerance.
        Returns (is_match, message).
        """
        # Shape mismatch
        if computed.shape != expected.shape:
            return False, (
                f"Shape mismatch: expected {expected.shape}, got {computed.shape}"
            )

        # Check for NaN/Inf in computed
        nan_count = np.isnan(computed).sum()
        inf_count = np.isinf(computed).sum()

        if nan_count > 0:
            return False, f"Computed tensor contains {nan_count} NaN values"
        if inf_count > 0:
            return False, f"Computed tensor contains {inf_count} Inf values"

        # Numerical comparison
        try:
            matches = np.allclose(
                computed,
                expected,
                rtol=self.rtol,
                atol=self.atol
            )
            if not matches:
                max_diff = np.abs(computed - expected).max()
                return False, (
                    f"Values don't match within tolerance "
                    f"(max diff: {max_diff:.2e}, rtol={self.rtol}, atol={self.atol})"
                )
        except Exception as e:
            return False, f"Comparison error: {str(e)}"

        return True, "Values match within tolerance"

    def verify_tensor(
        self,
        computed: np.ndarray,
        expected: Optional[np.ndarray],
        name: str,
        op_type: str = "unknown",
        npy_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify a single tensor against expected value and/or .npy file.
        Returns verification report with metadata.
        """
        metadata = TensorMetadata(name, computed, op_type)
        stats = metadata.compute_stats()

        report = {
            "name": name,
            "op_type": op_type,
            "stats": stats,
            "hash": metadata.compute_hash(),
            "checks": {},
            "passed": True,
            "warnings": [],
        }

        # Check 1: Compare against expected results (scalar/array from JSON)
        if expected is not None:
            expected_arr = np.asarray(expected)
            match, msg = self.compare_arrays(computed, expected_arr, name)
            report["checks"]["expected_match"] = {
                "passed": match,
                "message": msg,
            }
            if not match:
                report["passed"] = False
        else:
            report["checks"]["expected_match"] = {
                "passed": None,
                "message": "No expected value provided (reference only)",
            }

        # Check 2: Load and compare against .npy cache file
        if npy_path:
            npy_tensor = self.load_npy_file(npy_path)
            if npy_tensor is not None:
                npy_match, npy_msg = self.compare_arrays(computed, npy_tensor, name)
                report["checks"]["npy_cache_match"] = {
                    "passed": npy_match,
                    "message": npy_msg,
                    "cache_file": npy_path,
                }
                if not npy_match:
                    report["passed"] = False
            else:
                report["checks"]["npy_cache_match"] = {
                    "passed": None,
                    "message": f".npy file not found at {npy_path}",
                    "cache_file": npy_path,
                }

        # Check 3: Warn on numerical issues
        if stats["nan_count"] > 0:
            report["warnings"].append(
                f"⚠️ Tensor contains {stats['nan_count']} NaN values"
            )
        if stats["inf_count"] > 0:
            report["warnings"].append(
                f"⚠️ Tensor contains {stats['inf_count']} Inf values"
            )

        # Check 4: Warn on extreme values
        if stats["max"] is not None:
            abs_max = max(abs(stats["min"]), abs(stats["max"]))
            if abs_max > 1e6:
                report["warnings"].append(
                    f"⚠️ Large values detected (max abs: {abs_max:.2e})"
                )

        self.verification_log[name] = report
        return report

    def verify_all_tensors(
        self,
        computed_tensors: Dict[str, np.ndarray],
        expected_results: Dict[str, Any],
        cache_dir: str,
        op_types: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify all tensors from a test.
        Returns (all_passed, detailed_report).
        """
        all_passed = True
        detailed_report = {
            "tensors": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
            }
        }

        op_types = op_types or {}

        for tensor_name, computed_array in computed_tensors.items():
            expected = expected_results.get(tensor_name)
            npy_path = str(Path(cache_dir) / f"{tensor_name}.npy")
            op_type = op_types.get(tensor_name, "unknown")

            report = self.verify_tensor(
                computed_array,
                expected,
                tensor_name,
                op_type,
                npy_path
            )

            detailed_report["tensors"][tensor_name] = report
            detailed_report["summary"]["total"] += 1

            if report["passed"]:
                detailed_report["summary"]["passed"] += 1
            else:
                detailed_report["summary"]["failed"] += 1
                all_passed = False

            if report["warnings"]:
                detailed_report["summary"]["warnings"] += len(report["warnings"])

        return all_passed, detailed_report

    def format_report(self, report: Dict[str, Any], tensor_name: str) -> str:
        """Format verification report for console output."""
        tensor_report = report["tensors"][tensor_name]
        lines = []

        lines.append(f"\n  📊 Tensor: {tensor_name}")
        lines.append(f"     Op Type: {tensor_report['op_type']}")

        stats = tensor_report["stats"]
        lines.append(f"     Shape: {stats['shape']}, dtype: {stats['dtype']}")
        lines.append(
            f"     Range: [{stats['min']:.4e}, {stats['max']:.4e}], "
            f"mean: {stats['mean']:.4e}"
        )

        # Check results
        for check_name, check_result in tensor_report["checks"].items():
            if check_result["passed"] is True:
                lines.append(f"     ✅ {check_name}: {check_result['message']}")
            elif check_result["passed"] is False:
                lines.append(f"     ❌ {check_name}: {check_result['message']}")
            else:
                lines.append(f"     ⓘ  {check_name}: {check_result['message']}")

        # Warnings
        for warning in tensor_report["warnings"]:
            lines.append(f"     {warning}")

        return "\n".join(lines)

    def get_test_summary(self, report: Dict[str, Any]) -> str:
        """Get summary string for test results."""
        s = report["summary"]
        passed_str = f"✅ {s['passed']}/{s['total']}"
        if s["warnings"] > 0:
            passed_str += f" ({s['warnings']} ⚠️)"
        if s["failed"] > 0:
            passed_str += f" ❌ {s['failed']} failed"
        return passed_str
