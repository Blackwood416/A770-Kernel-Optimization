#!/usr/bin/env python3
"""Unit tests for the correctness-contract and compare helpers."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from compare_outputs import (  # noqa: E402
    compare_files,
    parse_stdout,
    validate_correctness_contract,
)


class CorrectnessContractTest(unittest.TestCase):
    def test_matched(self) -> None:
        compare = {
            "status": "PASS",
            "errors": 0,
            "total": 4,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "rel_tol": 1e-4,
            "abs_tol": 1e-4,
            "semantics_id": "gemv_n1_mkn_rowmajor",
            "relaxed_accuracy": False,
        }
        status, cls = validate_correctness_contract(
            compare, 1e-4, 1e-4, expected_semantics_id="gemv_n1_mkn_rowmajor"
        )
        self.assertEqual((status, cls), ("PASS", "matched"))

    def test_tolerance_mismatch(self) -> None:
        compare = {
            "status": "PASS",
            "errors": 0,
            "total": 4,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "rel_tol": 5e-2,
            "abs_tol": 1e-2,
            "semantics_id": "gemv_n1_mkn_rowmajor",
            "relaxed_accuracy": False,
        }
        status, cls = validate_correctness_contract(compare, 1e-4, 1e-4)
        self.assertEqual(status, "CORRECTNESS_CONTRACT_MISMATCH")
        self.assertEqual(cls, "unknown")

    def test_invalid_fail_without_semantics(self) -> None:
        compare = {
            "status": "FAIL",
            "errors": 100,
            "total": 4,
            "max_abs_err": 3.0,
            "max_rel_err": 1.0,
            "rel_tol": 1e-4,
            "abs_tol": 1e-4,
            "semantics_id": None,
            "relaxed_accuracy": False,
        }
        status, cls = validate_correctness_contract(compare, 1e-4, 1e-4)
        self.assertEqual((status, cls), ("FAIL", "invalid"))

    def test_fastest_only_requires_semantics_and_relaxed(self) -> None:
        compare = {
            "status": "FAIL",
            "errors": 1,
            "total": 4,
            "max_abs_err": 0.1,
            "max_rel_err": 0.1,
            "rel_tol": 5e-2,
            "abs_tol": 1e-2,
            "semantics_id": "gemv_n1_mkn_rowmajor",
            "relaxed_accuracy": True,
        }
        status, cls = validate_correctness_contract(compare, 5e-2, 1e-2)
        self.assertEqual((status, cls), ("FAIL_RELAXED", "fastest_only"))

    def test_relaxed_matched(self) -> None:
        compare = {
            "status": "PASS",
            "errors": 0,
            "total": 4,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "rel_tol": 5e-2,
            "abs_tol": 1e-2,
            "semantics_id": "matmul_mkn_rowmajor",
            "relaxed_accuracy": True,
        }
        status, cls = validate_correctness_contract(compare, 5e-2, 1e-2)
        self.assertEqual((status, cls), ("PASS_RELAXED", "relaxed_matched"))

    def test_semantics_mismatch(self) -> None:
        compare = {
            "status": "PASS",
            "errors": 0,
            "total": 4,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "rel_tol": 1e-4,
            "abs_tol": 1e-4,
            "semantics_id": "matmul_nkm_transpose",
            "relaxed_accuracy": False,
        }
        status, cls = validate_correctness_contract(
            compare, 1e-4, 1e-4, expected_semantics_id="matmul_mkn_rowmajor"
        )
        self.assertEqual((status, cls), ("SEMANTICS_MISMATCH", "invalid"))


class CompareFilesTest(unittest.TestCase):
    def test_length_mismatch_is_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            actual = root / "actual.bin"
            expected = root / "expected.bin"
            actual.write_bytes(b"\x00" * (2048 * 4))
            expected.write_bytes(b"\x00" * (4096 * 4))
            result = compare_files(actual, expected, dtype="f32")
            self.assertEqual(result["status"], "SHAPE_MISMATCH")
            self.assertEqual(result["actual_count"], 2048)
            self.assertEqual(result["expected_count"], 4096)

    def test_explicit_prefix_with_short_input_is_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            actual = root / "actual.bin"
            expected = root / "expected.bin"
            actual.write_bytes(b"\x00" * (2048 * 4))
            expected.write_bytes(b"\x00" * (4096 * 4))
            result = compare_files(actual, expected, dtype="f32", count=4096)
            self.assertEqual(result["status"], "SHAPE_MISMATCH")


class ParseStdoutTest(unittest.TestCase):
    def test_parses_json_contract(self) -> None:
        text = (
            '{"errors":0,"total":4,"max_abs_err":0.001,'
            '"max_rel_err":0.002,"rel_tol":0.05,"abs_tol":0.01,'
            '"reference":"cpu_f16_precast","semantics_id":"matmul_mkn_rowmajor",'
            '"accuracy_mode":"strict","relaxed_accuracy":false}\n'
        )
        result = parse_stdout(text)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["rel_tol"], 0.05)
        self.assertEqual(result["semantics_id"], "matmul_mkn_rowmajor")


if __name__ == "__main__":
    unittest.main()
