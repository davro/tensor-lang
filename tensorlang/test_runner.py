# tensorlang/test_runner.py
import os
import sys
import time
import json
import re
import numpy as np
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from tensorlang.tensor_verifier import TensorVerifier
except ImportError:
    print ("TensorVerifier failed to find class")
    TensorVerifier = None

# ================================================================
#                    ANSI Color Codes
# ================================================================
GREEN = "32"
RED = "31"
YELLOW = "33"
CYAN = "36"

class TestRunner:
    def __init__(self, parallel=True, jobs=None, verify_tensors=False, debug_mode=False):
        self.parallel = parallel
        self.jobs = jobs
        self.verify_tensors = verify_tensors
        self.debug_mode = debug_mode
    
    def color(self, text, code):
        """Apply ANSI color to text."""
        return f"\033[{code}m{text}\033[0m"

    def discover_tests(self):
        """Discover all .tl test files in tests directory."""
        test_dir = Path("tests")
        tl_files = list(test_dir.glob("*.tl"))
        return sorted([f.name for f in tl_files])

    
    def run_single_test(self, test_file, suite_start_time=None):
        """Run a single test and verify results using .npy files.
        
        Returns tuple: (test_file, success, timing_str, failure_reason, report_data)
        """
        original_dir = os.getcwd()
        
        try:
            test_start_abs = time.time()
            
            # Extract expected results from .tl file
            tl_path = Path("tests") / test_file
            expected_results = self.extract_expected_from_tl(tl_path)
            
            if expected_results is None:
                duration = time.time() - test_start_abs
                timing_str = self._format_timing(duration, test_start_abs, suite_start_time)
                return (test_file, False, timing_str, "No @EXPECTED block found in .tl file", None)
            
            # Run tensorlang.py in normal mode (not test mode)
            cmd = ["python3", "tensorlang.py", f"tests/{test_file}", "--cache-layers"]
            if self.debug_mode:
                cmd.append("--debug")
            
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                check=False, 
                text=True
            )
            
            test_end_abs = time.time()

            cache_dir = Path(f"cache/tests/{test_file}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Save log for debugging
            test_stem = Path(test_file).stem
            log_file = cache_dir / f"{test_stem}.log"
            with open(log_file, "w") as f:
                f.write(process.stdout)

            if process.returncode != 0:
                duration = test_end_abs - test_start_abs
                timing_str = self._format_timing(duration, test_start_abs, suite_start_time)
                return (test_file, False, timing_str, f"Compiler exited with code {process.returncode}", None)

            # Phase 1: Load and compare .npy files
            for variable_name, expected_value in expected_results.items():
                computed = self.load_npy_result(cache_dir, variable_name)
                
                if computed is None:
                    duration = test_end_abs - test_start_abs
                    timing_str = self._format_timing(duration, test_start_abs, suite_start_time)
                    return (test_file, False, timing_str, f"Missing variable '{variable_name}' (.npy file not found)", None)
                
                success, error_msg = self.compare_results(computed, expected_value, variable_name)
                if not success:
                    duration = test_end_abs - test_start_abs
                    timing_str = self._format_timing(duration, test_start_abs, suite_start_time)
                    return (test_file, False, timing_str, f"{variable_name}: {error_msg}", None)

            duration = test_end_abs - test_start_abs
            
            # Phase 2: Verify tensor integrity (if enabled)
            verification_report = None
            tensor_summary = {
                "passed": len(expected_results),
                "failed": 0,
                "warnings": 0,
                "total": len(expected_results)
            }
            
            if self.verify_tensors and TensorVerifier:
                verifier = TensorVerifier()
                computed_tensors = {
                    name: self.load_npy_result(cache_dir, name)
                    for name in expected_results.keys()
                }
                computed_tensors = {k: v for k, v in computed_tensors.items() if v is not None}
                
                all_passed, verification_report = verifier.verify_all_tensors(
                    computed_tensors,
                    expected_results,
                    str(cache_dir),
                    {}
                )
                
                if verification_report and "summary" in verification_report:
                    tensor_summary = verification_report["summary"]
                
                if not all_passed:
                    timing_str = self._format_timing(duration, test_start_abs, suite_start_time)
                    return (test_file, False, timing_str, "Tensor verification failed", (verification_report, tensor_summary))
            
            timing_str = self._format_timing(duration, test_start_abs, suite_start_time)
            return (test_file, True, timing_str, None, (verification_report, tensor_summary))

        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            sys.exit(1)


    def run_test_suite(self, test_files):
        """Run test suite with optional parallelization."""

        if not test_files:
            print("No tests to run.")
            return
        
        verify_note = " (tensor verification enabled)" if self.verify_tensors else ""
        start_suite = time.time()
        results = []
        
        if not self.parallel:
            print(self.color(f"Running {len(test_files)} tests sequentially{verify_note}...\n", CYAN))
            for test_file in test_files:
                test_result = self.run_single_test(test_file, self.verify_tensors, suite_start_time=None, debug_mode=self.debug_mode)
                results.append(test_result)
        else:
            worker_jobs = self.jobs or min(len(test_files), os.cpu_count() or 4)
            print(self.color(f"Running {len(test_files)} tests in parallel (jobs={worker_jobs}){verify_note}...\n", CYAN))

            suite_start_time_parallel = time.time()

            with ThreadPoolExecutor(max_workers=worker_jobs) as executor:

                future_to_test = {
                    executor.submit(
                        self.run_single_test, 
                        test,
                        suite_start_time=suite_start_time_parallel
                    ): test 
                    for test in test_files
                }

                iterator = as_completed(future_to_test)
                
                if tqdm:
                    pbar = tqdm(total=len(test_files), desc="Running tests", ncols=80, position=0, leave=True)

                for future in iterator:
                    test_result = future.result()
                    
                    if tqdm:
                        pbar.update(1)
                    
                    results.append(test_result)
                
                if tqdm:
                    pbar.close()

        duration_suite = time.time() - start_suite
        passed = sum(1 for _, ok, _, _, _ in results if ok)
        total = len(results)
        failed_tests = [(t, fr, rd) for t, ok, _, fr, rd in results if not ok]

        print("\n" + "=" * 80)
        is_parallel = self.parallel
        suite_start_for_sorting = start_suite if is_parallel else None
        self.print_results_table(results, self.verify_tensors, suite_start_for_sorting)
        print("=" * 80)
        
        print(self.color(f"Summary: {passed}/{total} tests passed in {duration_suite:.2f}s", CYAN))
        
        if passed == total:
            print(self.color("✅ All tests passed successfully!", GREEN))
        else:
            print(self.color(f"❌ {len(failed_tests)} test(s) failed.", RED))
        
        if failed_tests:
            print("\n" + self.color("Failures:", RED))
            for test_name, failure_reason, report_data in failed_tests:
                print(f"\n  ❌ {test_name}")
                if failure_reason:
                    print(f"     Reason: {failure_reason}")
                
                if report_data and isinstance(report_data, tuple):
                    v_report, _ = report_data
                    if v_report and "tensors" in v_report:
                        for tensor_name, tensor_report in v_report["tensors"].items():
                            if not tensor_report["passed"]:
                                for check_name, check_result in tensor_report["checks"].items():
                                    if check_result["passed"] is False:
                                        print(f"     • {tensor_name} ({check_name}): {check_result['message']}")
                
                log_path = f"cache/tests/{test_name}/{Path(test_name).stem}.log"
                print(f"     📋 Check logs: {log_path}")
        
        if self.verify_tensors:
            total_tensor_warnings = 0
            total_tensor_failures = 0
            
            for _, _, _, _, report_data in results:
                if report_data:
                    if isinstance(report_data, tuple):
                        _, tensor_summary = report_data
                    else:
                        tensor_summary = report_data.get("summary", {})
                    
                    total_tensor_warnings += tensor_summary.get("warnings", 0)
                    total_tensor_failures += tensor_summary.get("failed", 0)
            
            if total_tensor_warnings > 0 or total_tensor_failures > 0:
                issues = []
                if total_tensor_failures > 0:
                    issues.append(f"{total_tensor_failures} tensor failures")
                if total_tensor_warnings > 0:
                    issues.append(f"{total_tensor_warnings} tensor warnings")
                print(self.color(f"\n⚠️  Tensor verification: {', '.join(issues)}", YELLOW))
        
        print("=" * 80)

    
    def extract_expected_from_tl(self, tl_file_path):
        """Extract @EXPECTED block from a .tl file.
        
        Returns dict of expected results, or None if not found or parse error.
        """
        try:
            with open(tl_file_path, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Failed to read {tl_file_path}: {e}")
            return None
        
        # Find @EXPECTED marker
        expected_match = re.search(r"//\s*@EXPECTED\s*\n", content)
        if not expected_match:
            return None
        
        start_pos = expected_match.end()
        lines = content[start_pos:].split("\n")
        
        json_lines = []
        for line in lines:
            # Match comment lines with content
            comment_match = re.match(r"//\s*(.*)", line)
            if comment_match:
                json_lines.append(comment_match.group(1))
            else:
                # Stop at first non-comment line
                break
        
        if not json_lines:
            return None
        
        json_str = "\n".join(json_lines)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse @EXPECTED JSON in {tl_file_path}: {e}")
            return None


    def _extract_start_time(self, timing_str):
        """Extract start time from timing string for sorting."""
        if timing_str.startswith("["):
            try:
                start_str = timing_str.split(" → ")[0].replace("[", "").replace("s", "")
                return float(start_str)
            except:
                return 0
        return 0


    def _strip_ansi(self, text):
        """Remove ANSI color codes for length calculation."""
        ansi_escape = re.compile(r'\033\[[0-9;]*m')
        return ansi_escape.sub('', text)


    def load_npy_result(self, cache_dir, variable_name):
        """Load a single .npy file for a variable."""
        npy_path = Path(cache_dir) / f"{variable_name}.npy"
        
        if not npy_path.exists():
            return None
        
        try:
            return np.load(npy_path)
        except Exception as e:
            print(f"Warning: Failed to load {npy_path}: {e}")
            return None


    def compare_results(self, computed, expected, variable_name):
        """Compare computed result with expected result.
        
        Returns: (success: bool, error_message: str or None)
        """
        if isinstance(expected, (int, float)):
            if isinstance(computed, np.ndarray):
                if computed.size == 1:
                    computed = computed.item()
                else:
                    return False, f"Expected scalar, got array of shape {computed.shape}"
            
            if not np.isclose(computed, expected, rtol=1e-5, atol=1e-8):
                return False, f"Expected {expected}, got {computed}"
        else:
            try:
                expected_arr = np.asarray(expected)
                computed_arr = np.asarray(computed)
                
                if computed_arr.shape != expected_arr.shape:
                    return False, f"Shape mismatch: expected {expected_arr.shape}, got {computed_arr.shape}"
                
                if not np.allclose(computed_arr, expected_arr, rtol=1e-5, atol=1e-8):
                    max_diff = np.abs(computed_arr - expected_arr).max()
                    return False, f"Values differ (max diff: {max_diff})"
            except Exception as e:
                return False, f"Comparison error: {str(e)}"
        
        return True, None


    def _format_timing(self, duration, test_start_abs, suite_start_time):
        """Format timing string based on whether we're in parallel or sequential mode."""
        if suite_start_time is None:
            return f"({duration:.2f}s)"
        else:
            test_start_rel = test_start_abs - suite_start_time
            test_end_rel = test_start_rel + duration
            return f"[{test_start_rel:.1f}s → {test_end_rel:.1f}s]"


    def print_results_table(self, results, verify_tensors, suite_start_time=None):
        """Print results in a formatted table, sorted by execution start time."""
        if not results:
            return
        
        if suite_start_time is not None:
            results = sorted(results, key=lambda x: self._extract_start_time(x[2]))
        
        state_width = 5
        name_width = max(len(t) for t, _, _, _, _ in results) + 2
        name_width = min(name_width, 60)
        time_width = 15
        tensor_width = 12
        
        header = f"{'State':<{state_width}} {'TestCase':<{name_width}}"
        if verify_tensors:
            header += f" {'Tensors':<{tensor_width}}"
        header += f"{'Time':<{time_width}}"
        
        print(header)
        print("-" * 80)
        
        for test_file, success, message, failure_reason, report_data in results:
            test_display = test_file if len(test_file) <= name_width - 2 else test_file[:name_width - 5] + "..."
            
            status_str = self.color("PASS", GREEN) if success else self.color("FAIL", RED)
            time_str = message if message else "N/A"

            tensor_str = ""
            if verify_tensors:
                if report_data:
                    if isinstance(report_data, tuple):
                        v_report, tensor_summary = report_data
                        s = tensor_summary
                    else:
                        s = report_data.get("summary", {}) if report_data else {}
                    
                    if s.get("failed", 0) > 0:
                        tensor_str = self.color(f"✗ {s['failed']}/{s['total']}", RED)
                    elif s.get("warnings", 0) > 0:
                        tensor_str = self.color(f"⚠ {s['warnings']} warn", YELLOW)
                    else:
                        total = s.get("total", 0)
                        passed = s.get("passed", 0)
                        tensor_str = self.color(f"✓ {passed}/{total}", GREEN)
                else:
                    tensor_str = "-"
            
            row = f"{status_str:<{state_width+10}} {test_display:<{name_width}}"
            if verify_tensors:
                visible_len = len(self._strip_ansi(tensor_str))
                padding = max(0, 12 - visible_len)
                row += f" {tensor_str}{' ' * padding}"
            row += f" {time_str}"
            print(row)