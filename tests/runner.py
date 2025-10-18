import os
import sys
import json
import subprocess
import re
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from tensor_verifier import TensorVerifier
except ImportError:
    print("❌ Error: tensor_verifier module not found. Make sure tensor_verifier.py is in the tests directory.")
    sys.exit(1)


# ==============================
# Configuration
# ==============================
ROOT_DIR = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent
EXPECTED_RESULTS_FILE = TEST_DIR / "expected_results.json"

try:
    with open(EXPECTED_RESULTS_FILE, "r") as f:
        EXPECTED_RESULTS = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: {EXPECTED_RESULTS_FILE} not found.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ Error parsing expected_results.json: {e}")
    sys.exit(1)

# Colors
GREEN = "32"
RED = "31"
YELLOW = "33"
CYAN = "36"


# ==============================
# Helper Functions
# ==============================
def color(text, code):
    """Apply ANSI color to text."""
    return f"\033[{code}m{text}\033[0m"


def parse_args():
    """Parse command line arguments."""
    args = sys.argv[1:]
    opts = {
        "tests": [],
        "parallel": True,
        "jobs": None,
        "filter": None,
        "verify_tensors": False,
    }

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--no-parallel":
            opts["parallel"] = False
        elif a == "--jobs" and i + 1 < len(args):
            opts["jobs"] = int(args[i + 1])
            i += 1
        elif a == "--filter" and i + 1 < len(args):
            opts["filter"] = args[i + 1]
            i += 1
        elif a == "--verify-tensors":
            opts["verify_tensors"] = True
        elif a.endswith(".tl"):
            opts["tests"].append(os.path.basename(a))
        i += 1

    return opts


def parse_result(log_content):
    """Extract results and operation types from log."""
    result_pattern = r"Result (\w+) \(([\w]+)\):\n\s*(.*?)\s*(?=\n[A-Z]|\nResult|\nFreed|$)"
    results = {}
    op_types = {}
    
    for match in re.finditer(result_pattern, log_content, re.DOTALL):
        variable, operation, result_str = match.groups()
        result_str = result_str.strip()

        if not result_str:
            continue

        op_types[variable] = operation

        # Try parsing as scalar
        scalar_match = re.match(r"^([\d\.\-\+e]+)\s*$", result_str, re.DOTALL)
        if scalar_match:
            try:
                results[variable] = float(scalar_match.group(1))
            except ValueError:
                pass
        else:
            # Parse as array
            cleaned_result = re.sub(r"\s+", ",", result_str.strip())
            cleaned_result = re.sub(r"\]\s*\[", "],[", cleaned_result)
            cleaned_result = cleaned_result.rstrip(",")
            cleaned_result = cleaned_result.replace("[,", "[").replace(",]", "]")
            try:
                results[variable] = eval(cleaned_result)
            except SyntaxError:
                pass
    
    return results, op_types


def load_computed_tensors(cache_dir, debug=False):
    """Load all .npy files from cache directory."""
    computed = {}
    cache_path = Path(cache_dir)
    
    if debug:
        print(f"[DEBUG] Looking for .npy files in: {cache_path}")
        print(f"[DEBUG] Cache dir exists: {cache_path.exists()}")
    
    if not cache_path.exists():
        if debug:
            print(f"[DEBUG] Cache directory not found: {cache_path}")
        return computed
    
    npy_files = list(cache_path.glob("*.npy"))
    if debug:
        print(f"[DEBUG] Found {len(npy_files)} .npy files: {[f.name for f in npy_files]}")
    
    for npy_file in npy_files:
        tensor_name = npy_file.stem
        try:
            computed[tensor_name] = np.load(npy_file)
            if debug:
                print(f"[DEBUG] Loaded {tensor_name}: shape {computed[tensor_name].shape}")
        except Exception as e:
            print(f"Warning: Failed to load {npy_file}: {e}")
    
    return computed


def _format_timing(duration, test_start_abs, suite_start_time):
    """Format timing string based on whether we're in parallel or sequential mode."""
    if suite_start_time is None:
        # Sequential mode: just show individual duration
        return f"({duration:.2f}s)"
    else:
        # Parallel mode: show wall-clock interval
        test_start_rel = test_start_abs - suite_start_time
        test_end_rel = test_start_rel + duration
        return f"[{test_start_rel:.1f}s → {test_end_rel:.1f}s]"


def run_test(test_file, verify_tensors=False, suite_start_time=None):
    """Run a single test and verify results.
    
    Returns tuple: (test_file, success, timing_str, failure_reason, verification_report)
    """
    original_dir = os.getcwd()
    os.chdir(ROOT_DIR)

    try:
        # Record time RIGHT before subprocess starts
        test_start_abs = time.time()
        
        cmd = ["python3", "tensorlang.py", str(f"tests/{test_file}"), "--debug"]
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            check=False, 
            text=True
        )
        
        # Record end time right after subprocess completes
        test_end_abs = time.time()

        # Match the cache directory structure from tensorlang.py
        cache_dir = f"cache/tests/{test_file}"
        test_stem = Path(test_file).stem
        log_file = f"{cache_dir}/{test_stem}.log"
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(log_file, "w") as f:
            f.write(process.stdout)

        if process.returncode != 0:
            duration = test_end_abs - test_start_abs
            timing_str = _format_timing(duration, test_start_abs, suite_start_time)
            return (test_file, False, timing_str, f"Compiler exited with code {process.returncode}", None)

        with open(log_file, "r") as f:
            log_content = f.read()

        results, op_types = parse_result(log_content)
        
        if not results:
            duration = test_end_abs - test_start_abs
            timing_str = _format_timing(duration, test_start_abs, suite_start_time)
            return (test_file, False, timing_str, "No results parsed from log", None)

        expected_results = EXPECTED_RESULTS.get(test_file, {})
        
        # Phase 1: Validate expected results
        for variable, expected in expected_results.items():
            if variable not in results:
                duration = test_end_abs - test_start_abs
                timing_str = _format_timing(duration, test_start_abs, suite_start_time)
                return (test_file, False, timing_str, f"Missing variable '{variable}' in results", None)

            result = results[variable]
            
            if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
                if not np.isclose(result, expected, rtol=1e-5, atol=1e-8):
                    duration = test_end_abs - test_start_abs
                    timing_str = _format_timing(duration, test_start_abs, suite_start_time)
                    return (test_file, False, timing_str, f"{variable}: expected {expected}, got {result}", None)
            else:
                try:
                    result_arr = np.asarray(result)
                    expected_arr = np.asarray(expected)

                    if result_arr.shape != expected_arr.shape:
                        duration = test_end_abs - test_start_abs
                        timing_str = _format_timing(duration, test_start_abs, suite_start_time)
                        return (test_file, False, timing_str, f"{variable}: shape mismatch {result_arr.shape} vs {expected_arr.shape}", None)
                    
                    if not np.allclose(result_arr, expected_arr, rtol=1e-5, atol=1e-8):
                        max_diff = np.abs(result_arr - expected_arr).max()
                        duration = test_end_abs - test_start_abs
                        timing_str = _format_timing(duration, test_start_abs, suite_start_time)
                        return (test_file, False, timing_str, f"{variable}: expected {expected_arr.tolist()}, got {result_arr.tolist()}", None)
                except Exception as e:
                    duration = test_end_abs - test_start_abs
                    timing_str = _format_timing(duration, test_start_abs, suite_start_time)
                    return (test_file, False, timing_str, f"{variable}: {str(e)}", None)

        duration = test_end_abs - test_start_abs
        
        # Phase 2: Verify tensor integrity (if enabled)
        verification_report = None
        if verify_tensors:
            verifier = TensorVerifier()
            computed_tensors = load_computed_tensors(cache_dir, debug=False)
            all_passed, verification_report = verifier.verify_all_tensors(
                computed_tensors,
                expected_results,
                cache_dir,
                op_types
            )
            
            if not all_passed:
                timing_str = _format_timing(duration, test_start_abs, suite_start_time)
                return (test_file, False, timing_str, "Tensor verification failed", verification_report)
        
        timing_str = _format_timing(duration, test_start_abs, suite_start_time)
        return (test_file, True, timing_str, None, verification_report)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        sys.exit(1)
    finally:
        os.chdir(original_dir)


def _extract_start_time(timing_str):
    """Extract start time from timing string for sorting."""
    if timing_str.startswith("["):
        try:
            start_str = timing_str.split(" → ")[0].replace("[", "").replace("s", "")
            return float(start_str)
        except:
            return 0
    return 0


def _strip_ansi(text):
    """Remove ANSI color codes for length calculation."""
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    return ansi_escape.sub('', text)


def print_results_table(results, verify_tensors, suite_start_time=None):
    """Print results in a formatted table, sorted by execution start time."""
    if not results:
        return
    
    # Sort by start time if in parallel mode
    if suite_start_time is not None:
        results = sorted(results, key=lambda x: _extract_start_time(x[2]))
    
    # Column widths
    state_width = 5
    name_width = max(len(t) for t, _, _, _, _ in results) + 2
    name_width = min(name_width, 60)
    time_width = 15
    tensor_width = 12
    
    # Header
    header = f"{'State':<{state_width}} {'TestCase':<{name_width}}"
    if verify_tensors:
        header += f" {'Tensors':<{tensor_width}}"
    header += f"{'Time':<{time_width}}"
    
    print(header)
    print("-" * 80)
    
    # Rows
    for test_file, success, message, failure_reason, v_report in results:
        # Truncate test name if too long
        test_display = test_file if len(test_file) <= name_width - 2 else test_file[:name_width - 5] + "..."
        
        # Status
        status_str = color("PASS", GREEN) if success else color("FAIL", RED)
        
        # Time
        time_str = message if message else "N/A"

        # Tensor info
        tensor_str = ""
        if verify_tensors:
            if v_report:
                s = v_report["summary"]
                if s["failed"] > 0:
                    tensor_str = color(f"✗ {s['failed']}/{s['total']}", RED)
                elif s["warnings"] > 0:
                    tensor_str = color(f"⚠ {s['warnings']} warn", YELLOW)
                else:
                    tensor_str = color(f"✓ {s['passed']}/{s['total']}", GREEN)
            else:
                tensor_str = "-"
        
        # Print row
        row = f"{status_str:<{state_width+10}} {test_display:<{name_width}}"
        if verify_tensors:
            # Calculate padding correctly by stripping ANSI codes
            visible_len = len(_strip_ansi(tensor_str))
            padding = max(0, 12 - visible_len)
            row += f" {tensor_str}{' ' * padding}"
        row += f" {time_str}"
        print(row)


def main():
    """Main test runner."""
    opts = parse_args()
    all_tests = list(EXPECTED_RESULTS.keys())

    if opts["filter"]:
        all_tests = [t for t in all_tests if opts["filter"] in t]

    if opts["tests"]:
        test_files = opts["tests"]
    else:
        test_files = all_tests

    if not test_files:
        print("No tests to run.")
        return
    
    verify_note = " (tensor verification enabled)" if opts["verify_tensors"] else ""
    start_suite = time.time()
    results = []
    verification_reports = {}
    
    if not opts["parallel"]:
        print(color(f"Running {len(test_files)} tests sequentially{verify_note}...\n", CYAN))
        for test_file in test_files:
            test_file, success, message, failure_reason, v_report = run_test(test_file, opts["verify_tensors"], suite_start_time=None)
            results.append((test_file, success, message, failure_reason, v_report))
            if v_report:
                verification_reports[test_file] = v_report
    else:
        jobs = opts["jobs"] or min(len(test_files), os.cpu_count() or 4)
        print(color(f"Running {len(test_files)} tests in parallel (jobs={jobs}){verify_note}...\n", CYAN))

        # Record suite start time for wall-clock timing
        suite_start_time_parallel = time.time()

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_test = {
                executor.submit(run_test, test, opts["verify_tensors"], suite_start_time=suite_start_time_parallel): test 
                for test in test_files
            }
            iterator = as_completed(future_to_test)
            
            if tqdm:
                pbar = tqdm(total=len(test_files), desc="Running tests", ncols=80, position=0, leave=True)

            for future in iterator:
                test_file, success, message, failure_reason, v_report = future.result()
                
                if tqdm:
                    pbar.update(1)
                
                results.append((test_file, success, message, failure_reason, v_report))
                if v_report:
                    verification_reports[test_file] = v_report
            
            if tqdm:
                pbar.close()

    duration_suite = time.time() - start_suite
    passed = sum(1 for _, ok, _, _, _ in results if ok)
    total = len(results)
    failed_tests = [(t, fr, vr) for t, ok, _, fr, vr in results if not ok]

    # Print results table
    print("\n" + "=" * 80)
    is_parallel = opts["parallel"]
    suite_start_for_sorting = start_suite if is_parallel else None
    print_results_table(results, opts["verify_tensors"], suite_start_for_sorting)
    print("=" * 80)
    
    print(color(f"Summary: {passed}/{total} tests passed in {duration_suite:.2f}s", CYAN))
    
    if passed == total:
        print(color("✅ All tests passed successfully!", GREEN))
    else:
        print(color(f"❌ {len(failed_tests)} test(s) failed.", RED))
    
    # Print detailed failure information
    if failed_tests:
        print("\n" + color("Failures:", RED))
        for test_name, failure_reason, v_report in failed_tests:
            print(f"\n  ❌ {test_name}")
            if failure_reason:
                print(f"     Reason: {failure_reason}")
            
            if v_report:
                # Print tensor verification failures
                for tensor_name, tensor_report in v_report["tensors"].items():
                    if not tensor_report["passed"]:
                        for check_name, check_result in tensor_report["checks"].items():
                            if check_result["passed"] is False:
                                print(f"     • {tensor_name} ({check_name}): {check_result['message']}")
            
            # Provide call-to-action for logs
            log_path = f"cache/tests/{test_name}/{Path(test_name).stem}.log"
            print(f"     📋 Check logs: {log_path}")
    
    if opts["verify_tensors"] and verification_reports:
        total_tensor_warnings = sum(
            r["summary"]["warnings"] for r in verification_reports.values()
        )
        total_tensor_failures = sum(
            r["summary"]["failed"] for r in verification_reports.values()
        )
        if total_tensor_warnings > 0 or total_tensor_failures > 0:
            issues = []
            if total_tensor_failures > 0:
                issues.append(f"{total_tensor_failures} tensor failures")
            if total_tensor_warnings > 0:
                issues.append(f"{total_tensor_warnings} tensor warnings")
            print(color(f"\n⚠️  Tensor verification: {', '.join(issues)}", YELLOW))
    
    print("=" * 80)


if __name__ == "__main__":
    main()