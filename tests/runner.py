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

def main():

    # Example Commands
    # python3 tests/runner.py                          Run all tests in parallel
    # python3 tests/runner.py --no-parallel            Run sequentially
    # python3 tests/runner.py --jobs 6                 Run with 6 workers
    # python3 tests/runner.py tests/add.tl tests/mul.tl Run specific tests
    # python3 tests/runner.py --filter broadcast       Run tests matching filter
    # python3 tests/runner.py --verify-tensors         Enable full tensor verification

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
    
    # Show verification mode in compact form
    verify_note = " (tensor verification enabled)" if opts["verify_tensors"] else ""

    start_suite = time.time()
    results = []
    verification_reports = {}
    
    if not opts["parallel"]:
        print(color(f"Running {len(test_files)} tests sequentially{verify_note}...\n", CYAN))
        for test_file in test_files:
            test_file, success, message, v_report = run_test(test_file, opts["verify_tensors"])
            results.append((test_file, success, v_report))
            if v_report:
                verification_reports[test_file] = v_report
    else:
        jobs = opts["jobs"] or min(len(test_files), os.cpu_count() or 4)
        print(color(f"Running {len(test_files)} tests in parallel (jobs={jobs}){verify_note}...\n", CYAN))

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_test = {
                executor.submit(run_test, test, opts["verify_tensors"]): test 
                for test in test_files
            }
            iterator = as_completed(future_to_test)
            
            if tqdm:
                pbar = tqdm(total=len(test_files), desc="Running tests", ncols=80, position=0, leave=True)

            for future in iterator:
                test_file, success, message, v_report = future.result()
                
                if tqdm:
                    pbar.update(1)
                
                results.append((test_file, success, v_report))
                if v_report:
                    verification_reports[test_file] = v_report
            
            if tqdm:
                pbar.close()

    duration_suite = time.time() - start_suite
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failed_tests = [(t, vr) for t, ok, vr in results if not ok]

    # Print results table
    print("\n" + "=" * 80)
    print_results_table(results, opts["verify_tensors"])
    print("=" * 80)
    
    print(color(f"Summary: {passed}/{total} tests passed in {duration_suite:.2f}s", CYAN))
    
    if passed == total:
        print(color("✅ All tests passed successfully!", GREEN))
    else:
        print(color(f"❌ {len(failed_tests)} test(s) failed.", RED))
    
    if opts["verify_tensors"] and verification_reports:
        # Show count of tensor issues
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
            print(color(f"⚠️  Tensor verification: {', '.join(issues)}", YELLOW))
    
    print("=" * 80)


def print_results_table(results, verify_tensors):
    """Print results in a formatted table."""
    # Column widths
    name_width = max(len(t) for t, _, _ in results) + 2
    name_width = min(name_width, 50)  # Cap at 50
    status_width = 10
    time_width = 10
    tensor_width = 15 if verify_tensors else 0
    
    # Header
    header = f"{'Test':<{name_width}} {'Status':<{status_width}} {'Time':<{time_width}}"
    if verify_tensors:
        header += f" {'Tensors':<{tensor_width}}"
    print(header)
    #print("-" * len(header))
    print("-" * 80)
    
    # Rows
    for test_file, success, v_report in results:
        # Truncate test name if too long
        test_display = test_file if len(test_file) <= name_width - 2 else test_file[:name_width - 5] + "..."
        
        # Status
        status_str = color("PASS", GREEN) if success else color("FAIL", RED)
        
        # Time - try to extract from v_report if available
        time_str = "N/A"
        if v_report and "tensors" in v_report:
            # Rough estimate: use summary stats if available
            time_str = "✓"

        # Tensor info
        tensor_str = ""
        if v_report and verify_tensors:
            s = v_report["summary"]
            if s["failed"] > 0:
                tensor_str = color(f"✗ {s['failed']}/{s['total']}", RED)
            elif s["warnings"] > 0:
                tensor_str = color(f"⚠ {s['warnings']} warn", YELLOW)
            else:
                tensor_str = color(f"✓ {s['passed']}/{s['total']}", GREEN)
        
        # Print row
        row = f"{test_display:<{name_width}} {status_str:20} {time_str:<{time_width}}"
        if verify_tensors:
            row += f" {tensor_str:<{tensor_width}}"
        print(row)


def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def parse_args():
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

def run_test(test_file, verify_tensors=False):
    """Run a single test and verify results."""
    start_time = time.time()
    original_dir = os.getcwd()
    os.chdir(ROOT_DIR)

    try:
        cmd = ["python3", "tensorlang.py", str(f"tests/{test_file}"), "--debug"]
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            check=False, 
            text=True
        )

        # Match the cache directory structure from tensorlang.py
        cache_dir = f"cache/tests/{test_file}"
        test_stem = Path(test_file).stem
        log_file = f"{cache_dir}/{test_stem}.log"
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(log_file, "w") as f:
            f.write(process.stdout)

        if process.returncode != 0:
            duration = time.time() - start_time
            return (
                test_file, 
                False, 
                f"({duration:.2f}s)",
                None
            )

        with open(log_file, "r") as f:
            log_content = f.read()

        results, op_types = parse_result(log_content)
        
        if not results:
            return (
                test_file, 
                False, 
                "No results found in log.",
                None
            )

        expected_results = EXPECTED_RESULTS.get(test_file, {})
        
        # Phase 1: Validate expected results (original logic)
        for variable, expected in expected_results.items():
            if variable not in results:
                return (
                    test_file, 
                    False, 
                    f"Missing expected variable '{variable}'.",
                    None
                )

            result = results[variable]
            
            if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
                if not np.isclose(result, expected, rtol=1e-5, atol=1e-8):
                    return (
                        test_file, 
                        False, 
                        f"{variable}: expected {expected}, got {result}",
                        None
                    )
            else:
                try:
                    result_arr = np.asarray(result)
                    expected_arr = np.asarray(expected)

                    if result_arr.shape != expected_arr.shape:
                        return (
                            test_file, 
                            False, 
                            f"{variable}: shape mismatch - expected {expected_arr.shape}, got {result_arr.shape}",
                            None
                        )
                    
                    if not np.allclose(result_arr, expected_arr, rtol=1e-5, atol=1e-8):
                        return (
                            test_file, 
                            False, 
                            f"{variable}: values don't match",
                            None
                        )
                except Exception as e:
                    return (test_file, False, f"{variable}: comparison failed - {str(e)}", None)

        duration = time.time() - start_time
        
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
                return (test_file, False, "Tensor verification failed", verification_report)
        
        msg = f"({duration:.2f}s)"
        return (test_file, True, msg, verification_report)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        sys.exit(1)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
