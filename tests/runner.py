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
    tqdm = None  # optional dependency

# Example Commands
# Command	Description
# python3 tests/runner.py	Run all tests in parallel (auto jobs = CPU cores)
# python3 tests/runner.py --no-parallel	Run sequentially
# python3 tests/runner.py --jobs 6	Run 6 parallel workers
# python3 tests/runner.py tests/add.tl tests/mul.tl	Run specific tests
# python3 tests/runner.py --filter broadcast	Run only tests whose filenames contain "broadcast"

# ================================================================
# Paths and Globals
# ================================================================
ROOT_DIR = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent
EXPECTED_RESULTS_FILE = TEST_DIR / "expected_results.json"

# ================================================================
# Load Expected Results
# ================================================================
try:
    with open(EXPECTED_RESULTS_FILE, "r") as f:
        EXPECTED_RESULTS = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: {EXPECTED_RESULTS_FILE} not found. Please create it with expected results.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ Error parsing expected_results.json: {e}")
    sys.exit(1)

# ================================================================
# Helpers
# ================================================================
def color(text, code):
    return f"\033[{code}m{text}\033[0m"

GREEN = "32"
RED = "31"
YELLOW = "33"
CYAN = "36"

def parse_args():
    args = sys.argv[1:]
    opts = {
        "tests": [],
        "parallel": True,
        "jobs": None,
        "filter": None,
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
        elif a.endswith(".tl"):
            opts["tests"].append(os.path.basename(a))
        i += 1

    return opts

# ================================================================
# Parse Log Results
# ================================================================
def parse_result(log_content):
    """Extract and clean all results from the log using regex."""
    result_pattern = r"Result (\w+) \(([\w]+)\):\n\s*(.*?)\s*(?=\n[A-Z]|\nResult|\nFreed|$)"
    results = {}
    for match in re.finditer(result_pattern, log_content, re.DOTALL):
        variable, operation, result_str = match.groups()
        result_str = result_str.strip()

        if not result_str:
            print(f"Warning: Empty result for variable '{variable}' in log.")
            continue

        scalar_match = re.match(r"^([\d\.\-\+e]+)\s*$", result_str, re.DOTALL)
        if scalar_match:
            try:
                results[variable] = float(scalar_match.group(1))
            except ValueError as e:
                print(f"Error parsing scalar result for {variable}: {e}")
        else:
            #print(f"Debug: Raw result for {variable}: {result_str!r}")
            cleaned_result = re.sub(r"\s+", ",", result_str.strip())
            cleaned_result = re.sub(r"\]\s*\[", "],[", cleaned_result)
            cleaned_result = cleaned_result.rstrip(",")
            cleaned_result = cleaned_result.replace("[,", "[").replace(",]", "]")
            # print(f"Debug: Cleaned result for {variable}: {cleaned_result!r}")
            try:
                results[variable] = eval(cleaned_result)
            except SyntaxError as e:
                print(f"Error parsing array result for {variable}: {e}, cleaned_result={cleaned_result!r}")
    return results

# ================================================================
# Run a Single Test
# ================================================================
def run_test(test_file):
    """Run a single test and compare with expected results."""
    start_time = time.time()
    original_dir = os.getcwd()
    os.chdir(ROOT_DIR)
    test_passed = False

    try:
        cmd = ["python3", "tensorlang.py", str(f"tests/{test_file}"), "--debug"]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)

        log_file = f"cache/tests/{test_file}/{Path(test_file).stem}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            f.write(process.stdout)

        if process.returncode != 0:
            duration = time.time() - start_time
            return (test_file, False, f"Exited with code {process.returncode} ({duration:.2f}s)")

        with open(log_file, "r") as f:
            log_content = f.read()

        results = parse_result(log_content)
        if not results:
            return (test_file, False, "No results found in log or parsing failed.")

        expected_results = EXPECTED_RESULTS.get(test_file, {})
        for variable, expected in expected_results.items():
            if variable not in results:
                return (test_file, False, f"Missing expected variable '{variable}'.")

            result = results[variable]
            
            # Handle scalar comparisons
            if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
                if not np.isclose(result, expected, rtol=1e-5, atol=1e-8):
                    return (test_file, False, f"{variable}: expected {expected}, got {result}")
            else:
                # Convert to numpy arrays for comparison
                try:
                    result_arr = np.asarray(result)
                    expected_arr = np.asarray(expected)
                    
                    # print(f"Test: {test_file}, Variable: {variable}")
                    # print(f"Result shape: {result_arr.shape}, values: {result_arr}")
                    # print(f"Expected shape: {expected_arr.shape}, values: {expected_arr}")

                    # Check if shapes match
                    if result_arr.shape != expected_arr.shape:
                        return (test_file, False, 
                                f"{variable}: shape mismatch - expected {expected_arr.shape}, got {result_arr.shape}")
                    
                    # Compare values
                    if not np.allclose(result_arr, expected_arr, rtol=1e-5, atol=1e-8):
                        return (test_file, False, 
                                f"{variable}: expected {expected}, got {result}")
                except Exception as e:
                    return (test_file, False, 
                            f"{variable}: comparison failed - {str(e)}")

        duration = time.time() - start_time
        return (test_file, True, f"All checks passed ({duration:.2f}s)")

    except KeyboardInterrupt as e:
        print(f"Keyboard Interrupt")
        sys.exit(1)
    finally:
        os.chdir(original_dir)

# ================================================================
# Main Runner
# ================================================================
def main():
    opts = parse_args()
    all_tests = list(EXPECTED_RESULTS.keys())

    # Filter tests
    if opts["filter"]:
        all_tests = [t for t in all_tests if opts["filter"] in t]

    # If specific tests are given, use them
    if opts["tests"]:
        test_files = opts["tests"]
    else:
        test_files = all_tests

    if not test_files:
        print("No tests to run.")
        return

    start_suite = time.time()
    results = []

    # Decide on parallel or sequential
    if not opts["parallel"]:
        print(color(f"Running {len(test_files)} tests sequentially...\n", CYAN))
        for test_file in test_files:
            test_file, success, message = run_test(test_file)
            output = color(f"PASS  {test_file}: {message}", GREEN) if success else color(f"FAIL  {test_file}: {message}", RED)
            print(output)
            results.append((test_file, success))
    else:
        jobs = opts["jobs"] or min(len(test_files), os.cpu_count() or 4)
        print(color(f"Running {len(test_files)} tests in parallel (jobs={jobs})...\n", CYAN))

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_test = {executor.submit(run_test, test): test for test in test_files}
            iterator = as_completed(future_to_test)
            if tqdm:
                iterator = tqdm(iterator, total=len(test_files), desc="Running tests", ncols=100)

            for future in iterator:
                test_file, success, message = future.result()
                output = color(f"PASS  {test_file}: {message}", GREEN) if success else color(f"FAIL  {test_file}: {message}", RED)
                if tqdm:
                    tqdm.write(output)
                else:
                    print(output)
                results.append((test_file, success))

    duration_suite = time.time() - start_suite
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    failed_tests = [t for t, ok in results if not ok]

    print("\n" + "-" * 80)
    print(color(f"Summary: {passed}/{total} tests passed in {duration_suite:.2f}s", CYAN))
    if passed == total:
        print(color("✅ All tests passed successfully!", GREEN))
    else:
        print(color("❌ Some tests failed. Check logs for details.", RED))
    print("-" * 80)

    # Re-run failed tests sequentially for debugging
    if failed_tests:
        print(color("\nRe-running failed tests sequentially for detailed logs...\n", YELLOW))
        for test in failed_tests:
            test_file, success, message = run_test(test)
            output = color(f"PASS  {test_file} (after re-run): {message}", GREEN) if success else color(f"FAIL  {test_file} (after re-run): {message}", RED)
            print(output)

# ================================================================
if __name__ == "__main__":
    main()
