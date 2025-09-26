import os
import json
import subprocess
import re
from pathlib import Path
import numpy as np

# Define paths
ROOT_DIR = Path(__file__).parent.parent  # Move up one level from tests/ to root
TEST_DIR = Path(__file__).parent  # Current directory (tests/)
EXPECTED_RESULTS_FILE = TEST_DIR / "expected_results.json"

# Load expected results
try:
    with open(EXPECTED_RESULTS_FILE, 'r') as f:
        EXPECTED_RESULTS = json.load(f)
except FileNotFoundError:
    print(f"Error: {EXPECTED_RESULTS_FILE} not found. Please create it with expected results.")
    exit(1)

def parse_result(log_content):
    """Extract and clean all results from the log using regex."""
    # More flexible pattern to capture everything after "Result variable (operation):"
    result_pattern = r"Result (\w+) \(([\w]+)\):\n(.*?)(?=\n[A-Z]|\nResult|\nTensorLang|\nFreed|$)"
    results = {}
    for match in re.finditer(result_pattern, log_content, re.DOTALL):
        variable, operation, result_str = match.groups()
        ##print(f"{variable} ( RAW ) : {repr(result_str)}")  # Debug output
        result_str = result_str.strip()
        
        # Check if it's a scalar (just a number, possibly with newlines)
        scalar_match = re.match(r'^([\d\.\-\+e]+)\s*$', result_str, re.DOTALL)
        if scalar_match:
            try:
                result = float(scalar_match.group(1))
                results[variable] = result
                ##print(f"{variable} (SCALAR): {result}")  # Debug output
            except ValueError as e:
                print(f"Error parsing scalar result for {variable}: {e}")
        else:
            # Handle array results (existing logic)
            # Clean the result string: replace spaces with commas within brackets
            cleaned_result = re.sub(r'\s+', ',', result_str.strip())

            # Ensure commas between sublists and valid syntax
            cleaned_result = re.sub(r'\]\s*\[', '],[', cleaned_result)

            # Remove any trailing commas or brackets
            cleaned_result = cleaned_result.rstrip(',')
            cleaned_result = cleaned_result.replace('[,', '[').replace(',]', ']')
            ##print(f"{variable} (ARRAY) : {cleaned_result}")  # Debug output
            
            try:
                result = eval(cleaned_result)  # Safely evaluate the cleaned string
                results[variable] = result
            except SyntaxError as e:
                print(f"Error parsing array result for {variable}: {e}")
        
        ##print("")  # Empty line for readability
        
    return results

def run_test(test_file):
    """Run a single test and compare with expected result."""
    log_file = f"cache/tests/{test_file}/{Path(test_file).stem}.log"

    # Change to root directory to ensure tensorlang.lark is found
    original_dir = os.getcwd()
    os.chdir(ROOT_DIR)
    test_passed = False
    try:
        cmd = ["python3", "tensorlang.py", str(f"tests/{test_file}")]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        
        with open(log_file, 'w') as f:
            f.write(process.stdout)
        print(f"+------------------------------------------------------------------------------+")
        #print(f"|------------------------------------------------------------------------------+")
        print(f"| TestCase: {test_file}")
        #print(f"|------------------------------------------------------------------------------+")
        print(f"+------------------------------------------------------------------------------+")

        if process.returncode != 0:
            print(f"FAIL for {test_file}: parser.py exited with status {process.returncode}")
            print(f"Output: {process.stdout}")
            return False
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        results = parse_result(log_content)
        if not results:
            print(f"FAIL for {test_file}: No results found in log or parsing failed.")
            return False
        
        expected_results = EXPECTED_RESULTS.get(test_file, {})
        for variable, expected in expected_results.items():
            if variable not in results:
                print(f"FAIL for {test_file}: No result found for expected variable {variable}.")
                return False
            result = results[variable]
            
            # Handle comparison for both scalars and arrays
            if isinstance(expected, (int, float)) and isinstance(result, (int, float)):
                # Scalar comparison
                if not np.isclose(result, expected, rtol=1e-5, atol=1e-8):
                    print(f"FAIL for {test_file}: Expected {expected} for {variable}, got {result}")
                    return False
            else:
                # Array comparison
                if not np.allclose(result, expected, rtol=1e-5, atol=1e-8):
                    print(f"FAIL for {test_file}: Expected {expected} for {variable}, got {result}")
                    return False

        for variable, result in results.items():
            if variable in expected_results:
                print(f"PASS for {test_file}: {variable} = {result}")
                test_passed = True
        if not test_passed:
            print(f"FAIL for {test_file}: No expected variables matched the results.\n")
            return False

        print("")  # empty line between test cases

        return True
    finally:
        # Restore original directory
        os.chdir(original_dir)

def main():
    passed = 0
    total = 0
    
    for test_file in EXPECTED_RESULTS.keys():
        total += 1
        if run_test(test_file):
            passed += 1
    
    print(f"--------------------------------------------------------------------------------")
    print(f"Summary: {passed}/{total} tests passed.")
    if passed == total:
        print("All tests passed successfully!")
    else:
        print("Tests failed. Check the tests logs for details (available for failed tests).")
    print(f"--------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()