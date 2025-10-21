import os
import sys

from tensorlang.tensor_lang import TensorLang
from tensorlang.compiler import TensorCompiler
from tensorlang.test_runner import TestRunner


def main():
    tensorlang = TensorLang()
    args = tensorlang.parse_arguments()

    try:
        if args.test:
            # Test Mode: Run test suite
            runner = TestRunner(
                parallel=not args.no_parallel,
                jobs=args.jobs,
                verify_tensors=args.verify_tensors,
                debug_mode=args.debug
            )

            # Determine which tests to run
            if args.file:
                if args.file.endswith('.tl'):
                    test_files = [os.path.basename(args.file)]
                else:
                    all_tests = runner.discover_tests()
                    test_files = [t for t in all_tests if args.file in t]
            else:
                test_files = runner.discover_tests()
                if args.filter:
                    test_files = [t for t in test_files if args.filter in t]
            
            if not test_files:
                print(f"No tests found matching: {args.file or args.filter}")
                sys.exit(1)
            
            runner.run_test_suite(test_files)
        
        else:
            # Normal Mode: Compile and execute
            compiler = TensorCompiler(
                debug_mode=args.debug,
                debug_info=args.debug_info,
                debug_ast=args.debug_ast,
                cache_layers=args.cache_layers
            )
            compiler.compile_and_execute(args.file)

    except FileNotFoundError as e:
        print(f"Error: File not found - {args.file}")
        if args.debug:
            print(f"Details: {e}")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()