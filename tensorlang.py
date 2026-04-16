import os
import sys

from tensorlang.tensor_lang import TensorLang
from tensorlang.compiler import TensorCompiler
from tensorlang.test_runner import TestRunner
from tensorlang.app_runner import AppRunner

def main():
    tensorlang = TensorLang()
    args = tensorlang.parse_arguments()

    try:
        # App Mode: Run applications from apps/ directory
        if getattr(args, 'app', None) or getattr(args, 'list_apps', False):
            runner = AppRunner(
                debug_mode=args.debug,
                cache_layers=args.cache_layers,
                verify_tensors=args.verify_tensors
            )
            
            if getattr(args, 'list_apps', False):
                runner.list_apps(category=getattr(args, 'category', None))
                sys.exit(0)
            
            # Run the specified app
            runner.run_app(
                app_path=args.app,
                run_tests=args.test,
                test_filter=getattr(args, 'filter', None),
                dev_mode=getattr(args, 'dev', False),
                benchmark=getattr(args, 'benchmark', False),
                app_args=getattr(args, 'app_args', None)
            )
        
        elif args.test:
            # Test Mode: Run core test suite
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
            # Normal Mode: Compile and execute a single .tl file
            if not args.file:
                print("Error: No file specified. Use --file <path> or --app <name>")
                print("Run 'python tensorlang.py --help' for usage information")
                sys.exit(1)
                
            compiler = TensorCompiler(
                debug_mode=args.debug,
                debug_info=args.debug_info,
                debug_ast=args.debug_ast,
                cache_layers=args.cache_layers
            )
            compiler.compile_and_execute(args.file)

    except FileNotFoundError as e:
        file_ref = getattr(args, 'file', None) or getattr(args, 'app', 'unknown')
        print(f"Error: File not found - {file_ref}")
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