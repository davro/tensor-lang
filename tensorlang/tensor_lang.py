"""TensorLang utility and setup class.

This module defines the `TensorLang` class, which handles:
- Command-line argument parsing
- Console output formatting (headers, separators, debug messages)
- Version management

It serves as the entry point for running TensorLang scripts or tests.
"""

import sys
from pathlib import Path
import argparse

class TensorLang:
    """Main utility and setup class for the TensorLang runtime."""
    
    version = "0.3.0"
    width   = 80

    def __init__(self):
        """Initialize default configuration and version text."""
        self.config = None
        self.version_description = f"TensorLang version: {self.version}"

    # ---------------------------------------------------------------------
    # Output Utilities
    # ---------------------------------------------------------------------

    def print_header(self, version: str | None = None):
        """Print a formatted program header with version information."""
        border = "=" * (self.width - 3)
        print(f"// {border}")
        print(f"// {border}")
        version_text = version or self.version_description
        centered_text = version_text.center(self.width - 5)
        print(f"// {centered_text}//")
        print(f"// {border}")
        print(f"// {border}")

    def print(self, type: str = "[DEBUG]", message: str = ""):
        """Print a message with a labeled prefix (default: [DEBUG])."""
        print(f"{type} {message}")

    def separator(self):
        """Print a horizontal separator line."""
        print("#" * self.width)

    # ---------------------------------------------------------------------
    # Command-line Argument Parsing
    # ---------------------------------------------------------------------

    def parse_arguments(self):
        """Parse and return command-line configuration for TensorLang."""
        parser = argparse.ArgumentParser(
            description=self.version_description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run a .tl file
  python tensorlang.py script.tl
  python tensorlang.py --file script.tl
  
  # Run core tests
  python tensorlang.py --test
  python tensorlang.py --test --filter matrix
  
  # List available apps
  python tensorlang.py --list-apps
  python tensorlang.py --list-apps --category ml
  
  # Run an app
  python tensorlang.py --app web/dynamic/bitcoin
  python tensorlang.py --app ml/training/mnist --benchmark
  
  # Run app tests
  python tensorlang.py --app web/dynamic/bitcoin --test
            """
        )

        # Core arguments - file is now optional since we have --app and --list-apps
        parser.add_argument(
            "file",
            nargs="?",
            help="TensorLang source file or test directory"
        )
        
        # Main execution modes
        parser.add_argument(
            "--test",
            action="store_true",
            help="Run in test mode (discover and run .tl tests)"
        )
        parser.add_argument(
            "--app",
            type=str,
            help="Run an application from apps/ directory (e.g., web/dynamic/bitcoin)"
        )
        parser.add_argument(
            "--list-apps",
            action="store_true",
            help="List all available applications"
        )
        
        # Debug and compilation options
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable Debug mode (shown in output)"
        )
        parser.add_argument(
            "--debug-info",
            action="store_true",
            help="Enable Info (shown in output)"
        )
        parser.add_argument(
            "--debug-ast",
            type=self._str_to_bool,
            default=False,
            help="Display the Abstract Syntax Tree (AST) in debug mode"
        )
        parser.add_argument(
            "--cache-layers",
            action="store_true",
            help="Enable cache layers (shown in output)"
        )
        parser.add_argument(
            "--transpile",
            action="store_true",
            help="Transpile CUDA to WGSL (BETA)"
        )
        parser.add_argument(
            "--verify-tensors",
            action="store_true",
            help="Enable tensor verification (requires tensor_verifier.py)"
        )

        # Test mode arguments
        parser.add_argument(
            "--no-parallel",
            action="store_true",
            help="Disable parallel test execution"
        )
        parser.add_argument(
            "--jobs",
            type=int,
            default=None,
            help="Number of parallel jobs for tests"
        )
        parser.add_argument(
            "--filter",
            type=str,
            default=None,
            help="Filter tests by name pattern"
        )
        
        # App-specific options
        parser.add_argument(
            "--category",
            type=str,
            help="Filter apps by category when listing (e.g., ml, web, cuda)"
        )
        parser.add_argument(
            "--dev",
            action="store_true",
            help="Run app in development mode with hot reload"
        )
        parser.add_argument(
            "--benchmark",
            action="store_true",
            help="Run app in benchmark mode"
        )
        parser.add_argument(
            "--app-args",
            nargs="*",
            help="Arguments to pass to the application"
        )

        args = parser.parse_args()
        
        # Validation: some options only make sense with specific modes
        if args.category and not args.list_apps:
            parser.error("--category requires --list-apps")
        
        if (args.dev or args.benchmark) and not args.app:
            parser.error("--dev and --benchmark require --app")
        
        if args.app_args and not args.app:
            parser.error("--app-args requires --app")
        
        self.config = args
        return args
    
    # ---------------------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _str_to_bool(value: str) -> bool:
        """Convert a string input to boolean (used for CLI args)."""
        truthy = {"true", "t", "1", "yes", "y"}
        falsy = {"false", "f", "0", "no", "n"}

        value_lower = value.lower()
        if value_lower in truthy:
            return True
        if value_lower in falsy:
            return False
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")