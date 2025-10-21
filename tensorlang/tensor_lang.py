# File: tensorlang/TensorLang.py
"""TensorLang utility and setup class"""

import sys
from pathlib import Path
import argparse

class TensorLang:

    version = "0.2.9"
    width   = 80

    def __init__(self):
        self.config = None
        self.version_description = f"TensorLang version: {self.version}"


    def print_header(self, version: str):
        border = "-" * (self.width - 2)
        print(f"+{border}+")
        print(f"+{border}+")
        print(f"{'|                          '}{self.version_description}                          |")
        print(f"+{border}+")
        print(f"+{border}+")


    def print(self, type: str = "", message: str = ""):
        if not type:
            type = "[DEBUG]"
        print(f"{type} {message}")


    def seperator(self):
        print("-" * self.width)


    def parse_arguments(self):
        """Parse and return command-line configuration"""

        tensorlangparser = argparse.ArgumentParser(description=self.version_description)
        # tensorlangparser.add_argument('file', nargs='?', default='tests/basics.tl', 
        tensorlangparser.add_argument('file', nargs='?', 
            help='TensorLang source file or test directory')
        tensorlangparser.add_argument('--debug', action='store_true', 
            help='Enable Debug mode (shown in output)')
        tensorlangparser.add_argument('--debug-info', action='store_true', 
            help='Enable Info (shown in output)')
        tensorlangparser.add_argument('--debug-ast', type=self._str_to_bool, default=False, 
            help='AST Abstract Syntax Tree (show in debug)')
        tensorlangparser.add_argument('--cache-layers', action='store_true', 
            help='Enable Cache Layers (shown in output)')

        # Test mode arguments
        tensorlangparser.add_argument('--test', action='store_true',
            help='Run in test mode (discover and run .tl tests)')
        tensorlangparser.add_argument('--no-parallel', action='store_true',
            help='Disable parallel test execution')
        tensorlangparser.add_argument('--jobs', type=int, default=None,
            help='Number of parallel jobs for tests')
        tensorlangparser.add_argument('--filter', type=str, default=None,
            help='Filter tests by name pattern')
        tensorlangparser.add_argument('--verify-tensors', action='store_true',
            help='Enable tensor verification (requires tensor_verifier.py)')

        args = tensorlangparser.parse_args()

        # # Validate file
        # self._validate_file(args.file)

        self.config = args
        return args
        
    
    @staticmethod
    def _str_to_bool(value):
        """Convert string to boolean"""
        if value.lower() in ('true', 't', '1', 'yes', 'y'):
            return True
        if value.lower() in ('false', 'f', '0', 'no', 'n'):
            return False
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")
    