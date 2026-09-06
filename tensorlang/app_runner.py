import os
import sys
import time
import tomli
from pathlib import Path
from typing import Optional, List, Dict, Any

from tensorlang.compiler import TensorCompiler
from tensorlang.test_runner import TestRunner


# https://claude.ai/chat/f02bfbe4-7e6e-42c8-affb-e6e6809acd75

class AppRunner:
    """Manages running tensorlang applications from apps/ directory"""

    def __init__(self, debug_mode=False, cache_layers=False, verify_tensors=False):
        self.debug_mode = debug_mode
        self.cache_layers = cache_layers
        self.verify_tensors = verify_tensors
        self.apps_dir = Path("apps")

        if not self.apps_dir.exists():
            raise FileNotFoundError(f"Apps directory not found: {self.apps_dir}")

    def list_apps(self, category: Optional[str] = None):
        """List all available apps, optionally filtered by category"""
        print("=" * 60)
        print("Available TensorLang Applications")
        print("=" * 60)

        apps = self._discover_apps()

        # Group by category
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for app in apps:
            cat = app['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(app)

        # Filter if category specified
        if category:
            by_category = {k: v for k, v in by_category.items() if k.startswith(category)}

        if not by_category:
            print(f"No apps found" + (f" in category: {category}" if category else ""))
            return

        # Display
        for cat in sorted(by_category.keys()):
            print(f"\n{cat}/")
            print("-" * 60)
            for app in by_category[cat]:
                print(f"  {app['name']:<25} - {app['description']}")
                if app['requirements']:
                    print(f"    Requirements: {app['requirements']}")

        print("\n" + "=" * 60)
        print(f"Total apps: {len(apps)}")
        print("\nUsage: python tensorlang.py --app <name>")
        #print("Example: python tensorlang.py --app web/desktop/dynamic")

    def _discover_apps(self) -> List[Dict[str, Any]]:
        """Discover all apps with app.toml files"""
        apps = []

        for root, dirs, files in os.walk(self.apps_dir):
            if "app.toml" in files:
                app_path = Path(root)
                rel_path = app_path.relative_to(self.apps_dir)

                try:
                    config = self._load_app_config(app_path / "app.toml")
                    apps.append({
                        'name': config['app']['name'],
                        'path': str(rel_path),
                        'category': str(rel_path.parent) if rel_path.parent != Path('.') else 'root',
                        'description': config['app'].get('description', 'No description'),
                        'requirements': self._format_requirements(config.get('requirements', {})),
                        'config': config
                    })
                except Exception as e:
                    if self.debug_mode:
                        print(f"Warning: Failed to load {app_path / 'app.toml'}: {e}")

        return sorted(apps, key=lambda x: (x['category'], x['name']))

    def _format_requirements(self, reqs: Dict[str, Any]) -> str:
        """Format requirements for display"""
        parts = []
        if 'gpus' in reqs:
            parts.append(f"{reqs['gpus']} GPU(s)")
        if 'cuda_version' in reqs:
            parts.append(f"CUDA {reqs['cuda_version']}")
        if 'memory_gb' in reqs:
            parts.append(f"{reqs['memory_gb']}GB RAM")
        return ", ".join(parts) if parts else ""

    def _load_app_config(self, config_path: Path) -> Dict[str, Any]:
        """Load and parse app.toml configuration"""
        with open(config_path, 'rb') as f:
            return tomli.load(f)

    def run_app(
        self,
        app_path: str,
        run_tests: bool = False,
        test_filter: Optional[str] = None,
        dev_mode: bool = False,
        benchmark: bool = False,
        app_args: Optional[List[str]] = None
    ):
        """Run a tensorlang application"""
        # Resolve app path
        full_path = self.apps_dir / app_path

        if not full_path.exists():
            print(f"Error: App not found: {app_path} full_path: {full_path}")
            print("Run 'python tensorlang.py --list-apps' to see available apps")
            sys.exit(1)

        # Load app configuration
        config_path = full_path / "app.toml"
        if not config_path.exists():
            print(f"Error: No app.toml found in {app_path}")
            sys.exit(1)

        config = self._load_app_config(config_path)

        # Validate requirements
        self._validate_requirements(config.get('requirements', {}))

        print("=" * 60)
        print(f"Running: {config['app']['name']}")
        print(f"Category: {app_path}")
        print(f"Description: {config['app'].get('description', 'N/A')}")
        print("=" * 60)

        # Handle different modes
        if run_tests:
            self._run_app_tests(full_path, config, test_filter)
        elif benchmark:
            self._run_benchmark(full_path, config)
        elif dev_mode:
            self._run_dev_mode(full_path, config, app_args)
        else:
            self._run_app_normal(full_path, config, app_args)

    def _validate_requirements(self, requirements: Dict[str, Any]):
        """Validate system meets app requirements"""
        # TODO: Implement actual validation
        # - Check CUDA version
        # - Check available GPUs
        # - Check memory
        # - Check external dependencies
        pass

    def _run_app_normal(self, app_path: Path, config: Dict[str, Any], app_args: Optional[List[str]]):
        """Run app in normal mode"""
        entry_points = config.get('entry_points', {})
        main_entry = entry_points.get('main', entry_points.get('train', None))

        if not main_entry:
            print("Error: No entry point defined in app.toml")
            print("Expected 'entry_points.main' or 'entry_points.train'")
            sys.exit(1)

        main_file = app_path / main_entry
        if not main_file.exists():
            print(f"Error: Entry point not found: {main_file}")
            sys.exit(1)

        print(f"\nExecuting: {main_file}")
        if app_args:
            print(f"Arguments: {' '.join(app_args)}")
        print()

        # Compile and execute
        compiler = TensorCompiler(
            debug_mode=self.debug_mode,
            cache_layers=self.cache_layers
        )

        start_time = time.time()
        ok = compiler.compile_and_execute(str(main_file))
        elapsed = time.time() - start_time

        if not ok:
            print(f"\n{'=' * 60}")
            print(f"Execution FAILED after {elapsed:.2f}s (compile/type-check error above)")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print(f"Execution completed in {elapsed:.2f}s")

    def _run_app_tests(self, app_path: Path, config: Dict[str, Any], test_filter: Optional[str]):
        """Run app test suite"""
        tests_dir = app_path / "tests"

        if not tests_dir.exists():
            print(f"No tests directory found in {app_path}")
            sys.exit(1)

        print(f"\nRunning tests from: {tests_dir}\n")

        # Use TestRunner but with app tests directory
        runner = TestRunner(
            parallel=False,  # App tests may have dependencies
            verify_tensors=self.verify_tensors,
            debug_mode=self.debug_mode,
            tests_dir=str(tests_dir)
        )

        test_files = runner.discover_tests()

        if test_filter:
            test_files = [t for t in test_files if test_filter in t]

        if not test_files:
            print(f"No tests found" + (f" matching: {test_filter}" if test_filter else ""))
            sys.exit(1)

        runner.run_test_suite(test_files)

    def _run_benchmark(self, app_path: Path, config: Dict[str, Any]):
        """Run app in benchmark mode"""
        entry_points = config.get('entry_points', {})
        bench_entry = entry_points.get('benchmark', entry_points.get('main', None))

        if not bench_entry:
            print("Error: No benchmark entry point defined")
            sys.exit(1)

        bench_file = app_path / bench_entry

        print(f"\n{'=' * 60}")
        print("BENCHMARK MODE")
        print(f"{'=' * 60}\n")

        # Run multiple iterations
        iterations = config.get('benchmark', {}).get('iterations', 10)
        warmup = config.get('benchmark', {}).get('warmup', 2)

        compiler = TensorCompiler(
            debug_mode=False,
            cache_layers=self.cache_layers
        )

        times = []

        print(f"Warmup iterations: {warmup}")
        for i in range(warmup):
            print(f"  Warmup {i+1}/{warmup}...", end='', flush=True)
            ok = compiler.compile_and_execute(str(bench_file))
            if not ok:
                print(" FAILED")
                print("Benchmark aborted: warmup run failed to compile/execute (see error above).")
                sys.exit(1)
            print(" done")

        print(f"\nBenchmark iterations: {iterations}")
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...\n", end='', flush=True)
            start = time.time()
            ok = compiler.compile_and_execute(str(bench_file))
            elapsed = time.time() - start
            if not ok:
                print(f" FAILED after {elapsed:.4f}s")
                print("Benchmark aborted: iteration failed to compile/execute (see error above).")
                sys.exit(1)
            times.append(elapsed)
            print(f" {elapsed:.4f}s")

        # Statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\n{'=' * 60}")
        print("BENCHMARK RESULTS")
        print(f"{'=' * 60}")
        print(f"Average: {avg_time:.4f}s")
        print(f"Min:     {min_time:.4f}s")
        print(f"Max:     {max_time:.4f}s")
        print(f"Range:   {max_time - min_time:.4f}s")

    def _run_dev_mode(self, app_path: Path, config: Dict[str, Any], app_args: Optional[List[str]]):
        """Run app in development mode with hot reload"""
        print(f"\n{'=' * 60}")
        print("DEVELOPMENT MODE - File watching enabled")
        print("Press Ctrl+C to stop")
        print(f"{'=' * 60}\n")

        # TODO: Implement file watching and hot reload
        # For now, just run normally
        print("Note: Hot reload not yet implemented, running normally...\n")
        self._run_app_normal(app_path, config, app_args)