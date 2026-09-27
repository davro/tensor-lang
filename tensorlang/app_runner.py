import os
import sys
import time
import subprocess
import tomli
from pathlib import Path
from typing import Optional, List, Dict, Any

from tensorlang.compiler import TensorCompiler
from tensorlang.test_runner import TestRunner
from tensorlang.gpu import choose_device, pin_current_process, detect_gpus


# https://claude.ai/chat/f02bfbe4-7e6e-42c8-affb-e6e6809acd75

# Set in this process's environment (and therefore inherited by every
# subprocess it spawns) for the duration of a [lifecycle] pipeline run.
# See _run_lifecycle for why this exists: it's a tripwire against a
# nested `tensorlang.py --app <name>` call (with no --step) re-triggering
# the whole pipeline from inside one of its own steps — unbounded
# recursive subprocess spawning, each opening its own CUDA context, which
# can make a machine unresponsive well before anything errors out on its
# own. A tool script that needs to invoke itself as a subprocess should
# always pass an explicit --step.
_LIFECYCLE_ACTIVE_ENV = "TENSORLANG_LIFECYCLE_ACTIVE"


class AppRunner:
    """Manages running tensorlang applications from apps/ directory"""

    def __init__(self, debug_mode=False, cache_layers=False, verify_tensors=False, fixed_gpu=None):
        self.debug_mode = debug_mode
        self.cache_layers = cache_layers
        self.verify_tensors = verify_tensors
        # Set by tensorlang.py main() from --gpu N, which (if present) has
        # already called pin_current_process() before this AppRunner was
        # even constructed. Kept here only so _validate_requirements()
        # knows an explicit choice was already made and shouldn't be
        # second-guessed by choose_device()'s "least utilized" heuristic.
        self.fixed_gpu = fixed_gpu
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
        step: Optional[str] = None,
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

        # Validate requirements / pin a GPU if the app asks for one
        self._validate_requirements(config)

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
        elif step:
            self._run_entry_point(full_path, config, step, app_args)
        else:
            self._run_app_normal(full_path, config, app_args)

    def _validate_requirements(self, config: Dict[str, Any]):
        """Validate/act on system requirements declared in app.toml.

        Runs two independent checks, unconditionally, regardless of which
        mode `run_app` is about to dispatch to:
          - `requirements.packages` — pip packages the app needs (e.g.
            pygame for a game's viewer), auto-installed if missing. This
            is the declarative replacement for a run.sh's
            `python3 -c "import pygame" || pip install pygame` preamble.
          - `requirements.gpus` (Tier-A process-level pinning via
            tensorlang/gpu.py — see handover §18) and an optional
            `[gpu] device = N` override.
        cuda_version / memory_gb are still only surfaced by `list_apps`,
        not enforced here.

        This must finish before the app's entry point reaches
        `compiler.compile_and_execute()`, since that's where
        `pycuda.autoinit` gets imported and the process's CUDA context —
        and therefore its CUDA_VISIBLE_DEVICES binding — becomes fixed
        for the rest of the process's lifetime.
        """
        requirements = config.get('requirements', {})
        self._ensure_packages(requirements.get('packages', []))
        self._validate_gpu_requirements(config, requirements)

    def _ensure_packages(self, packages: List[str]):
        """Import-check each named package, `pip install`-ing any that are
        missing. Mirrors run.sh's `python3 -c "import X" || pip install X`
        pattern, including its fail-loud behavior on a failed install —
        those run.sh scripts all have `set -euo pipefail`, so a failed
        `pip install` aborted the whole script; a failed auto-install here
        exits the same way, since the app is about to fail anyway once it
        hits the missing import.
        """
        import importlib

        for pkg in packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                print(f"== installing missing dependency: {pkg} ==")
                result = subprocess.run([sys.executable, "-m", "pip", "install", pkg])
                if result.returncode != 0:
                    print(f"Error: failed to install required package '{pkg}' "
                          f"(exit {result.returncode}).")
                    sys.exit(result.returncode)

    def _validate_gpu_requirements(self, config: Dict[str, Any], requirements: Dict[str, Any]):
        """The `requirements.gpus` / `[gpu] device` half of
        `_validate_requirements` — see that method's docstring."""
        requested_gpus = requirements.get('gpus')

        if not requested_gpus:
            return

        # --gpu N on the CLI already pinned this whole process in
        # tensorlang.py's main(), before this AppRunner was constructed.
        # Respect that explicit choice rather than overriding it below.
        if self.fixed_gpu is not None:
            print(f"GPU: using explicitly pinned device {self.fixed_gpu} "
                  f"(--gpu overrides app.toml's requirements.gpus = {requested_gpus})")
            return

        # An app whose lifecycle shells out to `tensorlang.py --app ...`
        # itself (e.g. decision_boundary's chunked training, see
        # tlkit/chunked_runner.py) inherits this process's environment in
        # its subprocess, CUDA_VISIBLE_DEVICES included. If we let that
        # nested invocation call choose_device() again, nvidia-smi still
        # reports every physical GPU (it ignores CUDA_VISIBLE_DEVICES), so
        # the child could "re-pick" a different device than the parent
        # already committed to — bouncing a single chunked training run
        # across GPUs between chunks instead of staying put. If something
        # upstream already pinned this process, trust it and stop here.
        existing_pin = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing_pin:
            print(f"GPU: CUDA_VISIBLE_DEVICES already set to {existing_pin!r} "
                  f"(inherited from parent process); leaving as-is.")
            return

        # Optional app.toml override: [gpu] device = N lets an app pin to
        # a specific card instead of choose_device()'s "least utilized"
        # pick — e.g. an app that wants to stay off whichever GPU is also
        # driving the desktop, or that always wants device 1.
        gpu_cfg = config.get('gpu', {})
        explicit_device = gpu_cfg.get('device')

        if explicit_device is not None:
            available = detect_gpus()
            if available and explicit_device not in {g['index'] for g in available}:
                print(f"Warning: app.toml [gpu] device = {explicit_device} not among "
                      f"detected GPUs ({sorted(g['index'] for g in available)}); "
                      f"proceeding anyway.")
            pin_current_process(explicit_device)
            print(f"GPU: pinned to device {explicit_device} (app.toml [gpu].device)")
            return

        chosen, warning = choose_device(requested_gpus=requested_gpus)

        if not chosen:
            # No GPUs detected at all - nothing to pin, run unpinned.
            print(f"Warning: {warning}")
            return

        if warning:
            # e.g. requested_gpus > 1, degraded to a single device (Tier-A).
            print(f"Warning: {warning}")

        pin_current_process(chosen[0])
        print(f"GPU: pinned to device {chosen[0]} (least-utilized of "
              f"{len(detect_gpus())} detected; app requests {requested_gpus} GPU(s))")

    def _run_app_normal(self, app_path: Path, config: Dict[str, Any], app_args: Optional[List[str]]):
        """Run app in normal mode (bare `--app <name>`, no flags).

        If the app declares a `[lifecycle]` table, that's now the source
        of truth for what "just run it" means — see `_run_lifecycle`.
        Apps without one (hello_mlp, linear_regression, tic_tac_toe as of
        this writing) keep the original single-`.tl`-file behavior
        unchanged.
        """
        lifecycle = config.get('lifecycle')
        if lifecycle and lifecycle.get('default'):
            self._run_lifecycle(app_path, config, lifecycle)
            return

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

        self._execute_tl_file(main_file, app_args)

    def _execute_tl_file(self, tl_path: Path, app_args: Optional[List[str]]) -> bool:
        """Compile and execute a single `.tl` file. Shared by the plain
        `main` entry, `--step <name>` for a `.tl` entry, and each step of
        a `[lifecycle]` pipeline."""
        print(f"\nExecuting: {tl_path}")
        if app_args:
            print(f"Arguments: {' '.join(app_args)}")
        print()

        compiler = TensorCompiler(
            debug_mode=self.debug_mode,
            cache_layers=self.cache_layers
        )

        start_time = time.time()
        ok = compiler.compile_and_execute(str(tl_path))
        elapsed = time.time() - start_time

        if not ok:
            print(f"\n{'=' * 60}")
            print(f"Execution FAILED after {elapsed:.2f}s (compile/type-check error above)")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print(f"Execution completed in {elapsed:.2f}s")
        return ok

    def _run_entry_point(
        self,
        app_path: Path,
        config: Dict[str, Any],
        entry_name: str,
        app_args: Optional[List[str]] = None
    ):
        """Dispatch a single named `[entry_points]` entry by file type.

        This is what `--step <name>` calls directly, and what
        `_run_lifecycle` calls once per pipeline step. A `.tl` entry runs
        through the compiler exactly like `main` always has. Anything
        else — the `.py` tool scripts that run.sh files were shelling out
        to (reset/train/play/generate_data/promote/rollback, depending on
        the app) — runs as a subprocess of this (already GPU-pinned, see
        `_validate_requirements`) process, so a nested `tensorlang.py`
        invocation a tool script makes itself inherits the same pin.
        """
        entry_points = config.get('entry_points', {})
        target = entry_points.get(entry_name)

        if not target:
            print(f"Error: app.toml has no entry_points.{entry_name} defined "
                  f"for this app")
            sys.exit(1)

        target_path = app_path / target
        if not target_path.exists():
            print(f"Error: entry point not found: {target_path}")
            sys.exit(1)

        if target_path.suffix == '.tl':
            return self._execute_tl_file(target_path, app_args)

        print(f"\n== {entry_name}: {target_path} ==")
        result = subprocess.run([sys.executable, str(target_path)] + (app_args or []))
        if result.returncode != 0:
            print(f"\nStep '{entry_name}' failed (exit {result.returncode}).")
            sys.exit(result.returncode)

    def _lifecycle_state_exists(self, lifecycle: Dict[str, Any]) -> bool:
        """Whether `[lifecycle] state_marker` indicates the app has
        already been initialized — mirrors run.sh patterns like
        `[[ -d "$WEIGHTS_DIR" ]]` / `[[ -n "$(ls -A "$WEIGHTS_DIR")" ]]`.
        `state_marker` is resolved relative to the repo root (matching
        cache paths like `cache/apps/<app>/main.tl/weights`, which live
        outside the app's own directory), not relative to the app dir.
        """
        marker = lifecycle.get('state_marker')
        if not marker:
            return False
        marker_path = Path(marker)
        if not marker_path.exists():
            return False
        if marker_path.is_dir():
            return any(marker_path.iterdir())
        return True

    def _run_lifecycle(self, app_path: Path, config: Dict[str, Any], lifecycle: Dict[str, Any]):
        """Run app.toml's `[lifecycle] default` pipeline — the declarative
        replacement for a run.sh's "check state, maybe init, train, play"
        sequence for apps whose default action is more than one `.tl`
        file. Each step is an `entry_points` name; suffix a step with
        `_if_missing` (e.g. `reset_if_missing`) to only run it when
        `state_marker` isn't yet populated, otherwise it always runs.

        Guarded against a very specific, very bad failure mode: a step's
        tool script (e.g. a chunked trainer) shelling back into
        `tensorlang.py --app <this app>` with no `--step` would re-enter
        this exact method from inside itself — and since that nested call
        runs the *whole* pipeline again, including this same step, it
        recurses without bound, forking a new CUDA-context-holding
        process at every level. That's not a bug that fails loudly; it's
        one that can make the machine unresponsive first. So the first
        thing this method does is check (and set) an env var that every
        subprocess it spawns inherits, and refuses outright if it's
        already set.
        """
        if os.environ.get(_LIFECYCLE_ACTIVE_ENV):
            print(
                "Error: a [lifecycle] step tried to invoke `tensorlang.py --app "
                f"{app_path.name}` again from inside its own pipeline (no --step "
                "given). That would re-run the whole pipeline recursively — likely "
                "a tool script that needs to pass an explicit --step (e.g. "
                "'--step main') instead of a bare --app call. Refusing to recurse."
            )
            sys.exit(1)

        os.environ[_LIFECYCLE_ACTIVE_ENV] = "1"
        try:
            steps = lifecycle.get('default', [])
            state_ready = self._lifecycle_state_exists(lifecycle)

            for step in steps:
                if step.endswith('_if_missing'):
                    entry_name = step[: -len('_if_missing')]
                    if state_ready:
                        print(f"== existing state found ({lifecycle.get('state_marker')}); "
                              f"skipping '{entry_name}' ==")
                        continue
                else:
                    entry_name = step
                self._run_entry_point(app_path, config, entry_name)
        finally:
            os.environ.pop(_LIFECYCLE_ACTIVE_ENV, None)

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