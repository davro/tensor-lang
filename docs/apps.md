# Applications in TensorLang

How to lay out, configure and test an application under `apps/`. The commands for running apps and the list of notable ones are in the README ([Applications](../README.md#applications)). Run every command from the repository root.

Applications live under `apps/` and are discovered by `app.toml`. The shared toolkit `apps/tlkit/` (`chunked_runner.py`, `sequence_player.py`, `colormap.py`) supports visual/interactive apps.

```bash
python3 tensorlang.py --list-apps
python3 tensorlang.py --app examples/hello_mlp
python3 tensorlang.py --app examples/decision_boundary
python3 tensorlang.py --app games/2048 --app-args <16 numbers>
```

## App Layout

```
apps/
└── [<category>/]
    └── <app-name>/
        ├── app.toml
        ├── <entry>.tl
        └── tests/
            └── *.tl
```

## Minimal `app.toml`

```toml
[app]
name = "hello_mlp"
description = "Minimal 2-layer MLP training example"

[requirements]
gpus = 1
memory_gb = 4

[entry_points]
main = "main.tl"
```

## App Requirements

`app.toml` may declare:

```toml
[requirements]
gpus = 1
packages = ["pygame"]
```

* `requirements.gpus` is enforced via least-utilized GPU selection and process pinning.
* `requirements.packages` are auto-installed if missing.
* `cuda_version` and `memory_gb` are display-only.
* `[gpu] device = N` can pin an app to a specific GPU.

## App Tests

```bash
python3 tensorlang.py --app examples/hello_mlp --test
```

App tests run sequentially.
