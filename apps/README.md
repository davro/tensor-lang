# TensorLang Apps

## Creating a TensorLang App

Applications live under `apps/` and are described by an `app.toml`.

1. Create a directory under `apps/` (optionally inside a category folder).
2. Add an `app.toml`:

   ```toml
   [app]
   name = "my_app"
   description = "What this app does"

   [entry_points]
   main = "main.tl"

### Running a TensorLang App, Minimal example

```bash
python tensorlang.py --app examples/hello_mlp
```