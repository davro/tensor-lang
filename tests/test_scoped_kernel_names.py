import shutil
import textwrap
from pathlib import Path

import pytest

from tensorlang.compiler import TensorCompiler

GRAMMAR = Path(__file__).resolve().parents[1] / "tensorlang.lark"

pytestmark = pytest.mark.skipif(
    shutil.which("nvcc") is None, reason="nvcc is required for TensorLang execution tests"
)


def _run(tmp_path, monkeypatch, source):
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(GRAMMAR, tmp_path / "tensorlang.lark")
    Path("test.tl").write_text(textwrap.dedent(source).strip() + "\n")
    compiler = TensorCompiler(debug_mode=False)
    ok = compiler.compile_and_execute("test.tl")
    return ok, Path("cache/test.tl")


def test_loop_and_top_level_same_tensor_name_do_not_collide(tmp_path, monkeypatch):
    source = textwrap.dedent("""\
        let x: Tensor[f32, (2,)] = [1.0, 2.0]
        for i in range(2) {
            let y: Tensor[f32, (2,)] = add(x, x)
        }
        let y: Tensor[f32, (2,)] = add(x, x)
        save(y, "out.npy")
    """)
    ok, cache_dir = _run(tmp_path, monkeypatch, source)
    assert ok is True
    assert (cache_dir / "kernel.cu").exists()
    assert "loop0_" in (cache_dir / "kernel.cu").read_text()


def test_multiple_loops_get_distinct_scoped_names(tmp_path, monkeypatch):
    source = textwrap.dedent("""\
        let x: Tensor[f32, (2,)] = [1.0, 2.0]
        for i in range(2) {
            let y: Tensor[f32, (2,)] = add(x, x)
        }
        for j in range(2) {
            let y: Tensor[f32, (2,)] = add(x, x)
        }
        save(x, "out.npy")
    """)
    ok, cache_dir = _run(tmp_path, monkeypatch, source)
    assert ok is True
    text = (cache_dir / "kernel.cu").read_text()
    assert "loop0_" in text
    assert "loop1_" in text


def test_scoped_kernel_result_is_saved(tmp_path, monkeypatch):
    source = textwrap.dedent("""\
        let x: Tensor[f32, (2,)] = [1.0, 2.0]
        for i in range(2) {
            let y: Tensor[f32, (2,)] = add(x, x)
        }
        let y: Tensor[f32, (2,)] = add(x, x)
        save(y, "out.npy")
    """)
    ok, cache_dir = _run(tmp_path, monkeypatch, source)
    assert ok is True
    assert (cache_dir / "out.npy").exists()
