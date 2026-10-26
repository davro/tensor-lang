"""
Regression tests for shape-mismatch error wording.

@EXPECT_FAILURE .tl tests only check the exit code, so they cannot notice if
an error message loses its operand names or shapes. These tests assert on the
text itself.

Two layers are covered:
  * type checker (inferred shapes): called directly with small ASTs, so no
    GPU, nvcc or grammar is needed
  * compiler (annotated lets, which the type checker does not validate):
    compiled end to end, so it needs nvcc like the other execution tests
"""
import shutil
import textwrap
from pathlib import Path

import pytest

from tensorlang.type_checker import type_checker

GRAMMAR = Path(__file__).resolve().parents[1] / "tensorlang.lark"


def _program(shape_a, shape_b, op):
    """AST for: a, b declared; `c` has no annotation so its shape is inferred."""
    return [
        {'type': 'let', 'name': 'a', 'ty': {'dtype': 'f32', 'shape': shape_a}, 'expr': {'type': 'x'}},
        {'type': 'let', 'name': 'b', 'ty': {'dtype': 'f32', 'shape': shape_b}, 'expr': {'type': 'x'}},
        {'type': 'let', 'name': 'c', 'ty': None, 'expr': {'type': op, 'args': ['a', 'b']}},
    ]


@pytest.mark.parametrize(
    "op, shape_a, shape_b, expected_fragments",
    [
        # general broadcast branch
        ("add", (2, 3), (3, 2),
         ["add:", "'c'", "'a' (2, 3)", "'b' (3, 2)"]),
        ("mult", (2, 3), (4, 5),
         ["mult:", "'c'", "'a' (2, 3)", "'b' (4, 5)"]),
        # 2D op 1D branch, both orders
        ("add", (2, 3), (4,),
         ["add:", "'a' (2, 3)", "'b' (4,)", "1D operand"]),
        ("minus", (4,), (2, 3),
         ["minus:", "'a' (4,)", "'b' (2, 3)", "1D operand"]),
        # comparison ops must report operands in source order, not swapped
        ("less", (4,), (2, 3),
         ["less:", "'a' (4,)", "'b' (2, 3)"]),
        # matmul inner-dimension mismatch
        ("matmul", (2, 3), (2, 3),
         ["matmul:", "'c'", "'a' (2, 3)", "'b' (2, 3)",
          "left has 3 columns but right has 2 rows"]),
    ],
)
def test_type_checker_mismatch_message_names_operands_and_shapes(
    capsys, op, shape_a, shape_b, expected_fragments
):
    ok, _ = type_checker(_program(shape_a, shape_b, op), {})
    out = capsys.readouterr().out

    assert ok is False
    assert "[TYPE CHECKER] error:" in out
    assert "shape mismatch" in out
    for fragment in expected_fragments:
        assert fragment in out, f"missing {fragment!r} in message:\n{out}"


@pytest.mark.parametrize(
    "op, shape_a, shape_b",
    [
        ("add", (2, 3), (2, 3)),
        ("add", (2, 3), (3,)),
        ("add", (2, 3), (1, 1)),
        ("less", (2, 3), (3,)),
        ("matmul", (2, 3), (3, 4)),
    ],
)
def test_type_checker_valid_shapes_still_pass(capsys, op, shape_a, shape_b):
    ok, env = type_checker(_program(shape_a, shape_b, op), {})
    out = capsys.readouterr().out

    assert ok is True
    assert "shape mismatch" not in out
    assert "c" in env


@pytest.mark.skipif(
    shutil.which("nvcc") is None, reason="nvcc is required for TensorLang execution tests"
)
def test_compiler_mismatch_message_for_annotated_let(tmp_path, monkeypatch, capsys):
    """An annotated let skips the type checker, so the compiler reports it."""
    from tensorlang.compiler import TensorCompiler

    monkeypatch.chdir(tmp_path)
    shutil.copyfile(GRAMMAR, tmp_path / "tensorlang.lark")
    Path("test.tl").write_text(textwrap.dedent("""\
        let a: Tensor[f32, (2,3)] = [[1.0,2.0,3.0],[4.0,5.0,6.0]]
        let b: Tensor[f32, (3,2)] = [[1.0,2.0],[3.0,4.0],[5.0,6.0]]
        let c: Tensor[f32, (2,3)] = add(a, b)
    """))

    ok = TensorCompiler(debug_mode=False).compile_and_execute("test.tl")
    out = capsys.readouterr().out

    assert ok is False
    assert "[COMPILER] Error:" in out
    for fragment in ["add:", "shape mismatch", "'c'", "'a' (2, 3)", "'b' (3, 2)"]:
        assert fragment in out, f"missing {fragment!r} in message:\n{out}"
