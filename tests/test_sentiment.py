import ast
import pathlib
import sys


def load_sentiment():
    path = pathlib.Path(__file__).resolve().parents[1] / "Capstone_EDA.py"
    source = path.read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sentiment":
            ast.fix_missing_locations(node)
            code = compile(ast.Module([node], []), filename=str(path), mode="exec")
            scope = {}
            exec(code, scope)
            return scope["sentiment"]
    raise ImportError("sentiment function not found")


sentiment = load_sentiment()

import pytest

@pytest.mark.parametrize("rating, expected", [
    (6, 1),
    (10, 1),
    (5, 0),
    (1, 0),
])
def test_sentiment(rating, expected):
    assert sentiment(rating) == expected
