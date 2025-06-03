import ast
import pathlib


def load_sentiment():
    path = pathlib.Path(__file__).resolve().parents[1] / 'Capstone_EDA.py'
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'sentiment':
            module = ast.Module(body=[node], type_ignores=[])
            compiled = compile(module, filename=str(path), mode='exec')
            namespace = {}
            exec(compiled, namespace)
            return namespace['sentiment']
    raise AssertionError('sentiment function not found')

sentiment = load_sentiment()

def test_sentiment_positive():
    assert sentiment(6) == 1

def test_sentiment_negative():
    assert sentiment(5) == 0
