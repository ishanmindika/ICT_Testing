"""Project rule: no calculation may use a percentage of price or a price ratio.
Everything is in points, ticks or R. This scans the code base and fails on any division by a price
column, pct_change, or log of a price."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICE_NAME = re.compile(r"^(adj_)?(open|high|low|close)$", re.I)


def _names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
    return out


def violations(source: str, filename: str = "<src>") -> list[str]:
    found = []
    for n in ast.walk(ast.parse(source)):
        line = getattr(n, "lineno", 0)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv)):
            if any(PRICE_NAME.match(x) for x in _names(n.right)):
                found.append(f"{filename}:{line} divides by a price")
        elif isinstance(n, ast.AugAssign) and isinstance(n.op, (ast.Div, ast.FloorDiv)):
            if any(PRICE_NAME.match(x) for x in _names(n.value)):
                found.append(f"{filename}:{line} divides by a price")
        elif isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name)):
            fn = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            args = [*n.args, *(k.value for k in n.keywords)]
            if fn == "pct_change":
                found.append(f"{filename}:{line} pct_change is a ratio")
            elif fn in {"div", "divide", "truediv", "true_divide"} and any(
                    PRICE_NAME.match(x) for a in args for x in _names(a)):
                found.append(f"{filename}:{line} divides by a price")
            elif fn in {"log", "log1p", "log10", "log2"} and any(
                    PRICE_NAME.match(x) for a in args for x in _names(a)):
                found.append(f"{filename}:{line} log of a price is a ratio")
    return found


def test_detector_catches_ratios():
    assert violations("x = high / close")
    assert violations("x = (a - b) / df['open']")
    assert violations("x = df.close.pct_change()")
    assert violations("x = np.log(df['close'])")
    assert violations("x = a.div(df.low)")
    assert not violations("x = (close - open) / tick_size\nvol = a / volume")


def test_codebase_has_no_price_ratios():
    bad = []
    for path in sorted(ROOT.rglob("*.py")):
        if "tests" in path.relative_to(ROOT).parts:
            continue
        bad += violations(path.read_text(), str(path.relative_to(ROOT)))
    assert not bad, "price ratios are banned (use points/ticks/R):\n" + "\n".join(bad)
