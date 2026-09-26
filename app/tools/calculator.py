"""Calculator tool: safely evaluates arithmetic expressions.

We do NOT use Python's built-in eval() on raw text - that would let the LLM
(or a malicious prompt) run arbitrary code like __import__('os').system('rm -rf /').
Instead we parse the expression into an Abstract Syntax Tree (AST) and only
allow a small whitelist of numeric operations to be executed.
"""

import ast
import operator

# Whitelist of AST node types -> the actual Python operator they map to.
# Anything not in this dict (function calls, attribute access, imports, etc.)
# will be rejected before it ever runs.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_node(node: ast.AST) -> float:
    """Recursively walk the AST, evaluating only whitelisted node types."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        return _ALLOWED_OPERATORS[op_type](_evaluate_node(node.operand))

    # Anything else (function calls, names, attributes, subscripts...) is rejected.
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result as a string.

    Supports + - * / // % ** and parentheses, e.g. "(12 + 8) * 3 / 4".
    Raises ValueError with a human-readable message on anything unsafe or invalid,
    so the agent can show that message back to the model as an "Observation".
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse '{expression}' as a math expression: {exc}")

    result = _evaluate_node(parsed.body)
    return str(result)


TOOL_SPEC = {
    "name": "calculator",
    "description": (
        "Evaluate a basic arithmetic expression. Input must be a single math "
        "expression using only numbers and + - * / // % ** ( ). "
        "Example input: '(340 + 12) * 3 / 4'"
    ),
}
