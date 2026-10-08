"""A bounded AST interpreter for Experiment B's small Python examples.

Candidate text is interpreted node-by-node; it is never passed to ``exec`` or
``eval``. The interpreter accepts only basic functions, literals, arithmetic,
conditionals, bounded loops/comprehensions, and explicit local workspace imports.
Unsupported operations fail closed. This limits accidental/model-generated code
from touching the host process or consuming unbounded compute.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


class RestrictedPythonError(ValueError):
    """A candidate used unsupported syntax or exceeded interpreter limits."""


@dataclass(eq=False)
class _Function:
    node: ast.FunctionDef
    globals: dict[str, Any]


@dataclass(eq=False)
class _Class:
    name: str
    members: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class _Instance:
    cls: _Class
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class _Module:
    members: dict[str, Any]


@dataclass(eq=False)
class _BoundMethod:
    function: _Function
    receiver: Any


@dataclass(eq=False)
class _NativeMethod:
    receiver: Any
    name: str


@dataclass(eq=False)
class _Builtin:
    name: str


@dataclass(eq=False)
class _Frame:
    local: dict[str, Any]
    global_: dict[str, Any]
    module_scope: bool = False


class _Return(Exception):
    def __init__(self, value: Any):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Interpreter:
    MAX_SOURCE = 8_000
    MAX_NODES = 500
    MAX_STEPS = 20_000
    MAX_DEPTH = 32
    MAX_ITEMS = 512
    MAX_TEXT = 4_096
    MAX_INT_BITS = 2_048

    BUILTIN_NAMES = frozenset({
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "print", "range", "set", "sorted",
        "str", "sum", "tuple", "zip",
    })
    METHOD_NAMES = frozenset({
        "append", "casefold", "extend", "get", "join", "lower", "replace",
        "split", "strip",
    })
    BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
    }
    CMPOPS = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.Is: lambda a, b: a is b,
        ast.IsNot: lambda a, b: a is not b,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    def __init__(self, files: Mapping[str, str] | None = None):
        self.files = dict(files or {})
        self.modules: dict[str, _Module] = {}
        self.loading: set[str] = set()
        self.builtins = {name: _Builtin(name) for name in self.BUILTIN_NAMES}
        self.steps = 0
        self.depth = 0
        self.stdout: list[str] = []
        self.stdout_chars = 0

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > self.MAX_STEPS:
            raise RestrictedPythonError("execution step budget exceeded")

    def _parse(self, source: str, filename: str) -> ast.Module:
        if not isinstance(source, str) or len(source) > self.MAX_SOURCE:
            raise RestrictedPythonError("source exceeds the review limit")
        try:
            tree = ast.parse(source, filename=filename, mode="exec")
        except (SyntaxError, ValueError, RecursionError):
            raise RestrictedPythonError("invalid Python syntax") from None
        nodes = list(ast.walk(tree))
        if len(nodes) > self.MAX_NODES:
            raise RestrictedPythonError("syntax tree exceeds the review limit")
        for node in nodes:
            if isinstance(node, (ast.While, ast.Lambda, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
                raise RestrictedPythonError(f"unsupported syntax: {type(node).__name__}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise RestrictedPythonError("dunder attributes are not permitted")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise RestrictedPythonError("dunder names are not permitted")
            if isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 1_000_000:
                    raise RestrictedPythonError("integer literal exceeds the review limit")
                if isinstance(value, float) and (not math.isfinite(value) or abs(value) > 1_000_000):
                    raise RestrictedPythonError("float literal exceeds the review limit")
                if isinstance(value, (str, bytes)) and len(value) > 512:
                    raise RestrictedPythonError("string literal exceeds the review limit")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    raise RestrictedPythonError("dynamic call targets are not permitted")
                if any(keyword.arg is None for keyword in node.keywords):
                    raise RestrictedPythonError("keyword expansion is not permitted")
        return tree

    def _check_value(self, value: Any) -> Any:
        if type(value) is int and value.bit_length() > self.MAX_INT_BITS:
            raise RestrictedPythonError("integer result exceeds the review limit")
        if type(value) is float and (not math.isfinite(value) or abs(value) > 1e100):
            raise RestrictedPythonError("float result exceeds the review limit")
        if type(value) in (str, bytes) and len(value) > self.MAX_TEXT:
            raise RestrictedPythonError("text result exceeds the review limit")
        if type(value) in (list, tuple, set, dict) and len(value) > self.MAX_ITEMS:
            raise RestrictedPythonError("collection exceeds the review limit")
        return value

    def _plain(self, value: Any, active: set[int] | None = None, depth: int = 0, budget: list[int] | None = None) -> bool:
        """Check a value graph before native comparison, hashing, sorting or coercion."""
        if value is None or type(value) in (bool, int, float, str, bytes):
            return True
        if type(value) not in (list, tuple, set, dict):
            return False
        if depth >= 16 or len(value) > self.MAX_ITEMS:
            raise RestrictedPythonError("nested value exceeds the review limit")
        active = set() if active is None else active
        budget = [0] if budget is None else budget
        identity = id(value)
        if identity in active:
            raise RestrictedPythonError("cyclic collections are not permitted")
        active.add(identity)
        budget[0] += len(value) + 1
        if budget[0] > 4_096:
            raise RestrictedPythonError("nested value exceeds the review limit")
        try:
            children = [child for pair in value.items() for child in pair] if type(value) is dict else list(value)
            return all(self._plain(child, active, depth + 1, budget) for child in children)
        finally:
            active.remove(identity)

    def _iter(self, value: Any) -> list[Any]:
        if type(value) is range:
            result = list(value)
        elif type(value) in (list, tuple, str, set, dict):
            result = list(value)
        else:
            raise RestrictedPythonError("only bounded built-in collections may be iterated")
        if len(result) > self.MAX_ITEMS:
            raise RestrictedPythonError("iteration exceeds the review limit")
        return result

    def _lookup(self, name: str, frame: _Frame) -> Any:
        if name in frame.local:
            return frame.local[name]
        if name in frame.global_:
            return frame.global_[name]
        if name in self.builtins:
            return self.builtins[name]
        raise RestrictedPythonError(f"unknown name: {name}")

    def _attribute(self, value: Any, name: str) -> Any:
        if name.startswith("__"):
            raise RestrictedPythonError("dunder attributes are not permitted")
        if isinstance(value, _Instance):
            if name in value.attrs:
                return value.attrs[name]
            member = value.cls.members.get(name)
            if isinstance(member, _Function):
                return _BoundMethod(member, value)
            if member is not None:
                return member
            raise RestrictedPythonError(f"unknown instance attribute: {name}")
        if isinstance(value, _Class):
            if name in value.members:
                return value.members[name]
            raise RestrictedPythonError(f"unknown class attribute: {name}")
        if isinstance(value, _Module):
            if name in value.members:
                return value.members[name]
            raise RestrictedPythonError(f"unknown module attribute: {name}")
        if name not in self.METHOD_NAMES:
            raise RestrictedPythonError(f"attribute is not permitted: {name}")
        valid = (
            type(value) is str and name in {"casefold", "join", "lower", "replace", "split", "strip"}
            or type(value) is list and name in {"append", "extend"}
            or type(value) is dict and name == "get"
        )
        if not valid:
            raise RestrictedPythonError(f"method is not permitted on {type(value).__name__}: {name}")
        return _NativeMethod(value, name)

    def _call_native(self, method: _NativeMethod, args: list[Any], kwargs: dict[str, Any]) -> Any:
        obj, name = method.receiver, method.name
        if name == "get" and type(obj) is dict and 1 <= len(args) <= 2 and not kwargs and self._plain(args):
            return self._check_value(obj.get(*args))
        if name == "append" and type(obj) is list and len(args) == 1 and not kwargs:
            if len(obj) >= self.MAX_ITEMS:
                raise RestrictedPythonError("collection exceeds the review limit")
            obj.append(args[0])
            return obj
        if name == "extend" and type(obj) is list and len(args) == 1 and not kwargs:
            items = self._iter(args[0])
            if len(obj) + len(items) > self.MAX_ITEMS:
                raise RestrictedPythonError("collection exceeds the review limit")
            obj.extend(items)
            return obj

        if name in {"lower", "casefold"} and type(obj) is str and not args and not kwargs:
            return self._check_value(getattr(obj, name)())
        if name == "strip" and type(obj) is str and len(args) <= 1 and not kwargs and all(type(arg) is str for arg in args):
            return self._check_value(obj.strip(*args))
        if name == "split" and type(obj) is str and len(args) <= 2 and not kwargs and all(arg is None or type(arg) in (str, int) for arg in args):
            return self._check_value(obj.split(*args))
        if name == "replace" and type(obj) is str and 2 <= len(args) <= 3 and not kwargs:
            old, new = args[:2]
            count = args[2] if len(args) == 3 else -1
            if type(old) is not str or type(new) is not str or type(count) is not int:
                raise RestrictedPythonError("replace requires text values and an integer count")
            available = len(obj) + 1 if old == "" else obj.count(old)
            replacements = available if count < 0 else min(available, count)
            if len(obj) + replacements * (len(new) - len(old)) > self.MAX_TEXT:
                raise RestrictedPythonError("replace result exceeds the review limit")
            return self._check_value(obj.replace(old, new, count))
        if name == "join" and type(obj) is str and len(args) == 1 and not kwargs:
            items = self._iter(args[0])
            if any(type(item) is not str for item in items):
                raise RestrictedPythonError("join accepts only text values")
            if sum(map(len, items)) + max(0, len(items) - 1) * len(obj) > self.MAX_TEXT:
                raise RestrictedPythonError("join result exceeds the review limit")
            return self._check_value(obj.join(items))
        raise RestrictedPythonError(f"invalid arguments for method: {name}")

    def _call_builtin(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if name == "range":
            if len(args) != 1 or kwargs or type(args[0]) is not int or abs(args[0]) > self.MAX_ITEMS:
                raise RestrictedPythonError("range requires one bounded integer argument")
            return range(args[0])
        if name == "print":
            if len(args) > 32 or set(kwargs) - {"sep", "end", "flush"}:
                raise RestrictedPythonError("unsupported print arguments")
            sep, end = kwargs.get("sep", " "), kwargs.get("end", "\n")
            if type(sep) is not str or type(end) is not str or any(type(arg) not in (bool, int, float, str) for arg in args):
                raise RestrictedPythonError("print accepts only scalar values")
            text = sep.join(map(str, args)) + end
            if self.stdout_chars + len(text) > self.MAX_TEXT:
                raise RestrictedPythonError("printed output exceeds the review limit")
            self.stdout.append(text)
            self.stdout_chars += len(text)
            return None
        if name in {"all", "any", "list", "set", "tuple", "sorted", "max", "min"} and len(args) == 1 and not kwargs:
            values = self._iter(args[0])
            if name == "all": return all(values)
            if name == "any": return any(values)
            if name == "list": return self._check_value(values)
            if name == "tuple": return self._check_value(tuple(values))
            if not self._plain(values): raise RestrictedPythonError(f"{name} accepts only plain data")
            if name == "set": return self._check_value(set(values))
            if name == "sorted": return self._check_value(sorted(values))
            if name == "max": return self._check_value(max(values))
            return self._check_value(min(values))
        if name == "dict" and len(args) <= 1 and not kwargs:
            if not args: return {}
            rows = list(args[0].items()) if type(args[0]) is dict else self._iter(args[0])
            result = {}
            for row in rows:
                pair = self._iter(row)
                if len(pair) != 2 or not self._plain(pair[0]):
                    raise RestrictedPythonError("dict accepts only bounded key/value pairs")
                result[pair[0]] = pair[1]
            return self._check_value(result)
        if name == "enumerate" and 1 <= len(args) <= 2 and not kwargs:
            values = self._iter(args[0])
            start = args[1] if len(args) == 2 else 0
            if type(start) is not int: raise RestrictedPythonError("enumerate start must be an integer")
            return [(start + index, value) for index, value in enumerate(values)]
        if name == "zip" and args and not kwargs:
            if len(args) > 32: raise RestrictedPythonError("zip has too many inputs")
            columns = [self._iter(arg) for arg in args]
            rows = min(map(len, columns), default=0)
            if rows * len(columns) > self.MAX_ITEMS: raise RestrictedPythonError("zip result exceeds the review limit")
            return list(zip(*columns))
        if name == "len" and len(args) == 1 and not kwargs and type(args[0]) in (str, bytes, list, tuple, set, dict):
            return len(args[0])
        if name == "sum" and 1 <= len(args) <= 2 and not kwargs:
            values = self._iter(args[0])
            start = args[1] if len(args) == 2 else 0
            if type(start) not in (int, float) or any(type(item) not in (int, float) for item in values):
                raise RestrictedPythonError("sum accepts only numbers")
            return self._check_value(sum(values, start))
        if name == "bool" and len(args) == 1 and not kwargs:
            return bool(args[0])
        if name == "str" and len(args) == 1 and not kwargs and (args[0] is None or type(args[0]) in (bool, int, float, str, bytes)):
            return self._check_value(str(args[0]))
        if name in {"abs", "float", "int"} and len(args) == 1 and not kwargs and type(args[0]) in (bool, int, float, str):
            if type(args[0]) is str and len(args[0]) > 128:
                raise RestrictedPythonError("numeric conversion input is too long")
            fn = {"abs": abs, "float": float, "int": int}[name]
            return self._check_value(fn(args[0]))
        raise RestrictedPythonError(f"invalid call to builtin: {name}")

    def _call(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
        self._tick()
        if isinstance(target, _Builtin):
            if target.name == "Counter":
                if len(args) != 1 or kwargs:
                    raise RestrictedPythonError("Counter requires one bounded iterable")
                values = self._iter(args[0])
                result: dict[Any, int] = {}
                for value in values:
                    if not self._plain(value):
                        raise RestrictedPythonError("Counter accepts only plain values")
                    result[value] = result.get(value, 0) + 1
                return self._check_value(result)
            return self._call_builtin(target.name, args, kwargs)
        if isinstance(target, _NativeMethod):
            return self._call_native(target, args, kwargs)
        if isinstance(target, _BoundMethod):
            return self._call_function(target.function, [target.receiver, *args], kwargs)
        if isinstance(target, _Function):
            return self._call_function(target, args, kwargs)
        if isinstance(target, _Class):
            if kwargs: raise RestrictedPythonError("class keyword arguments are not permitted")
            obj = _Instance(target)
            init = target.members.get("__init__")
            if isinstance(init, _Function): self._call_function(init, [obj, *args], {})
            elif args: raise RestrictedPythonError("class has no initializer accepting arguments")
            return obj
        raise RestrictedPythonError("call target is not in the restricted language")

    def _call_function(self, function: _Function, args: list[Any], kwargs: dict[str, Any]) -> Any:
        self._tick()
        self.depth += 1
        if self.depth > self.MAX_DEPTH:
            self.depth -= 1
            raise RestrictedPythonError("function call depth exceeded")
        try:
            positional_params = [*function.node.args.posonlyargs, *function.node.args.args]
            keyword_params = [*function.node.args.args, *function.node.args.kwonlyargs]
            names = [arg.arg for arg in (*positional_params, *function.node.args.kwonlyargs)]
            keyword_names = {arg.arg for arg in keyword_params}
            if len(args) > len(positional_params) or set(kwargs) - keyword_names:
                raise RestrictedPythonError("invalid function arguments")
            local = {arg.arg: value for arg, value in zip(positional_params, args)}
            for key, value in kwargs.items():
                if key in local: raise RestrictedPythonError("duplicate function argument")
                local[key] = value
            if set(local) != set(names): raise RestrictedPythonError("missing function argument")
            frame = _Frame(local, function.globals)
            try:
                self._block(function.node.body, frame)
            except _Return as returned:
                return self._check_value(returned.value)
            return None
        finally:
            self.depth -= 1

    def _binary_operation(self, operator_node: ast.operator, left: Any, right: Any) -> Any:
        operation = self.BINOPS.get(type(operator_node))
        if operation is None:
            raise RestrictedPythonError("unsupported binary operation")
        sequence_types = (str, list, tuple)
        numeric_types = (int, float)
        value_types = (*numeric_types, *sequence_types)
        if isinstance(operator_node, ast.Add):
            numeric = type(left) in numeric_types and type(right) in numeric_types
            matching_sequence = type(left) is type(right) and type(left) in sequence_types
            if not numeric and not matching_sequence:
                raise RestrictedPythonError("matching supported operand types required")
            if type(left) is str and len(left) + len(right) > self.MAX_TEXT:
                raise RestrictedPythonError("text result exceeds the review limit")
            if type(left) in (list, tuple) and len(left) + len(right) > self.MAX_ITEMS:
                raise RestrictedPythonError("collection exceeds the review limit")
        elif isinstance(operator_node, ast.Mult):
            numeric = type(left) in numeric_types and type(right) in numeric_types
            repeated = (
                type(left) in sequence_types and type(right) is int
            ) or (
                type(right) in sequence_types and type(left) is int
            )
            if not numeric and not repeated:
                raise RestrictedPythonError("unsupported multiplication")
            if repeated:
                sequence, count = (left, right) if type(left) in sequence_types else (right, left)
                limit = self.MAX_TEXT if type(sequence) is str else self.MAX_ITEMS
                if len(sequence) * max(0, count) > limit:
                    raise RestrictedPythonError("repeated value exceeds the review limit")
        else:
            if (
                isinstance(operator_node, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod))
                and (type(left) not in numeric_types or type(right) not in numeric_types)
            ):
                raise RestrictedPythonError("numeric operands required")
            if type(left) not in value_types or type(right) not in value_types:
                raise RestrictedPythonError("unsupported binary operand")
        return self._check_value(operation(left, right))

    def _eval(self, node: ast.AST, frame: _Frame) -> Any:
        self._tick()
        if isinstance(node, ast.Constant): return self._check_value(node.value)
        if isinstance(node, ast.Name): return self._lookup(node.id, frame)
        if isinstance(node, ast.List): return self._check_value([self._eval(item, frame) for item in node.elts])
        if isinstance(node, ast.Tuple): return self._check_value(tuple(self._eval(item, frame) for item in node.elts))
        if isinstance(node, ast.Set):
            values = [self._eval(item, frame) for item in node.elts]
            if not self._plain(values): raise RestrictedPythonError("set literals accept only plain data")
            return self._check_value(set(values))
        if isinstance(node, ast.Dict):
            result = {}
            for key_node, value_node in zip(node.keys, node.values):
                key = self._eval(key_node, frame)
                if not self._plain(key): raise RestrictedPythonError("dictionary keys must be plain data")
                result[key] = self._eval(value_node, frame)
            return self._check_value(result)
        if isinstance(node, ast.Attribute): return self._attribute(self._eval(node.value, frame), node.attr)
        if isinstance(node, ast.Subscript):
            container, key = self._eval(node.value, frame), self._eval(node.slice, frame)
            if type(container) not in (str, bytes, list, tuple, dict):
                raise RestrictedPythonError("subscript target is not a built-in collection")
            if type(container) is dict and not self._plain(key):
                raise RestrictedPythonError("dictionary keys must be plain data")
            return self._check_value(container[key])
        if isinstance(node, ast.Slice):
            lower = self._eval(node.lower, frame) if node.lower else None
            upper = self._eval(node.upper, frame) if node.upper else None
            step = self._eval(node.step, frame) if node.step else None
            if any(value is not None and type(value) is not int for value in (lower, upper, step)):
                raise RestrictedPythonError("slice bounds must be integers")
            return slice(lower, upper, step)
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, frame)
            if isinstance(node.op, ast.Not): return not value
            if type(value) in (int, float) and isinstance(node.op, ast.USub): return self._check_value(-value)
            if type(value) in (int, float) and isinstance(node.op, ast.UAdd): return value
            raise RestrictedPythonError("unsupported unary operation")
        if isinstance(node, ast.BinOp):
            left, right = self._eval(node.left, frame), self._eval(node.right, frame)
            return self._binary_operation(node.op, left, right)
        if isinstance(node, ast.BoolOp):
            value = self._eval(node.values[0], frame)
            for child in node.values[1:]:
                if isinstance(node.op, ast.And) and not value: return value
                if isinstance(node.op, ast.Or) and value: return value
                value = self._eval(child, frame)
            return value
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, frame)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, frame)
                operation = self.CMPOPS.get(type(op))
                if operation is None: raise RestrictedPythonError("unsupported comparison")
                if isinstance(op, (ast.Is, ast.IsNot)):
                    pass
                elif isinstance(op, (ast.In, ast.NotIn)):
                    if type(right) not in (str, bytes, list, tuple, set, dict):
                        raise RestrictedPythonError("membership requires a built-in collection")
                    if not self._plain(left) or not self._plain(right):
                        raise RestrictedPythonError("membership accepts only plain data")
                elif not self._plain(left) or not self._plain(right):
                    raise RestrictedPythonError("comparison accepts only plain data")
                if not operation(left, right): return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._eval(node.body if self._eval(node.test, frame) else node.orelse, frame)
        if isinstance(node, ast.Call):
            target = self._eval(node.func, frame)
            args = [self._eval(arg, frame) for arg in node.args]
            kwargs = {item.arg: self._eval(item.value, frame) for item in node.keywords}
            return self._check_value(self._call(target, args, kwargs))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            output: Any = [] if isinstance(node, ast.ListComp) else set() if isinstance(node, ast.SetComp) else {}
            comp = _Frame(dict(frame.local), frame.global_)
            def visit(index: int) -> None:
                self._tick()
                if index == len(node.generators):
                    if isinstance(node, ast.DictComp):
                        key, value = self._eval(node.key, comp), self._eval(node.value, comp)
                        if not self._plain(key): raise RestrictedPythonError("comprehension keys must be plain data")
                        output[key] = value
                    else:
                        value = self._eval(node.elt, comp)
                        if isinstance(output, set):
                            if not self._plain(value): raise RestrictedPythonError("set comprehensions require plain data")
                            output.add(value)
                        else:
                            output.append(value)
                    if len(output) > self.MAX_ITEMS: raise RestrictedPythonError("comprehension exceeds the review limit")
                    return
                generator = node.generators[index]
                for value in self._iter(self._eval(generator.iter, comp)):
                    self._assign(generator.target, value, comp)
                    if all(self._eval(test, comp) for test in generator.ifs): visit(index + 1)
            visit(0)
            return self._check_value(output)
        raise RestrictedPythonError(f"unsupported expression: {type(node).__name__}")

    def _assign(self, target: ast.AST, value: Any, frame: _Frame) -> None:
        if isinstance(target, ast.Name):
            (frame.global_ if frame.module_scope else frame.local)[target.id] = self._check_value(value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            values = self._iter(value)
            if len(values) != len(target.elts): raise RestrictedPythonError("unpacking length mismatch")
            for item, child in zip(values, target.elts): self._assign(child, item, frame)
            return
        if isinstance(target, ast.Subscript):
            container, key = self._eval(target.value, frame), self._eval(target.slice, frame)
            if type(container) not in (list, dict): raise RestrictedPythonError("only list and dict items may be assigned")
            if type(container) is dict and not self._plain(key): raise RestrictedPythonError("dictionary keys must be plain data")
            container[key] = self._check_value(value)
            self._check_value(container)
            return
        if isinstance(target, ast.Attribute):
            obj = self._eval(target.value, frame)
            if target.attr.startswith("__"): raise RestrictedPythonError("dunder attributes are not permitted")
            if isinstance(obj, _Instance): obj.attrs[target.attr] = self._check_value(value); return
            if isinstance(obj, _Class): obj.members[target.attr] = self._check_value(value); return
        raise RestrictedPythonError("unsupported assignment target")

    def _block(self, statements: list[ast.stmt], frame: _Frame) -> None:
        for statement in statements: self._statement(statement, frame)

    def _statement(self, node: ast.stmt, frame: _Frame) -> None:
        self._tick()
        if isinstance(node, ast.FunctionDef):
            self._validate_function(node)
            frame.global_[node.name] = _Function(node, frame.global_)
            return
        if isinstance(node, ast.Return): raise _Return(self._eval(node.value, frame) if node.value else None)
        if isinstance(node, ast.Expr): self._eval(node.value, frame); return
        if isinstance(node, ast.Pass): return
        if isinstance(node, ast.Assign):
            value = self._eval(node.value, frame)
            for target in node.targets: self._assign(target, value, frame)
            return
        if isinstance(node, ast.AugAssign):
            current, value = self._eval(node.target, frame), self._eval(node.value, frame)
            self._assign(node.target, self._binary_operation(node.op, current, value), frame)
            return
        if isinstance(node, ast.If):
            self._block(node.body if self._eval(node.test, frame) else node.orelse, frame)
            return
        if isinstance(node, ast.For):
            items = self._iter(self._eval(node.iter, frame))
            for value in items:
                self._tick()
                self._assign(node.target, value, frame)
                try: self._block(node.body, frame)
                except _Continue: continue
                except _Break: break
            else:
                self._block(node.orelse, frame)
            return
        if isinstance(node, ast.Break): raise _Break()
        if isinstance(node, ast.Continue): raise _Continue()
        if isinstance(node, ast.ClassDef):
            if node.decorator_list or node.bases or node.keywords:
                raise RestrictedPythonError("class decorators, inheritance, and keywords are not permitted")
            cls = _Class(node.name)
            frame.global_[node.name] = cls
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    self._validate_function(member, allow_init=True)
                    cls.members[member.name] = _Function(member, frame.global_)
                elif isinstance(member, ast.Assign) and all(isinstance(target, ast.Name) for target in member.targets):
                    value = self._eval(member.value, frame)
                    for target in member.targets: cls.members[target.id] = value
                elif isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant) and type(member.value.value) is str:
                    continue
                else:
                    raise RestrictedPythonError("classes may contain only methods and simple assignments")
            return
        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module or any(alias.name == "*" for alias in node.names):
                raise RestrictedPythonError("only explicit absolute local imports are permitted")
            if node.module in {"typing", "collections"}:
                allowed_imports = {
                    "typing": {"Dict", "List", "Tuple", "Set"},
                    "collections": {"Counter"},
                }
                for alias in node.names:
                    if alias.name not in allowed_imports[node.module]:
                        raise RestrictedPythonError("import is outside the annotation/helper whitelist")
                    if node.module == "typing":
                        imported = {"Dict": dict, "List": list, "Tuple": tuple, "Set": set}[alias.name]
                    else:
                        imported = _Builtin("Counter")
                    frame.global_[alias.asname or alias.name] = imported
                return
            module = self._load_module(node.module)
            for alias in node.names:
                if alias.name not in module.members: raise RestrictedPythonError("imported name is missing")
                frame.global_[alias.asname or alias.name] = module.members[alias.name]
            return
        raise RestrictedPythonError(f"unsupported statement: {type(node).__name__}")

    def _validate_function(self, node: ast.FunctionDef, allow_init: bool = False) -> None:
        if node.decorator_list:
            raise RestrictedPythonError("decorators are not permitted")
        if node.returns is not None:
            annotation = node.returns
            allowed_names = {"None", "bool", "int", "float", "str", "list", "dict", "tuple"}
            allowed_subscripts = {
                ("Dict", "str", "int"),
                ("List", "str"),
                ("List", "int"),
                ("Tuple", "str", "int"),
                ("Set", "str"),
            }
            simple_name = isinstance(annotation, ast.Constant) and annotation.value is None
            simple_name = simple_name or (
                isinstance(annotation, ast.Name) and annotation.id in allowed_names
            )
            simple_subscript = (
                isinstance(annotation, ast.Subscript)
                and isinstance(annotation.value, ast.Name)
                and all(isinstance(child, ast.Name) for child in (annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]))
                and (annotation.value.id, *(child.id for child in (annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]))) in allowed_subscripts
            )
            if not simple_name and not simple_subscript:
                raise RestrictedPythonError("unsupported return annotation")

        args = node.args
        if args.defaults or any(item is not None for item in args.kw_defaults) or args.vararg or args.kwarg:
            raise RestrictedPythonError("default and variadic arguments are not permitted")
        allowed_annotations = {"bool", "int", "float", "str", "list", "dict", "tuple"}
        allowed_argument_annotations = {"str", "int", "float", "bool", "None"}
        if any(
            arg.annotation is not None
            and not (
                isinstance(arg.annotation, ast.Name)
                and arg.annotation.id in allowed_argument_annotations
            )
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        ):
            raise RestrictedPythonError("unsupported argument annotation")
        if node.name.startswith("__") and not (allow_init and node.name == "__init__"):
            raise RestrictedPythonError("dunder function names are not permitted")
        for child in ast.walk(node):
            if child is not node and isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                raise RestrictedPythonError("nested definitions are not permitted")
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == node.name:
                raise RestrictedPythonError("direct recursive calls are not permitted")

    def _load_module(self, name: str) -> _Module:
        if not name.isidentifier() or f"{name}.py" not in self.files:
            raise RestrictedPythonError("imports must refer to an in-workspace Python module")
        if name in self.loading: raise RestrictedPythonError("cyclic local imports are not permitted")
        if name in self.modules: return self.modules[name]
        self.loading.add(name)
        try:
            tree = self._parse(self.files[f"{name}.py"], f"{name}.py")
            members: dict[str, Any] = {}
            module = _Module(members)
            self.modules[name] = module
            self._block(tree.body, _Frame(members, members, module_scope=True))
            return module
        except (_Return, _Break, _Continue):
            self.modules.pop(name, None)
            raise RestrictedPythonError("invalid module control flow") from None
        except Exception:
            self.modules.pop(name, None)
            raise
        finally:
            self.loading.remove(name)

    def load_candidate(self, source: str) -> dict[str, Any]:
        tree = self._parse(source, "<teacher-code-candidate>")
        if not tree.body or any(
            not isinstance(node, (ast.FunctionDef, ast.ImportFrom))
            for node in tree.body
        ):
            raise RestrictedPythonError("only function definitions and whitelisted imports are accepted")
        members: dict[str, Any] = {}
        self._block(tree.body, _Frame(members, members, module_scope=True))
        return members

    def call_candidate(self, function: _Function, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.steps = 0
        self.depth = 0
        self.stdout.clear()
        self.stdout_chars = 0
        return self._call_function(function, list(args), kwargs)

    def execute_workspace(self, task_name: str) -> str:
        if task_name == "two_file_fix":
            quadruple = self._load_module("api").members.get("quadruple")
            return "2 passed" if self._call(quadruple, [0], {}) == 0 and self._call(quadruple, [7], {}) == 28 else "FAILED 2 - double is wrong"
        if task_name == "class_counter":
            cls = self._load_module("counter").members.get("Counter")
            first, second = self._call(cls, [], {}), self._call(cls, [], {})
            passed = (
                self._call(self._attribute(first, "add"), [2], {}) == 2
                and self._call(self._attribute(first, "add"), [3], {}) == 5
                and self._call(self._attribute(second, "add"), [7], {}) == 7
                and self._attribute(first, "total") == 5
                and self._attribute(second, "total") == 7
            )
            return "2 passed" if passed else "FAILED 2 - shared class state"
        return "ERROR: no restricted interpreter contract for this development task"


class _SafeCallable:
    def __init__(self, interpreter: _Interpreter, function: _Function):
        self._interpreter = interpreter
        self._function = function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        safe_args, safe_kwargs = _copy_external((args, kwargs))
        result = self._interpreter.call_candidate(self._function, safe_args, safe_kwargs)
        return _copy_external(result)

    @property
    def last_stdout(self) -> str:
        return "".join(self._interpreter.stdout)


def _copy_external(
    value: Any,
    active: set[int] | None = None,
    depth: int = 0,
    budget: list[int] | None = None,
    memo: dict[int, Any] | None = None,
) -> Any:
    if depth > 16:
        raise RestrictedPythonError("external value nesting exceeds the review limit")
    if value is None or type(value) in (bool, str, bytes, int, float):
        if type(value) in (str, bytes) and len(value) > _Interpreter.MAX_TEXT:
            raise RestrictedPythonError("external text exceeds the review limit")
        if type(value) is int and value.bit_length() > _Interpreter.MAX_INT_BITS:
            raise RestrictedPythonError("external integer exceeds the review limit")
        if type(value) is float and not math.isfinite(value):
            raise RestrictedPythonError("external float must be finite")
        return value
    if type(value) not in (list, tuple, set, dict):
        raise RestrictedPythonError("only plain data may cross the interpreter boundary")
    if len(value) > _Interpreter.MAX_ITEMS:
        raise RestrictedPythonError("external collection exceeds the review limit")
    active = set() if active is None else active
    budget = [0] if budget is None else budget
    memo = {} if memo is None else memo
    identity = id(value)
    if identity in active:
        raise RestrictedPythonError("cyclic external collections are not permitted")
    if identity in memo:
        return memo[identity]
    active.add(identity)
    budget[0] += len(value) + 1
    if budget[0] > 4_096:
        active.remove(identity)
        raise RestrictedPythonError("external value exceeds the review limit")
    try:
        if type(value) is list:
            copied = [_copy_external(item, active, depth + 1, budget, memo) for item in value]
        elif type(value) is tuple:
            copied = tuple(_copy_external(item, active, depth + 1, budget, memo) for item in value)
        elif type(value) is set:
            copied = {_copy_external(item, active, depth + 1, budget, memo) for item in value}
        else:
            copied = {
                _copy_external(key, active, depth + 1, budget, memo): _copy_external(item, active, depth + 1, budget, memo)
                for key, item in value.items()
            }
        memo[identity] = copied
        return copied
    finally:
        active.remove(identity)


def candidate_namespace(source: str) -> dict[str, _SafeCallable]:
    """Return bounded-call wrappers for top-level candidate functions."""
    interpreter = _Interpreter()
    members = interpreter.load_candidate(source)
    return {
        name: _SafeCallable(interpreter, value)
        for name, value in members.items()
        if isinstance(value, _Function)
    }


def execute_workspace(task_name: str, files: Mapping[str, str]) -> str:
    """Interpret a fixed development workspace without executing Python source."""
    if not isinstance(files, Mapping) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in files.items()):
        return "ERROR: invalid in-memory workspace"
    if sum(len(source) for source in files.values()) > 32_000:
        return "ERROR: workspace exceeds the review limit"
    try:
        return _Interpreter(files).execute_workspace(task_name)
    except Exception as error:
        if task_name == "two_file_fix":
            return f"FAILED 2 - double is wrong ({type(error).__name__})"
        if task_name == "class_counter":
            return f"FAILED 2 - shared class state ({type(error).__name__})"
        return f"ERROR: restricted interpreter failed: {type(error).__name__}"
