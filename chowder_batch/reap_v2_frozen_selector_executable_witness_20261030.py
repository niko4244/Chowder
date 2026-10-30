"""CPU-only source and live-code witness for the unchanged frozen Gen-0 selector.

An approved evidence digest is insufficient if the executing selector/gate code
or imported function bindings have drifted. Authenticate exact original source
bytes, then compare every module-owned live function to code compiled from
those same bytes WITHOUT executing the newly compiled module. This also checks
the frozen literal policy constants and imported selector aliases.

This is a same-process consistency check, NOT an external execution attestation
or defense against an attacker able to replace this witness/bridge itself.
There are no production evidence approvals; it never authorizes release.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import json.decoder
import json.encoder
import json.scanner
import math
import operator
from pathlib import Path
import re
import statistics
import sys
from types import BuiltinFunctionType, CodeType, FunctionType, ModuleType

from chowder_batch import reap_gates, reap_select, reap_serve, reap_v2_source_identity


SCHEMA = "chowder.reap-v2-frozen-selector-executable-witness.v1"
_ROOT = Path(__file__).resolve().parent
# These are original, source-reviewed Git LF blobs, not caller-supplied hashes,
# current Git HEAD, a receipt, or the live module's own claims. Git may check
# out the same bytes as CRLF on Windows; normalize ONLY line endings before
# checking the immutable Git content hash, never arbitrary whitespace/text.
FROZEN_SOURCE_SHA256 = {
    "reap_select": "eec6a908433f194b62aa111cbca239be43d22ef711e307c758ae2babf81d7502",
    "reap_gates": "524961a0c75c41e7c2d5a0c8538569fdcb123272d0a875835c34cab6c91d428b",
    "reap_serve": "d0306415fb3812380153b8b8a81be767ed9e068f9092c6542d067551be8ab4b0",
    "reap_v2_source_identity": "c08146f94c5740f588f08fd3acc6cacad02ad8316c7e677b551501562a9c99c2",
}
# Original Windows CPython 3.11.9 statistical runtime, measured independently
# from its installed Lib/statistics.py. Other runtimes must receive a separate
# source review; silently substituting a different CI/interpreter version
# cannot qualify the ORIGINAL frozen statistical gate.
FROZEN_STATISTICS_SHA256 = "99ea70b550d381eadc2099866206a716273eec54764e27c78daf7a66039588da"
FROZEN_JSON_SHA256 = {
    "__init__": "feb17670e443e5db2723f217727dcc5d5e155c40e4e6935b16061c88542f24e7",
    "decoder": "b719fbcfcebd2b174f076e71292e22b1a17d9e258dbe896c768325383bad4f80",
    "scanner": "9841566fb17315ebdd40a1ca9cb214f02cde7171b187d4dc821c80120ea853c3",
    "encoder": "7db65c95f09297d57cc6273d78858b25ffcba65bc83709c4a15584d1dbef3c38",
}
_MODULES = {
    "reap_select": reap_select,
    "reap_gates": reap_gates,
    "reap_serve": reap_serve,
    "reap_v2_source_identity": reap_v2_source_identity,
}
_LITERALS = {
    "reap_select": frozenset({
        "REQUIRED", "MARGIN", "FROZEN_SERVE_FLAGS", "FROZEN_SCORING",
        "FROZEN_INPUT_SCHEMA", "FROZEN_SOURCE_COUNTS", "FROZEN_MAX_TOKENS",
        "FROZEN_REFERENCE_MODEL_SHA256", "PAIRED_METHOD_DISCREPANCY",
    }),
    "reap_gates": frozenset({
        "Z", "EPS", "G0_TOL", "G1_THRESH", "G2_THRESH", "SPEEDUP",
        "PUBLISHED_REAP128_V2_Q4_SHA256", "DEGENERATE_TRIGRAM",
        "FROZEN_SUITE_SIZES", "FROZEN_SOURCE_SHA256",
    }),
    "reap_serve": frozenset({
        "DEFAULT_SERVER", "FLAGS", "HEALTH_TIMEOUT",
        "REGISTERED_V2_CONSTRUCTION", "REGISTERED_BF16_SHA256",
        "REGISTERED_STAGE_SHA256", "REGISTERED_QUANTIZER_SHA256",
        "REGISTERED_Q8_GGUF_SHA256", "REGISTERED_Q4_GGUF_SHA256",
        "LOG_VERBOSITY", "REGISTERED_V2_EVALUATION_ARMS",
        "REGISTERED_DENSE_Q4_SHA256",
    }),
    "reap_v2_source_identity": frozenset({
        "FROZEN_V2_BASE_ROWS", "FROZEN_V2_BASE_SHA256",
        "FROZEN_ORNITH_ADMITTED_SHA256", "FROZEN_ORNITH_POOL",
        "ORNITH_QUOTA", "ABSENT_ORNITH",
    }),
}


class SelectorExecutableRefused(ValueError):
    """The original selector source or running implementation is untrusted."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise SelectorExecutableRefused(reason)


def _top_level_functions(code: CodeType) -> dict[str, CodeType]:
    return {
        item.co_name: item for item in code.co_consts
        if isinstance(item, CodeType) and item.co_name != "<lambda>"
    }


def _code_signature(code: CodeType) -> tuple:
    """Structural code identity; marshal's memo references are not canonical."""
    return (
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
        code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code,
        tuple(_code_signature(c) if isinstance(c, CodeType) else (type(c), c)
              for c in code.co_consts),
        code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
        code.co_filename, code.co_name, code.co_qualname, code.co_firstlineno,
        code.co_linetable, code.co_exceptiontable,
    )


def _literal_assignments(tree: ast.Module) -> dict[str, object]:
    result: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id.isupper() and value is not None:
            try:
                result[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                # Expressions such as 16 * 1024 * 1024 are not literal AST.
                pass
    return result


def _builtin_identity(value: object, *, module: str, name: str) -> bool:
    return (isinstance(value, BuiltinFunctionType)
            and value.__module__ == module and value.__name__ == name)


def _git_lf_source(raw: bytes) -> bytes:
    """Normalize Windows checkout EOL only; reject mixed or lone CR bytes."""
    canonical = raw.replace(b"\r\n", b"\n")
    _require(b"\r" not in canonical, "frozen source has noncanonical line endings")
    return canonical


def _matches_default(value: object, expression: ast.expr,
                     module: ModuleType) -> bool:
    """Check literal and bound-source defaults without evaluating source AST."""
    try:
        expected = ast.literal_eval(expression)
    except (ValueError, TypeError, SyntaxError):
        if isinstance(expression, ast.Name):
            owner = module if hasattr(module, expression.id) else builtins
            return value is getattr(owner, expression.id, object())
        if (isinstance(expression, ast.Attribute)
                and isinstance(expression.value, ast.Name)):
            parent_name = expression.value.id
            owner = module if hasattr(module, parent_name) else builtins
            parent = getattr(owner, parent_name, None)
            if parent is None:
                return False
            if (isinstance(value, BuiltinFunctionType)
                    and getattr(value, "__self__", None) is parent
                    and value.__name__ == expression.attr):
                return True
            return value is getattr(parent, expression.attr, None)
        return False
    return type(value) is type(expected) and value == expected


def _check_defaults(live: FunctionType, node: ast.FunctionDef,
                    module: ModuleType) -> bool:
    positional = live.__defaults__ or ()
    keywords = live.__kwdefaults__ or {}
    expected_kw = {
        argument.arg: expression
        for argument, expression in zip(node.args.kwonlyargs, node.args.kw_defaults)
        if expression is not None
    }
    return (len(positional) == len(node.args.defaults)
            and all(_matches_default(value, expression, module)
                    for value, expression in zip(positional, node.args.defaults))
            and set(keywords) == set(expected_kw)
            and all(_matches_default(keywords[name], expression, module)
                    for name, expression in expected_kw.items()))


def _check_json_runtime() -> dict[str, object]:
    """Authenticate the actual JSON parser/encoder used for frozen row intake."""
    modules = {
        "__init__": json, "decoder": json.decoder,
        "scanner": json.scanner, "encoder": json.encoder,
    }
    verified = 0
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        _require(path.is_file() and not path.is_symlink()
                 and path.name == ("__init__.py" if name == "__init__" else f"{name}.py")
                 and sys.modules.get(module.__name__) is module
                 and Path(module.__spec__.origin).resolve() == path
                 and path.parent == Path(json.__file__).resolve().parent,
                 f"original JSON runtime module origin invalid: {name}")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == FROZEN_JSON_SHA256[name],
                 f"original JSON runtime source SHA drift: {name}")
        source = raw.decode("utf-8")
        tree = ast.parse(source)
        expected = _top_level_functions(compile(
            source, str(path), "exec", dont_inherit=True,
            optimize=sys.flags.optimize,
        ))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                live = vars(module).get(node.name)
                _require(isinstance(live, FunctionType)
                         and live.__module__ == module.__name__
                         and live.__globals__ is vars(module)
                         and _code_signature(live.__code__)
                         == _code_signature(expected[node.name])
                         and _check_defaults(live, node, module),
                         f"original JSON function changed: {name}.{node.name}")
                verified += 1
            elif isinstance(node, ast.ClassDef):
                cls = vars(module).get(node.name)
                _require(isinstance(cls, type) and cls.__module__ == module.__name__,
                         f"original JSON class changed: {name}.{node.name}")
                methods = _top_level_functions(expected[node.name])
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        live = getattr(cls, child.name, None)
                        _require(isinstance(live, FunctionType)
                                 and live.__globals__ is vars(module)
                                 and _code_signature(live.__code__)
                                 == _code_signature(methods[child.name])
                                 and _check_defaults(live, child, module),
                                 f"original JSON method changed: {name}.{node.name}.{child.name}")
                        verified += 1
    _require(json.JSONDecoder is json.decoder.JSONDecoder
             and json.JSONEncoder is json.encoder.JSONEncoder
             and json.decoder.scanstring is json.decoder.c_scanstring
             and _builtin_identity(json.decoder.scanstring, module="_json", name="scanstring")
             and json.scanner.make_scanner.__module__ == "_json"
             and json.scanner.make_scanner.__name__ == "Scanner"
             and type(json._default_decoder) is json.decoder.JSONDecoder
             and type(json._default_decoder.scan_once) is json.scanner.make_scanner
             and json._default_decoder.parse_object is json.decoder.JSONObject
             and json._default_decoder.parse_array is json.decoder.JSONArray
             and json._default_decoder.parse_string is json.decoder.scanstring
             and json._default_decoder.object_hook is None
             and json._default_decoder.object_pairs_hook is None
             and json._default_decoder.strict is True
             and json._default_decoder.parse_int is int
             and json._default_decoder.parse_float is float
             and json._default_decoder.scan_once.parse_int is int
             and json._default_decoder.scan_once.parse_float is float
             and json._default_decoder.scan_once.object_hook is None
             and json._default_decoder.scan_once.object_pairs_hook is None
             and json._default_decoder.scan_once.strict is True
             and json._default_decoder.parse_constant.__self__ is json.decoder._CONSTANTS
             and json._default_decoder.scan_once.parse_constant.__self__
             is json.decoder._CONSTANTS,
             "original JSON decoder binding changed")
    return {"source_sha256": dict(FROZEN_JSON_SHA256),
            "verified_functions_and_methods": verified}


def _check_numeric_dependencies() -> dict[str, object]:
    """Refuse ambient stdlib hijacks that can turn a frozen Gate1 FAIL to PASS."""
    path = Path(statistics.__file__).resolve()
    _require(path.is_file() and not path.is_symlink()
             and sys.modules.get("statistics") is statistics
             and Path(statistics.__spec__.origin).resolve() == path
             and reap_gates.statistics is statistics
             and reap_gates.math is math and reap_gates.json is json
             and reap_select.json is json and reap_select.hashlib is hashlib
             and reap_select.re is re and reap_v2_source_identity.hashlib is hashlib,
             "frozen selector statistical/stdlib module binding changed")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == FROZEN_STATISTICS_SHA256,
             "original statistical runtime source SHA drift")
    source = raw.decode("utf-8")
    compiled = compile(source, str(path), "exec", dont_inherit=True,
                       optimize=sys.flags.optimize)
    tree = ast.parse(source)
    expected = _top_level_functions(compiled)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    # CPython's optimized statistics module replaces this pure-Python helper
    # with the built-in _statistics implementation after module execution.
    replaced = {"_normal_dist_inv_cdf"}
    _require(functions.keys() - replaced <= expected.keys(),
             "original statistical runtime function inventory incomplete")
    for name in functions.keys() - replaced:
        live = vars(statistics).get(name)
        _require(isinstance(live, FunctionType)
                 and live.__module__ == "statistics"
                 and live.__globals__ is vars(statistics)
                 and _code_signature(live.__code__) == _code_signature(expected[name])
                 and _check_defaults(live, functions[name], statistics),
                 f"statistics.{name}: executing statistical function differs from source")
    _require(_builtin_identity(math.sqrt, module="math", name="sqrt")
             and _builtin_identity(math.isfinite, module="math", name="isfinite")
             and _builtin_identity(math.fsum, module="math", name="fsum")
             and statistics.fsum is math.fsum
             and statistics.mul is operator.mul
             and _builtin_identity(statistics._normal_dist_inv_cdf,
                                   module="_statistics", name="_normal_dist_inv_cdf")
             and _builtin_identity(hashlib.sha256, module="_hashlib", name="openssl_sha256")
             and isinstance(json.loads, FunctionType)
             and json.loads.__module__ == "json"
             and json.loads.__globals__ is vars(json),
             "original statistical/hash/JSON dependency binding changed")
    return {"statistics_source_sha256": FROZEN_STATISTICS_SHA256,
            "verified_statistical_functions": len(functions.keys() - replaced)}


def _check_source_and_code(name: str, module: ModuleType) -> dict[str, object]:
    path = _ROOT / f"{name}.py"
    _require(path.is_file() and not path.is_symlink(), f"{name}: original source missing")
    metadata = path.stat()
    _require(0 < metadata.st_size <= 128 * 1024, f"{name}: source size invalid")
    raw = path.read_bytes()
    digest = hashlib.sha256(_git_lf_source(raw)).hexdigest()
    _require(len(raw) == metadata.st_size and digest == FROZEN_SOURCE_SHA256[name],
             f"{name}: frozen original source SHA drift")
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    _require(sys.modules.get(f"chowder_batch.{name}") is module
             and getattr(module, "__name__", None) == f"chowder_batch.{name}"
             and getattr(module, "__package__", None) == "chowder_batch"
             and isinstance(origin, str)
             and Path(origin).resolve() == path.resolve()
             and Path(getattr(module, "__file__", "")).resolve() == path.resolve(),
             f"{name}: loaded module does not originate from frozen source")
    try:
        source = raw.decode("utf-8")
        tree = ast.parse(source, filename=str(path))
        compiled = compile(source, str(path), "exec", dont_inherit=True,
                           optimize=sys.flags.optimize)
    except (UnicodeError, ValueError, SyntaxError, TypeError) as exc:
        raise SelectorExecutableRefused(f"{name}: frozen source cannot compile") from exc
    expected = _top_level_functions(compiled)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    declared = set(functions)
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    _require(expected.keys() == declared | classes and bool(declared),
             f"{name}: original top-level function inventory invalid")
    for function_name in declared:
        code = expected[function_name]
        live = vars(module).get(function_name)
        _require(isinstance(live, FunctionType)
                 and live.__module__ == module.__name__
                 and live.__globals__ is vars(module)
                 and _code_signature(live.__code__) == _code_signature(code),
                 f"{name}.{function_name}: executing function differs from frozen source")
        # Function defaults are mutable runtime attributes not included in a
        # code object's bytecode. All original selector defaults are literals.
        node = functions[function_name]
        try:
            defaults = tuple(ast.literal_eval(item) for item in node.args.defaults)
            keyword_defaults = {
                argument.arg: ast.literal_eval(value)
                for argument, value in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if value is not None
            }
        except (ValueError, TypeError, SyntaxError) as exc:
            raise SelectorExecutableRefused(
                f"{name}.{function_name}: original default is not a frozen literal"
            ) from exc
        _require(live.__defaults__ == (defaults or None)
                 and live.__kwdefaults__ == (keyword_defaults or None),
                 f"{name}.{function_name}: executing defaults differ from source")
    literals = _literal_assignments(tree)
    _require(_LITERALS[name].issubset(literals),
             f"{name}: frozen policy literal inventory incomplete")
    for key in _LITERALS[name]:
        actual, original = vars(module).get(key), literals[key]
        _require(type(actual) is type(original) and actual == original,
                 f"{name}.{key}: running frozen policy differs from source")
    return {"source_sha256": digest, "verified_functions": len(declared)}


def verify_frozen_selector_executable() -> dict[str, object]:
    """Refuse unpinned source, foreign modules, code swaps and policy mutation."""
    json_runtime = _check_json_runtime()
    numeric = _check_numeric_dependencies()
    results = {
        name: _check_source_and_code(name, module)
        for name, module in _MODULES.items()
    }
    _require(reap_select.GateError is reap_gates.GateError
             and reap_select.FROZEN_SOURCE_SHA256 is reap_gates.FROZEN_SOURCE_SHA256
             and reap_select._load is reap_gates._load
             and reap_select.compute is reap_gates.compute
             and reap_select.lower_bound is reap_gates.lower_bound
             and reap_select.V2SourceIdentityError
             is reap_v2_source_identity.V2SourceIdentityError
             and reap_select.frozen_v2_source_identity
             is reap_v2_source_identity.frozen_v2_source_identity,
             "frozen selector imported gate/calibration function binding changed")
    _require(reap_gates.GateError.__module__ == reap_gates.__name__
             and reap_gates.GateError.__bases__ == (Exception,)
             and reap_v2_source_identity.V2SourceIdentityError.__module__
             == reap_v2_source_identity.__name__
             and reap_v2_source_identity.V2SourceIdentityError.__bases__ == (ValueError,),
             "frozen gate/calibration exception identity changed")
    _require(reap_v2_source_identity.MAX_ADMITTED_BYTES == 16 * 1024 * 1024,
             "frozen calibration source size bound changed")
    # The admission bridge consumes native offload parsing directly. An
    # unchanged parse_offload function with replaced regex globals can still
    # fabricate a favorable offload measurement, so bind the exact patterns.
    regexes = {
        "_KEEP": (r"llama_params_fit|common_fit|n_cpu_moe|overflowing|offloaded \d+/\d+ layers|exps.*(?:overridden|CPU)", re.I),
        "_CPU_MOE": (r"n_cpu_moe\s*[=:]?\s*(\d+)", re.I),
        "_OVERFLOW": (r"\((\d+) overflowing\)", 0),
        "_OVERRIDE": (r"blk\.(\d+)\..*exps.*CPU", re.I),
        "_ALL_LAYERS": (r"offloaded (\d+)/(\d+) layers to GPU", re.I),
    }
    for name, (pattern, flags) in regexes.items():
        original = re.compile(pattern, flags)
        current = getattr(reap_serve, name, None)
        _require(isinstance(current, re.Pattern)
                 and current.pattern == original.pattern
                 and current.flags == original.flags,
                 f"frozen native serving offload regex changed: {name}")
    return {
        "schema": SCHEMA,
        "modules": results,
        "json_runtime": json_runtime,
        "numeric_runtime": numeric,
        "original_selector_unchanged": True,
        "independent_execution_attestation": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
