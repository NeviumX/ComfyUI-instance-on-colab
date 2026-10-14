"""The conventions the runtime package relies on (docs: runtime/__init__.py).

The modules refer to each other as module.name, so that replacing a name on its module (the app
action's paths.BASE, a test's monkeypatch.setattr(runtime.commands, "run", ...)) reaches every
caller. These tests keep the three ways that convention breaks out of the package."""
import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "runtime"
SOURCES = {path.stem: ast.parse(path.read_text(encoding="utf-8"), str(path)) for path in sorted(PACKAGE.glob("*.py"))}
MODULES = set(SOURCES) - {"__init__", "__main__"}
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def siblings(tree):
    """The modules of the package a module imports: from . import a, b."""
    return {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
            and node.level == 1 and node.module is None for alias in node.names}


def bindings(tree):
    """Every name the module binds anywhere, with its line, apart from its own imports of siblings."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            yield node.id, node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node.lineno
        elif isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, ast.ExceptHandler) and node.name:
            yield node.name, node.lineno
        elif isinstance(node, ast.Import) or (isinstance(node, ast.ImportFrom) and node.level == 0):
            for alias in node.names:
                yield (alias.asname or alias.name).split(".")[0], node.lineno


def at_import(tree):
    """The nodes that run when the module is imported: not function bodies, but their defaults
    and decorators, and everything else at the top level."""
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, SCOPES):
            arguments = node.args
            pending += [*getattr(node, "decorator_list", []), *arguments.defaults,
                        *(default for default in arguments.kw_defaults if default is not None)]
            continue
        yield node
        pending += ast.iter_child_nodes(node)


def depends_on(module):
    """Every module this one imports, directly or through others."""
    found, pending = set(), [module]
    while pending:
        for name in siblings(SOURCES[pending.pop()]) - found:
            found.add(name)
            pending.append(name)
    return found


@pytest.mark.parametrize("module", sorted(SOURCES))
def test_modules_import_each_other_whole(module):
    # from .commands import run would keep the function it found at import: replacing
    # commands.run afterwards would not reach this module. t() is never replaced.
    # __init__.py is the exception: it hands the notebook its names (runtime.setup, ...).
    if module == "__init__":
        return
    names = [f"line {node.lineno}: from .{node.module} import " + ", ".join(alias.name for alias in node.names)
             for node in ast.walk(SOURCES[module]) if isinstance(node, ast.ImportFrom) and node.level >= 1
             and node.module is not None and not (node.module == "messages" and [a.name for a in node.names] == ["t"])]
    assert not names, f"runtime/{module}.py imports names, not modules: " + "; ".join(names)


@pytest.mark.parametrize("module", sorted(SOURCES))
def test_no_name_hides_a_module_it_uses(module):
    # A local called drive in a module that uses drive.hide_drive_models() makes that call reach
    # the local instead (it happened with google.colab's drive, and with a list called files).
    used = siblings(SOURCES[module])
    hiding = sorted({f"{name} (line {line})" for name, line in bindings(SOURCES[module]) if name in used})
    assert not hiding, f"runtime/{module}.py binds the name of a module it imports: " + ", ".join(hiding)


@pytest.mark.parametrize("module", sorted(SOURCES))
def test_nothing_at_import_reaches_into_a_module_that_imports_it_back(module):
    # Modules that import each other work only while they reach into each other at call time:
    # at import one of them is still half made, and a name it has not defined yet is missing.
    tree = SOURCES[module]
    used = siblings(tree)
    early = sorted({f"{node.value.id}.{node.attr} (line {node.lineno})" for node in at_import(tree)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in used
                    and module in depends_on(node.value.id)})
    assert not early, f"runtime/{module}.py uses at import a module that imports it back: " + ", ".join(early)


def test_the_checks_see_what_they_are_meant_to():
    # Each check on a module written to break it, so that none of them passes by seeing nothing.
    broken = ast.parse("from . import commands, drive\nfrom .commands import run\n"
                       "def stop():\n    from google.colab import drive\n    drive.flush_and_unmount()\n"
                       "LIMIT = commands.APP_TIMEOUT\n")
    assert {name for name, _ in bindings(broken)} >= {"drive", "stop", "LIMIT"}
    assert any(isinstance(node, ast.ImportFrom) and node.module == "commands" for node in ast.walk(broken))
    reached = {f"{node.value.id}.{node.attr}" for node in at_import(broken) if isinstance(node, ast.Attribute)}
    assert reached == {"commands.APP_TIMEOUT"}  # Not drive.flush_and_unmount: that runs when stop() is called.
    assert MODULES >= {"paths", "commands", "service", "proxy"} and depends_on("service") >= {"commands", "paths"}
