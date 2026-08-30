"""Import-graph guard for ``src/ticksim`` (spine AD-4, AD-7).

Walks the AST of every module in ``src/ticksim`` and asserts:
  * no import targets ``src.data`` / ``src.research`` / ``src.detection`` /
    ``src.execution`` / ``src.ml`` / ``src.risk`` / ``src.monitoring`` /
    ``src.dashboard`` (AD-4 -- package isolation);
  * the leaf modules ``config.py`` and ``orders.py`` import nothing from
    ``src.ticksim`` (AD-7 -- dependency direction points inward).

Designed to grow: as ``book.py``, ``events.py``, ``fills.py`` etc. land, add
their permitted-edge rows to ``PERMITTED_INTERNAL_EDGES``.

Known blind spot: an AST check cannot see dynamic imports
(``importlib.import_module(...)``, ``__import__(...)``). ``src/ticksim`` must not
use them; that convention is out of scope for a static check.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TICKSIM_DIR = REPO_ROOT / "src" / "ticksim"

FORBIDDEN_SRC_PACKAGES = {
    "src.data",
    "src.research",
    "src.detection",
    "src.execution",
    "src.ml",
    "src.risk",
    "src.monitoring",
    "src.dashboard",
}

# spine AD-7 permitted internal import edges (module -> set of ticksim modules it
# may import). Absent key => the module may import nothing from src.ticksim.
PERMITTED_INTERNAL_EDGES: dict[str, set[str]] = {
    "config": set(),
    "orders": set(),
    "book": {"config"},
    "events": {"book", "orders"},
    "fills": {"book", "orders", "config"},
    "sim": {"config", "book", "orders", "events", "fills"},
    "report": {"orders", "config"},
    "invariants": {"sim", "orders"},
    "part_a": {"orders", "config"},
}


def _module_files() -> list[Path]:
    return sorted(p for p in TICKSIM_DIR.rglob("*.py"))


def _file_package(path: Path) -> str:
    """Dotted package that ``path`` lives in, e.g. ``src.ticksim.parity``."""
    rel = path.resolve().relative_to(REPO_ROOT)
    return ".".join(rel.parts[:-1])


def _resolve_imports(source: str, file_package: str) -> set[str]:
    """Every dotted module name an ``import`` / ``from ... import`` can reach.

    Relative imports are resolved against ``file_package``. For
    ``from X import a, b`` both ``X`` and ``X.a`` / ``X.b`` are emitted, so the
    module-level form ``from src import data`` is caught as ``src.data``.
    """
    names: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                pkg_parts = file_package.split(".") if file_package else []
                drop = node.level - 1
                base_parts = pkg_parts[: len(pkg_parts) - drop] if drop else pkg_parts
                if node.module:
                    base_parts = [*base_parts, *node.module.split(".")]
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                names.add(base)
                for alias in node.names:
                    names.add(f"{base}.{alias.name}")
    return names


def _imported_modules(path: Path) -> set[str]:
    return _resolve_imports(path.read_text(), _file_package(path))


def _hits_forbidden(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        for pkg in FORBIDDEN_SRC_PACKAGES
        if name == pkg or name.startswith(pkg + ".")
    }


def _internal_targets(names: set[str]) -> set[str]:
    """The set of ``src.ticksim`` submodule stems referenced in ``names``."""
    targets: set[str] = set()
    for name in names:
        if name == "src.ticksim" or not name.startswith("src.ticksim."):
            continue
        targets.add(name.removeprefix("src.ticksim.").split(".")[0])
    return targets


def test_module_files_nonempty() -> None:
    assert _module_files(), f"no .py modules found under {TICKSIM_DIR}"


class TestImportResolver:
    """Pin the resolver itself against known relative-import forms."""

    def test_plain_import(self) -> None:
        got = _resolve_imports("import databento, sortedcontainers", "src.ticksim")
        assert got == {"databento", "sortedcontainers"}

    def test_absolute_from_import(self) -> None:
        got = _resolve_imports("from src.data import models", "src.ticksim")
        assert "src.data" in got and "src.data.models" in got

    def test_module_level_from_import(self) -> None:
        got = _resolve_imports("from src import data, research", "src.ticksim")
        assert {"src", "src.data", "src.research"} <= got

    def test_single_dot_relative_module(self) -> None:
        got = _resolve_imports("from . import book", "src.ticksim")
        assert "src.ticksim.book" in got

    def test_single_dot_relative_symbol(self) -> None:
        got = _resolve_imports("from .book import OrderBook", "src.ticksim")
        assert "src.ticksim.book" in got and "src.ticksim.book.OrderBook" in got

    def test_double_dot_relative_from_subpackage(self) -> None:
        # from src/ticksim/parity/foo.py: `from ..orders import Fill`
        got = _resolve_imports("from ..orders import Fill", "src.ticksim.parity")
        assert "src.ticksim.orders" in got

    def test_triple_dot_relative_reaches_src(self) -> None:
        # `from ...data import x` inside src.ticksim.parity resolves to src.data
        got = _resolve_imports("from ...data import x", "src.ticksim.parity")
        assert "src.data" in got

    def test_resolver_flags_forbidden_relative_import(self) -> None:
        got = _resolve_imports("from ...ml import inference", "src.ticksim.parity")
        assert _hits_forbidden(got) == {"src.ml", "src.ml.inference"}


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
class TestTicksimImportIsolation:
    def test_no_forbidden_src_package_import(self, path: Path) -> None:
        hits = _hits_forbidden(_imported_modules(path))
        assert hits == set(), f"{path.name} imports forbidden packages: {hits}"

    def test_internal_edges_are_permitted(self, path: Path) -> None:
        if path.stem == "__init__":
            pytest.skip("package marker")
        internal = _internal_targets(_imported_modules(path))
        permitted = PERMITTED_INTERNAL_EDGES.get(path.stem, set())
        assert internal <= permitted, (
            f"{path.name} imports {internal - permitted} from src.ticksim, "
            f"not in its permitted edge set {permitted} (spine AD-7)"
        )


class TestLeafModulesArePure:
    """``config.py`` and ``orders.py`` pull in no other ``src.ticksim`` module."""

    @pytest.mark.parametrize("stem", ["config", "orders"])
    def test_leaf_imports_nothing_internal(self, stem: str) -> None:
        path = TICKSIM_DIR / f"{stem}.py"
        internal = {
            name for name in _imported_modules(path) if name.startswith("src.ticksim")
        }
        assert internal == set(), f"{stem}.py imports {internal}"

    def test_config_and_orders_exist(self) -> None:
        assert (TICKSIM_DIR / "config.py").is_file()
        assert (TICKSIM_DIR / "orders.py").is_file()
