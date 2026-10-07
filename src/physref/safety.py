"""Hard runtime guard against any write into ROOT (the frozen Batch-1 repository).

`assert_safe_output(path)` is called before every experiment/run directory is created. It
ABORTS (raises RootWriteError, after printing the attempted path) when the output path:
  1. is, or lies inside, ROOT (recorded path, $PINNRP_ROOT_PATH, or any git checkout whose
     origin remote is the PINN-replication repository) - checked on the literal AND the
     symlink-resolved path;
  2. reaches ROOT through a symlink anywhere along the path;
  3. contains a ROOT-style result directory component (`results/`, `results_optimization/`);
  4. (strict mode) is not inside PINN-RP/results_batch2/.
Non-strict mode additionally admits the system temp directory; it exists ONLY so that the
software tests can build throw-away run directories. The experiment runner uses strict mode.
"""
import sys
import tempfile
from pathlib import Path

from .paths import FORBIDDEN_OUTPUT_DIRNAMES, RESULTS_B2, ROOT_REMOTE_MARKER, root_candidates


class RootWriteError(PermissionError):
    """Raised when an output path could write into ROOT or a ROOT-derived output tree."""


def _inside(p: Path, parent: Path) -> bool:
    return p == parent or parent in p.parents


def _git_toplevel_is_root(p: Path) -> bool:
    """True if `p` lies inside a git checkout whose origin remote names PINN-replication."""
    for d in [p, *p.parents]:
        cfg = d / ".git" / "config"
        if cfg.is_file():
            try:
                return ROOT_REMOTE_MARKER in cfg.read_text().lower()
            except OSError:
                return True                     # unreadable: fail closed
    return False


def _symlinks_on_path(p: Path):
    """Existing components of `p` (and ancestors) that are symlinks, with their targets."""
    out = []
    cur = Path(p.anchor)
    for part in p.parts[1:]:
        cur = cur / part
        if cur.is_symlink():
            out.append((cur, cur.resolve()))
        if not cur.exists():
            break
    return out


def _abort(msg, path):
    text = f"[ROOT-GUARD] ABORT: {msg}\n[ROOT-GUARD] attempted output path: {path}"
    print(text, file=sys.stderr)
    raise RootWriteError(text)


def check_output_path(path, strict=False):
    """Return a list of violations (empty list = safe). Never creates anything."""
    lit = Path(path).expanduser().absolute()
    res = lit.resolve()
    v = []
    roots = [r.expanduser().absolute() for r in root_candidates()]
    roots += [r.resolve() for r in roots]
    for r in roots:
        if _inside(lit, r) or _inside(res, r):
            v.append(f"path is inside ROOT ({r})")
    for link, target in _symlinks_on_path(lit):
        if any(_inside(target, r) for r in roots) or _git_toplevel_is_root(target):
            v.append(f"symlink {link} -> {target} points into ROOT")
    if _git_toplevel_is_root(res) or _git_toplevel_is_root(lit):
        v.append("path is inside a git checkout of the ROOT repository (PINN-replication)")
    bad = FORBIDDEN_OUTPUT_DIRNAMES.intersection(lit.parts) | FORBIDDEN_OUTPUT_DIRNAMES.intersection(res.parts)
    if bad:
        v.append(f"path contains ROOT-style result directory {sorted(bad)}")
    b2 = RESULTS_B2.resolve()
    allowed = _inside(res, b2)
    if not strict:
        allowed = allowed or _inside(res, Path(tempfile.gettempdir()).resolve())
    if not allowed:
        v.append(f"path is outside {'results_batch2/' if strict else 'results_batch2/ and the temp dir'}")
    return v


def assert_safe_output(path, strict=False, purpose="output"):
    v = check_output_path(path, strict=strict)
    if v:
        _abort(f"{purpose}: " + "; ".join(v), Path(path).expanduser().absolute())
    return Path(path).expanduser().absolute()
