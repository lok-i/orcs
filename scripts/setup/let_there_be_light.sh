#!/usr/bin/env bash
# let_there_be_light.sh — configure both .claude and .vscode for this env.
#
# Run it INSIDE the active project venv (conda/uv), from the repo root, e.g.:
#     conda activate mjlab && ./scripts/let_there_be_light.sh
#
# Idempotent — safe to re-run after env / dependency changes.
#
# Philosophy: track the GENERATOR (this script), never the generated config.
# Re-run anytime, on any machine, to materialize the setup. Nothing under
# .claude/ or .vscode/ that this script writes is tracked — they're gitignored
# and regenerated from the constants below + autodetected dependency roots
# (via importlib.find_spec, no module execution → no pxr/kit needed):
#
#   .claude/
#     settings.json         agnostic perms + durable bash baseline
#     settings.local.json   abs-path Read() rules for ext deps. Claude also
#                           appends newly-approved perms here at runtime — the
#                           script MERGES (never clobbers) so those survive.
#   .vscode/
#     settings.json         agnostic editor prefs + python.analysis.extraPaths
#                           (in-repo deps relative, external deps abs).
#                           Fully regenerated each run — edit the script, not it.
#
#   .gitignore              ensures /data, /dependencies and the above config
#                           files are ignored — no manual upkeep.
#
# Import names to expose live in CANDIDATES — extend as the project grows.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"

# import names whose source trees Claude should read / VSCode should jump to
CANDIDATES="mjlab rsl_rl"

# machine-agnostic Claude rules (tracked settings.json):
# relative dep/data reads + a minimal, durable dev baseline (no abs paths).
AGNOSTIC_RULES='[
  "Read(configs/**)",
  "Read(//tmp/**)",
  "Read(//etc/passwd)",
  "Read(//proc/cpuinfo)",
  "Read(//proc/meminfo)",
  "Bash(conda run *)",
  "Bash(conda env *)",
  "Bash(conda list *)",
  "Bash(conda info *)",
  "Bash(uv run *)",
  "Bash(uv venv *)",
  "Bash(pip show *)",
  "Bash(pip list *)",
  "Bash(uv pip show *)",
  "Bash(uv pip list *)",
  "Bash(python -m py_compile *)",
  "Bash(python -c *)",
  "Bash(python -m pytest *)",
  "Bash(which *)",
  "Bash(ls *)",
  "Bash(wc *)",
  "Bash(du *)",
  "Bash(df *)",
  "Bash(env)",
  "Bash(nvidia-smi *)",
  "Bash(git *)"
]'

# machine-agnostic VSCode editor prefs (tracked settings.shared.json):
# keep code (group 1) and the Claude panel (group 2) separated — new editors
# open beside, existing ones are revealed instead of duplicated, and terminal
# editors auto-lock so spawned tabs don't scatter. (Lock the Claude group once
# via right-click → "Lock Group" to pin it as group 2.)
AGNOSTIC_VSCODE='{
  "workbench.editor.openSideBySideDirection": "right",
  "workbench.editor.revealIfOpen": true,
  "workbench.editor.autoLockGroups": {
    "terminalEditor": true
  }
}'

# .gitignore lines this script guarantees exist
GITIGNORE_LINES='/data/
/dependencies/
.claude/settings.json
.claude/settings.local.json
.vscode/settings.json'

command -v "$PY" >/dev/null 2>&1 || { echo "✗ no '$PY' on PATH — activate the venv first"; exit 1; }
"$PY" - "$REPO_ROOT" "$AGNOSTIC_RULES" "$AGNOSTIC_VSCODE" "$GITIGNORE_LINES" $CANDIDATES <<'PY'
import importlib.util as iu, json, os, subprocess, sys, sysconfig
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
agnostic_rules = json.loads(sys.argv[2])
agnostic_vscode = json.loads(sys.argv[3])
gitignore_lines = [l for l in sys.argv[4].splitlines() if l.strip()]
cands = sys.argv[5:]
claude = repo / ".claude"; claude.mkdir(exist_ok=True)
vscode = repo / ".vscode"; vscode.mkdir(exist_ok=True)

# site-packages dirs — Pylance resolves these by default, so we don't add them
# to extraPaths (Claude may still read them via the abs Read() rules).
site_dirs = {Path(p).resolve() for p in (
    sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")) if p}

def in_venv():
    if os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"):
        return True
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)
env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV", "?")
print("⚠  not inside a venv — resolving against base interpreter"
      if not in_venv() else f"· resolving deps in env: {env}")

def resolve(name):
    """import name -> (pkg_dir, read_tree). pkg_dir = dir holding the package
    (its parent is the import root for extraPaths); read_tree = broad tree for
    Claude reads (git root > src/source layout root > pkg_dir). No execution."""
    try:
        s = iu.find_spec(name)
    except Exception:
        s = None
    if not s:
        return None
    origin = s.origin if s.origin and s.origin not in ("namespace", "built-in") else None
    p = Path(origin) if origin else (
        Path(next(iter(s.submodule_search_locations))) if s.submodule_search_locations else None)
    if not p:
        return None
    pkg_dir = (p.parent if p.name == "__init__.py" else p).resolve()
    # broad tree for Claude: prefer git checkout root
    read_tree = pkg_dir
    try:
        top = subprocess.check_output(
            ["git", "-C", str(pkg_dir), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL).decode().strip()
        read_tree = Path(top)
    except Exception:
        for marker in ("source", "src"):
            if marker in pkg_dir.parts:
                i = len(pkg_dir.parts) - 1 - pkg_dir.parts[::-1].index(marker)
                read_tree = Path(*pkg_dir.parts[:i]); break
    return pkg_dir, read_tree.resolve()

def is_repo_local(path):
    return path == repo or repo in path.parents
def is_site(path):
    return any(sp == path or sp in path.parents for sp in site_dirs)

resolved, seen = [], set()
for name in cands:
    r = resolve(name)
    if not r:
        print(f"    ? {name:16} (not importable — skipped)"); continue
    pkg_dir, read_tree = r
    if str(pkg_dir).startswith("/tmp/") or pkg_dir in seen:
        continue
    seen.add(pkg_dir)
    resolved.append((name, pkg_dir, read_tree))

# ---- merge helpers ----
def load(path):
    return json.loads(path.read_text()) if path.exists() else {}

def merge_allow(path, rules):
    """ensure a permissions.allow list contains rules; return newly-added."""
    s = load(path)
    allow = s.setdefault("permissions", {}).setdefault("allow", [])
    added = [x for x in rules if x not in allow]
    allow.extend(added)
    path.write_text(json.dumps(s, indent=2) + "\n")
    return added

# ======================= .claude =======================
# external (non-repo) read trees -> abs Read() in local; in-repo covered by cwd.
claude_roots = []
for name, _, read_tree in resolved:
    if is_repo_local(read_tree):
        continue
    if any(read_tree != o and o in read_tree.parents for _, o in claude_roots):
        continue  # nested under an already-kept tree
    claude_roots = [(n, o) for n, o in claude_roots if read_tree not in o.parents]
    if read_tree not in [o for _, o in claude_roots]:
        claude_roots.append((name, read_tree))

agn_added = merge_allow(claude / "settings.json", agnostic_rules)
# NB: a single leading '/' anchors to the project root (gitignore-style); an
# absolute filesystem path needs '//' or it never matches → endless re-prompts.
abs_reads = [f"Read(/{r}/**)" for _, r in claude_roots]
loc_added = merge_allow(claude / "settings.local.json", abs_reads)

# ======================= .vscode =======================
# single generated settings.json (nothing tracked): agnostic editor prefs +
# python.analysis.extraPaths — in-repo deps relative, external deps abs.
extra_paths = []

def import_root_of(dep_dir):
    """import root for a dep dir, src-layout ('<dep>/src') or flat ('<dep>').
    Returns the dir whose children include a package (has __init__.py)."""
    for cand in (dep_dir / "src", dep_dir):
        if cand.is_dir() and any(c.is_dir() and (c / "__init__.py").exists()
                                 for c in cand.iterdir()):
            return cand
    return None

# in-repo custom deps: scan dependencies/* directly (find_spec misses
# non-editable installs that resolve to site-packages, e.g. rsl_rl).
deps_dir = repo / "dependencies"
if deps_dir.is_dir():
    for dep in sorted(deps_dir.iterdir()):
        if not dep.is_dir():
            continue
        root = import_root_of(dep)
        if root:
            rel = os.path.relpath(root, repo)
            if rel not in extra_paths: extra_paths.append(rel)

# external custom deps (outside the repo, e.g. mjlab) -> abs extraPaths.
for name, pkg_dir, _ in resolved:
    import_root = str(pkg_dir.parent)
    if is_repo_local(pkg_dir.parent) or is_site(pkg_dir.parent):
        continue  # in-repo handled above; site-packages auto-resolved by Pylance
    if import_root not in extra_paths: extra_paths.append(import_root)

vs = {"//": "GENERATED by let_there_be_light.sh - regenerated each run; edit the script, not this file"}
vs.update(agnostic_vscode)
vs["python.analysis.extraPaths"] = extra_paths
(vscode / "settings.json").write_text(json.dumps(vs, indent=2) + "\n")

# ======================= .gitignore =======================
gi = repo / ".gitignore"
text = gi.read_text() if gi.exists() else ""
existing = set(l.strip() for l in text.splitlines())
new = [l for l in gitignore_lines if l not in existing]
if new:
    if text and not text.endswith("\n"):
        text += "\n"
    text += "# device-specific config (managed by let_there_be_light.sh)\n" + "\n".join(new) + "\n"
    gi.write_text(text)

# ======================= report =======================
print(f"\n.claude/")
print(f"  settings.json       : {len(agnostic_rules)} agnostic rule(s) (+{len(agn_added)} new)")
print(f"  settings.local.json : {len(claude_roots)} external dep read root(s) (+{len(loc_added)} new)")
for n, r in claude_roots:
    print(f"      · {n:10} {r}")
print(f".vscode/")
print(f"  settings.json       : editor prefs + {len(extra_paths)} extraPath(s)")
for p in extra_paths: print(f"      · {p}")
print(f".gitignore            : {len(gitignore_lines)} line(s) ensured (+{len(new)} new)")
print(f"\n→ reload VSCode window; lock the Claude editor group once "
      f"(right-click its tab bar → 'Lock Group') to pin it as group 2.")
PY
