"""Portable figures, tables and a self-contained local HTML report."""

from __future__ import annotations

import csv
import html
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from .metrics import METRICS, aggregate


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["trial_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def short_name(identity: str) -> str:
    parts = Path(identity).parts
    if parts[-1] == "motion.npz":
        return parts[-3] if parts[-2].startswith("sample") else parts[-2]
    return Path(identity).stem


def render(output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    receipt = json.loads((output / "receipt.json").read_text())
    trials = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines() if line.strip()]
    summary = aggregate(receipt, trials)
    rows = sorted(summary["per_motion"], key=lambda r: (-r["failure_rate"], r["motion_index"]))
    names = [short_name(r["motion"]) for r in rows]
    # Expand only ambiguous labels, preserving sample/take identity.
    names = [name if names.count(name) == 1 else row["motion"]
             for name, row in zip(names, rows, strict=True)]
    indices = np.arange(len(rows))
    style = {"figure.facecolor": "#101820", "axes.facecolor": "#101820", "savefig.facecolor": "#101820",
             "text.color": "#e1eaf0", "axes.labelcolor": "#b9cbd7", "xtick.color": "#b9cbd7",
             "ytick.color": "#b9cbd7", "axes.edgecolor": "#40505f", "font.size": 10,
             "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(max(12, min(30, len(rows)*.23)), 5.8), layout="constrained")
        bottom = np.zeros(len(rows))
        for key, color, label in (("failed", "#ed8b73", "Failed"), ("invalid", "#c4a1f0", "Invalid"),
                                  ("censored", "#e1bf72", "Censored"), ("pending", "#586b7a", "Pending")):
            values = [r[key] for r in rows]
            ax.bar(indices, values, bottom=bottom, color=color, label=label, width=.8)
            bottom += values
        ax.set(xticks=indices, xticklabels=names, ylabel="Trials", xlabel="Reference motion · sorted by failure rate",
               ylim=(0, max(1, receipt["trials_per_motion"])*1.08))
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", ncols=4, frameon=False)
        for ext in ("png", "svg"):
            fig.savefig(output / f"failures.{ext}", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, max(4, len(rows)*.20)), layout="constrained")
        heat = np.array([[float("nan") if v is None else v for v in r["bin_failure_rate"]] for r in rows])
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#42505b")
        shown = ax.imshow(heat, aspect="auto", vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
        bins = receipt["bins"]
        ax.set(yticks=indices, yticklabels=names, xticks=list(range(0, bins, max(1, bins//5))) + [bins],
               xticklabels=[f"{b/bins:.0%}" for b in range(0, bins, max(1, bins//5))] + ["Hold"],
               xlabel="Normalized motion time" if receipt["horizon"] == "clip" else "Normalized episode time")
        ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(shown, ax=ax, label="Failures / trials entering the bin")
        for ext in ("png", "svg"):
            fig.savefig(output / f"failure_time.{ext}", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
        for row in rows:
            ax.plot(np.linspace(0, 1, bins+1), row["survival"], color="#8ed7cb", alpha=.15, lw=.8)
        ax.plot(np.linspace(0, 1, bins+1), np.mean([r["survival"] for r in rows], axis=0),
                color="#f1d294", lw=2.5, label="Mean across motions")
        ax.set(xlabel="Normalized progress reached (hold excluded)", ylabel="Fraction of scheduled trials",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.15)
        ax.legend(frameon=False)
        for ext in ("png", "svg"):
            fig.savefig(output / f"progress.{ext}", dpi=150)
        plt.close(fig)
    _csv(output / "per_motion.csv", summary["per_motion"])
    _csv(output / "trials.csv", [{k: v for k, v in t.items() if not k.startswith("bin_")} for t in trials])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    payload = dict(receipt=receipt, summary=summary, metrics=METRICS,
                   labels={str(r["motion_index"]): name for r, name in zip(rows, names, strict=True)})
    template = files("orcs.core.motion_benchmark").joinpath("report.html").read_text()
    document = template.replace("__TITLE__", html.escape(receipt["task"])).replace(
        "__PAYLOAD__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c"))
    (output / "report.html").write_text(document)
    return summary


def report_server(directory: Path, port: int = 0) -> ThreadingHTTPServer:
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def translate_path(self, path):
            resolved = Path(super().translate_path(path)).resolve()
            # Also prevent symlinks from exposing files outside the report.
            return str(resolved if resolved.is_relative_to(directory.resolve()) else directory / "__missing__")

    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(directory)))


def serve(directory: Path, port: int = 0) -> None:
    with report_server(directory, port) as server:
        print(f"\nMotion repertoire report: http://localhost:{server.server_port}/report.html\n"
              "GPU collection has exited. Ctrl-C stops the server; files remain saved.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
