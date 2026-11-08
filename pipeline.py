"""SIP Discovery Engine - pipeline CLI.

  python pipeline.py collect [--only playstore,appstore,reddit,tradingqna,youtube,manual]
  python pipeline.py clean
  python pipeline.py extract [--limit 200]      # LLM structured coding (cached, resumable)
  python pipeline.py themes                     # emergent clusters of unmet needs
  python pipeline.py insights                   # grounded findings per research question
  python pipeline.py all                        # collect -> clean -> extract -> themes -> insights
  python pipeline.py propose-taxonomy           # LLM open coding -> config/taxonomy_proposed.yaml
  python pipeline.py golden-sample [--n 20]     # sample for hand labelling
  python pipeline.py eval [--fresh] [--compare-model MODEL]
  python pipeline.py theme-audit [--per-theme 10]
  python pipeline.py demo                       # synthetic data to preview the app
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from engine import settings

COLLECTORS = ["playstore", "appstore", "reddit", "tradingqna", "youtube", "manual"]


def _clear_demo():
    if settings.is_demo():
        settings.DEMO_FLAG.unlink()
        for f in (settings.INSIGHTS_FILE, settings.CLUSTERS_FILE, settings.EVALS / "report.json"):
            if f.exists() and "demo" in f.read_text(encoding="utf-8").lower():
                f.unlink()
        print("  removed DEMO data flag")


def cmd_collect(only: str | None):
    import importlib

    from engine.collect.common import save_raw

    cfg = settings.sources_cfg()
    names = only.split(",") if only else COLLECTORS
    for name in names:
        print(f"\n== {name}")
        t = time.time()
        mod = importlib.import_module(f"engine.collect.{name}")
        items = mod.collect(cfg)
        if items:
            save_raw(name, items)
        print(f"  ({time.time() - t:.0f}s)")


def main(argv=None):
    p = argparse.ArgumentParser(description="SIP discovery engine pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--only")
    sub.add_parser("clean")
    e = sub.add_parser("extract"); e.add_argument("--limit", type=int)
    t = sub.add_parser("themes"); t.add_argument("--k", type=int)
    sub.add_parser("insights")
    a = sub.add_parser("all"); a.add_argument("--only"); a.add_argument("--limit", type=int)
    sub.add_parser("propose-taxonomy")
    g = sub.add_parser("golden-sample"); g.add_argument("--n", type=int, default=20)
    v = sub.add_parser("eval"); v.add_argument("--fresh", action="store_true"); v.add_argument("--compare-model")
    v.add_argument("--from-cache", help="score an older prompt version's cache file"); v.add_argument("--label", default="v1")
    ta = sub.add_parser("theme-audit"); ta.add_argument("--per-theme", type=int, default=10)
    sub.add_parser("demo")
    args = p.parse_args(argv)

    if args.cmd in ("collect", "all"):
        cmd_collect(args.only)
    if args.cmd in ("clean", "all"):
        print("\n== clean"); from engine import clean; clean.run()
    if args.cmd in ("extract", "all"):
        print("\n== extract"); _clear_demo(); from engine import extract; extract.run(limit=args.limit)
    if args.cmd in ("themes", "all"):
        print("\n== themes"); from engine import themes; themes.clusters(getattr(args, "k", None))
    if args.cmd in ("insights", "all"):
        print("\n== insights"); from engine import insights; insights.run()
    if args.cmd == "propose-taxonomy":
        from engine import themes; themes.propose_taxonomy()
    if args.cmd == "golden-sample":
        from engine import evals; evals.golden_sample(args.n)
    if args.cmd == "eval":
        from engine import evals
        if args.from_cache:
            evals.rescore_cache(args.from_cache, args.label)
        else:
            evals.run(fresh=args.fresh, compare_model=args.compare_model)
    if args.cmd == "theme-audit":
        from engine import evals; evals.theme_audit(args.per_theme)
    if args.cmd == "demo":
        from engine import demo; demo.build()
    if args.cmd == "all":
        print("\nDone. Run:  streamlit run app.py")


if __name__ == "__main__":
    sys.exit(main())
