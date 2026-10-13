"""CLI over the archive: `python -m oajoint.manifest [--verify|--gates|--assets]`.

Plan section 12 step 9 requires `--verify` to exit nonzero on any drift, so this is the
single command that answers "is the archive still telling the truth". It does not write.

Every listing here is read from the provenance registry, never from the filesystem, which
is what makes it an audit rather than a directory walk: a result table that exists on disk
but was never registered will NOT appear, and that is the point.
"""
from __future__ import annotations

import argparse
import sys

from . import provenance as pv


def _print(con, title: str, sql: str) -> int:
    df = con.execute(sql).df()
    print(f"\n=== {title} ({len(df)} rows) ===")
    print(df.to_string(index=False) if len(df) else "  (none)")
    return len(df)


def _check_figures() -> int:
    """Plan section 12 step 8: every panel must rebuild from the DATABASE alone.

    Deletes each registered render, re-runs the registering scripts, and re-hashes. A
    script that quietly reads a loose file instead of the archive will still produce a
    figure here, so this check is only meaningful because save_panel writes deterministic
    bytes -- matplotlib stamps a creation date into PDF, and until that was suppressed
    two renders of identical data disagreed, which made this test vacuous for the vector
    deliverable.
    """
    import hashlib
    import subprocess
    import sys

    with pv.connect(read_only=True) as con:
        rows = con.execute(
            """SELECT fig_id, panel_id, rendered_relpath, rendered_sha256, script_path
               FROM fig_panel WHERE rendered_relpath NOT LIKE '%.md'
               ORDER BY fig_id, panel_id, rendered_relpath""").fetchall()
    if not rows:
        print("no registered renders")
        return 0
    for r in rows:
        (pv.ROOT / r[2]).unlink(missing_ok=True)
    scripts = sorted({r[4] for r in rows})
    print(f"deleted {len(rows)} renders; rebuilding via {len(scripts)} script(s)")
    for sc in scripts:
        subprocess.run([sys.executable, sc], capture_output=True)
    ok = bad = 0
    for fig, pan, rel, sha, _ in rows:
        p = pv.ROOT / rel
        if not p.exists():
            print(f"  MISSING  {fig}:{pan} {rel}"); bad += 1; continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h == sha:
            ok += 1
        else:
            print(f"  DRIFT    {fig}:{pan} {rel}"); bad += 1
    print(f"\n{ok} byte-identical, {bad} problem(s)")
    print("OK: figures reproduce from the database" if not bad else "FAIL")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m oajoint.manifest")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash every registered output; nonzero exit on any problem")
    ap.add_argument("--gates", action="store_true", help="pre-registration ledger")
    ap.add_argument("--assets", action="store_true", help="registered input assets")
    ap.add_argument("--figures", action="store_true",
                    help="delete every registered render, rebuild from the DB, re-hash")
    ap.add_argument("--all", action="store_true", help="every listing, then verify")
    a = ap.parse_args(argv)
    if not any((a.verify, a.gates, a.assets, a.figures, a.all)):
        a.all = True
    if a.figures:
        return _check_figures()

    with pv.connect(read_only=True) as con:
        if a.all or not (a.gates or a.assets or a.verify):
            _print(con, "archived objects (latest per object)",
                   """SELECT object_name, object_kind, storage, n_rows,
                             substr(script_path, 1, 40) AS script, created_at
                      FROM v_manifest_latest ORDER BY created_at""")
        if a.all or a.assets:
            _print(con, "input assets",
                   """SELECT asset_id, accession, status, genome_build,
                             coordinate_system, md5_match, licence
                      FROM prov_input_asset ORDER BY asset_id""")
        if a.all or a.gates:
            _print(con, "pre-registration ledger",
                   """SELECT gate_name, gate_family, metric_name, comparator, threshold,
                             round(metric_value, 6) AS value,
                             CASE WHEN evaluated_at IS NULL THEN 'pending'
                                  WHEN passed IS NULL THEN 'UNEVALUABLE'
                                  WHEN passed THEN 'pass' ELSE 'FAIL' END AS outcome
                      FROM prov_decision ORDER BY gate_family, gate_name""")
            for label, sql in [
                ("prereg integrity breaks (registered_at must precede evaluated_at)",
                 """SELECT gate_name, registered_at, evaluated_at FROM prov_decision
                    WHERE evaluated_at IS NOT NULL AND registered_at >= evaluated_at"""),
                ("lineage gaps", "SELECT * FROM manifest_gaps()"),
                ("genome-build mixing", "SELECT * FROM manifest_builds()")]:
                n = _print(con, label, sql)
                if n:
                    print(f"  ^^ {n} problem(s)")

    if a.verify or a.all:
        print("\n=== re-hashing registered outputs ===")
        problems = pv.verify()
        print(f"\n{'OK: archive verified' if problems == 0 else f'{problems} PROBLEM(S)'}")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
