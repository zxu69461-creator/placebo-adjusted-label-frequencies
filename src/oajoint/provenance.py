"""Provenance layer: the ONLY sanctioned write path into the project DuckDB.

Every analysis script must run inside `run_context()` and emit its results through
`archive_table()` / `archive_parquet()` / `archive_figure()`.  Nothing else may write
to db/oajoint.duckdb.  This is what mechanically enforces the project rule that no
loose intermediate TSV/RDS files are ever produced.

Concurrency contract (the thing that will otherwise bite you):
    DuckDB is single-writer.  A long-running job must NEVER hold the write lock.
    So every function here opens a connection, writes, and closes immediately.
    Long jobs write Parquet into `db/staging/<run_id>/` (no lock at all) and then call
    `ingest_staging()`, which takes the lock for seconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import resource
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb

# --------------------------------------------------------------------------- paths

ROOT = Path(os.environ.get("OAJOINT_ROOT", Path(__file__).resolve().parents[2]))
DB_PATH = ROOT / "db" / "oajoint.duckdb"
PARQUET_ROOT = ROOT / "db" / "parquet"
STAGING_ROOT = ROOT / "db" / "staging"
FIGURE_ROOT = ROOT / "figures"
SCHEMA_PATH = ROOT / "db" / "schema.sql"

_LOCK_RETRIES = 60
_LOCK_SLEEP = 2.0

# --------------------------------------------------------------------------- utils

_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford, ULID alphabet


def ulid() -> str:
    """Lexicographically sortable 26-char ID. No external dependency."""
    ms = int(time.time() * 1000)
    ts = "".join(_B32[(ms >> (45 - 5 * i)) & 31] for i in range(10))
    rand = "".join(random.choice(_B32) for _ in range(16))
    return ts + rand


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Order-stable digest of a directory (used for partitioned Parquet stores)."""
    root = Path(path)
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        h.update(str(p.relative_to(root)).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def md5_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return ru / 1e9 if sys.platform == "darwin" else ru / 1e6


def _cmd(args: Sequence[str]) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _git_commit() -> str | None:
    return _cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"])


def _gpu_model() -> str | None:
    return _cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])


def _env_lock_sha256(env_name: str | None) -> str | None:
    """Hash the lockfile pinning the environment, so a result is tied to its stack."""
    if not env_name:
        return None
    for cand in (
        ROOT / "config" / "env" / f"{env_name}.lock",
        ROOT / "config" / "env" / f"{env_name}.yml",
        ROOT / "config" / "env" / "renv.lock",
    ):
        if cand.exists():
            return sha256_file(cand)
    return None


def _software_versions(extra: Mapping[str, str] | None = None) -> str:
    vers: dict[str, str] = {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "platform": platform.platform(),
    }
    for mod in ("numpy", "pandas", "pyarrow", "scipy", "scanpy", "anndata", "torch", "statsmodels"):
        try:
            from importlib.metadata import version

            vers[mod] = version(mod)
        except Exception:
            pass
    if extra:
        vers.update({k: str(v) for k, v in extra.items()})
    return json.dumps(vers, sort_keys=True)


# ----------------------------------------------------------------- connection

@contextmanager
def connect(read_only: bool = False):
    """Short-lived connection. Retries while another process holds the write lock."""
    last: Exception | None = None
    for attempt in range(_LOCK_RETRIES):
        try:
            con = duckdb.connect(str(DB_PATH), read_only=read_only)
            break
        except duckdb.Error as exc:  # lock contention
            last = exc
            if "lock" not in str(exc).lower():
                raise
            time.sleep(_LOCK_SLEEP * (1 + 0.25 * attempt))
    else:
        raise RuntimeError(f"could not acquire DuckDB after {_LOCK_RETRIES} tries: {last}")
    try:
        yield con
    finally:
        con.close()


def init_db(force: bool = False) -> None:
    """Apply db/schema.sql. Idempotent."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if force and DB_PATH.exists():
        DB_PATH.unlink()
    with connect() as con:
        con.execute(SCHEMA_PATH.read_text())


# --------------------------------------------------------------------- run context

class Run:
    """Handle for one script execution. Obtained from `run_context()`."""

    def __init__(self, run_id: str, script_path: Path, env_name: str | None):
        self.run_id = run_id
        self.script_path = script_path
        self.env_name = env_name
        self.started_at = _now()
        self._t0 = time.process_time()

    # ------------------------------------------------------------ input assets
    def register_input(
        self,
        asset_id: str,
        *,
        asset_kind: str,
        local_path: str | Path | None = None,
        accession: str | None = None,
        url: str | None = None,
        source_registry: str | None = None,
        genome_build: str = "unknown",
        build_evidence: str | None = None,
        coordinate_system: str | None = None,
        licence: str | None = None,
        licence_url: str | None = None,
        published_md5: str | None = None,
        status: str = "present",
        notes: str | None = None,
    ) -> str:
        """Register an input. Hashes the file and checks any published md5.

        `genome_build` must be stated explicitly and `build_evidence` must say HOW it
        was determined -- the project rule is never to assume a build.
        """
        size = sha = None
        md5_match = None
        if local_path is not None and status == "present":
            p = Path(local_path)
            if not p.exists():
                raise FileNotFoundError(p)
            size = p.stat().st_size
            sha = sha256_tree(p) if p.is_dir() else sha256_file(p)
            if published_md5 and p.is_file():
                md5_match = md5_file(p).lower() == published_md5.lower()
                if not md5_match:
                    raise ValueError(
                        f"md5 mismatch for {asset_id}: published={published_md5} "
                        f"computed={md5_file(p)}"
                    )
        if genome_build == "unknown" and status == "present":
            print(f"  [warn] {asset_id}: genome_build still 'unknown' -- resolve before analysis")

        vals = [asset_kind, source_registry, accession, url,
                str(local_path) if local_path else None, size, sha, published_md5,
                md5_match, _now(), genome_build, build_evidence, coordinate_system,
                licence, licence_url, status, self.run_id, notes]
        with connect() as con:
            # A true upsert, not INSERT OR REPLACE. DuckDB refuses to replace a row
            # that another table's foreign key references (ref_phenotype.asset_id
            # points here), so re-registering an asset must UPDATE in place.
            exists = con.execute(
                "SELECT 1 FROM prov_input_asset WHERE asset_id = ?", [asset_id]
            ).fetchone()
            if exists:
                con.execute(
                    """UPDATE prov_input_asset SET
                         asset_kind=?, source_registry=?, accession=?, url=?,
                         local_path=?, file_bytes=?, file_sha256=?, published_md5=?,
                         md5_match=?, downloaded_at=?, genome_build=?, build_evidence=?,
                         coordinate_system=?, licence=?, licence_url=?, status=?,
                         registered_by=?, notes=?
                       WHERE asset_id=?""", vals + [asset_id])
            else:
                con.execute(
                    """INSERT INTO prov_input_asset
                       (asset_kind, source_registry, accession, url, local_path,
                        file_bytes, file_sha256, published_md5, md5_match, downloaded_at,
                        genome_build, build_evidence, coordinate_system, licence,
                        licence_url, status, registered_by, notes, asset_id)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    vals + [asset_id])
        return asset_id

    # ---------------------------------------------------------------- outputs
    def _register_output(
        self, con, object_name: str, object_kind: str, storage: str,
        relpath: str | None, n_rows: int | None, n_cols: int | None,
        file_sha256: str | None, inputs: Iterable[str], upstream: Iterable[str],
    ) -> str:
        out_id = ulid()
        con.execute(
            """INSERT INTO prov_output_asset
               (out_id, run_id, object_name, object_kind, storage, relpath,
                n_rows, n_cols, file_sha256, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [out_id, self.run_id, object_name, object_kind, storage, relpath,
             n_rows, n_cols, file_sha256, _now()],
        )
        rows = [(out_id, a, "input_asset", None) for a in inputs]
        rows += [(out_id, u, "upstream_output", None) for u in upstream]
        if rows:
            con.executemany(
                "INSERT INTO prov_lineage (out_id, asset_id, edge_kind, role) VALUES (?,?,?,?)",
                rows,
            )
        return out_id

    def archive_table(
        self, table: str, df, *, inputs: Iterable[str] = (), upstream: Iterable[str] = (),
        mode: str = "append",
    ) -> str:
        """Write a DataFrame into a native DuckDB result table and register it.

        `mode`: 'append' (default), 'replace_run' (delete this run's rows first),
        'replace_all' (truncate -- only for fully regenerated reference tables), or
        'upsert' (INSERT OR REPLACE -- for keyed caches that are re-fetched).
        The table must already exist in schema.sql -- ad-hoc tables are refused, which
        keeps the schema the single source of truth.
        """
        with connect() as con:
            exists = con.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [table]
            ).fetchone()[0]
            if not exists:
                raise KeyError(f"table '{table}' is not in schema.sql -- add it there first")
            # accept a DataFrame/Arrow table, or a plain list of dicts (R handoffs)
            if mode == "replace_all":
                con.execute(f"DELETE FROM {table}")
            if isinstance(df, list):
                if not df:
                    raise ValueError(f"archive_table({table}): empty payload")
                cols = list(df[0].keys())
                if mode == "replace_run" and "run_id" in cols:
                    con.execute(f"DELETE FROM {table} WHERE run_id = ?", [self.run_id])
                collist = ", ".join(f'"{c}"' for c in cols)
                marks = ", ".join("?" for _ in cols)
                verb = "INSERT OR REPLACE INTO" if mode == "upsert" else "INSERT INTO"
                con.executemany(
                    f"{verb} {table} ({collist}) VALUES ({marks})",
                    [[r.get(c) for c in cols] for r in df],
                )
            else:
                con.register("_incoming", df)
                cols = [r[0] for r in con.execute("DESCRIBE _incoming").fetchall()]
                if mode == "replace_run" and "run_id" in cols:
                    con.execute(f"DELETE FROM {table} WHERE run_id = ?", [self.run_id])
                collist = ", ".join(f'"{c}"' for c in cols)
                verb = "INSERT OR REPLACE INTO" if mode == "upsert" else "INSERT INTO"
                con.execute(f"{verb} {table} ({collist}) SELECT {collist} FROM _incoming")
                con.unregister("_incoming")
            n = len(df)
            out_id = self._register_output(
                con, table, "table", "duckdb", None, n, len(cols), None, inputs, upstream
            )
        print(f"  [archived] {table}: +{n} rows  (out_id={out_id})")
        return out_id

    def archive_parquet(
        self, domain: str, src: str | Path, *,
        inputs: Iterable[str] = (), upstream: Iterable[str] = (),
    ) -> str:
        """Move a Parquet file/dir into db/parquet/<domain>/ and register it."""
        with connect() as con:
            row = con.execute(
                "SELECT relpath, view_name FROM ref_parquet_domain WHERE domain = ?", [domain]
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown parquet domain '{domain}' -- add it to ref_parquet_domain")
        relpath, _ = row
        dest = ROOT / relpath
        src = Path(src)
        if src.resolve() != dest.resolve():
            dest.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest / src.name)
        if not dest.exists():
            raise FileNotFoundError(f"parquet destination missing after copy: {dest}")
        sha = sha256_tree(dest)
        ensure_views()
        with connect() as con:
            try:
                n = con.execute(
                    f"SELECT count(*) FROM read_parquet('{dest}/**/*.parquet', hive_partitioning=true)"
                ).fetchone()[0]
            except Exception:
                n = None
            out_id = self._register_output(
                con, domain, "parquet", "parquet", relpath, n, None, sha, inputs, upstream
            )
        print(f"  [archived] parquet {domain}: {n} rows  (out_id={out_id})")
        return out_id

    def archive_document(
        self, object_name: str, path: str | Path, *, kind: str = "document",
        inputs: Iterable[str] = (), upstream: Iterable[str] = (),
    ) -> str:
        """Register a deliverable file (manuscript, supplementary workbook) built from the database."""
        p = Path(path)
        rel = str(p.resolve().relative_to(ROOT))
        with connect() as con:
            out_id = self._register_output(con, object_name, kind, "file", rel, None, None,
                                           sha256_file(p), inputs, upstream)
        print(f"  [archived] {kind} {object_name}: {rel}  (out_id={out_id})")
        return out_id

    def archive_figure_qa(self, fig_id: str, saved) -> None:
        """Record the figkit QA of one saved figure (a figkit.Saved) in fig_qa, replacing earlier rows."""
        import figkit
        pdf = saved.pdf
        rel = str(pdf.relative_to(ROOT)) if pdf.is_absolute() and ROOT in pdf.parents else str(pdf)
        with connect() as con:
            con.execute("DELETE FROM fig_qa WHERE fig_id = ?", [fig_id])
            for c in saved.report.checks:
                con.execute("INSERT INTO fig_qa VALUES (?,?,?,?,?,?,?,?)",
                            [fig_id, c.name, c.status, c.detail.lstrip("- "), rel, saved.paths[pdf],
                             figkit.version(), _now()])

    def archive_figure_abbrev(self, fig_id: str, abbreviations: dict) -> None:
        """Record the label abbreviations of one figure (figkit Saved.abbreviations) for its legend."""
        with connect() as con:
            con.execute("DELETE FROM fig_abbrev WHERE fig_id = ?", [fig_id])
            for short, full in sorted(abbreviations.items()):
                con.execute("INSERT INTO fig_abbrev VALUES (?,?,?,?)", [fig_id, short, full, _now()])

    def retire(self, relpaths: Iterable[str], reason: str) -> None:
        """Withdraw outputs from the live archive (file deleted, history kept, fig_panel row removed)."""
        with connect() as con:
            for rel in relpaths:
                con.execute("INSERT INTO prov_retired VALUES (?,?,?,?)", [rel, _now(), self.run_id, reason])
                con.execute("DELETE FROM fig_panel WHERE rendered_relpath = ?", [rel])
                (ROOT / rel).unlink(missing_ok=True)
                print(f"  [retired] {rel}")

    def archive_figure(
        self, fig_id: str, panel_id: str, path: str | Path, *,
        panel_title: str | None = None, source_objects: Sequence[str] = (),
        needs_schematic: bool = False, imagegen_prompt: str | None = None,
        inputs: Iterable[str] = (), upstream: Iterable[str] = (),
    ) -> str:
        p = Path(path)
        sha = sha256_file(p) if p.exists() else None
        rel = str(p.relative_to(ROOT)) if p.is_absolute() and ROOT in p.parents else str(p)
        with connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO fig_panel
                   (fig_id, panel_id, panel_title, source_objects, script_path,
                    needs_handmade_schematic, imagegen_prompt_text,
                    rendered_relpath, rendered_sha256, rendered_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [fig_id, panel_id, panel_title, list(source_objects), str(self.script_path),
                 needs_schematic, imagegen_prompt, rel, sha, _now()],
            )
            return self._register_output(
                con, f"{fig_id}:{panel_id}", "figure", p.suffix.lstrip(".") or "png",
                rel, None, None, sha, inputs, upstream
            )

    # ---------------------------------------------------------------- decisions
    def register_gate(
        self, gate_name: str, rule_text: str, *, gate_family: str = "qc",
        metric_name: str | None = None, threshold: float | None = None,
        comparator: str | None = None, notes: str | None = None,
    ) -> str:
        """Write a gate BEFORE the data are seen. This is the pre-registration.

        IDEMPOTENT on gate_name, and deliberately so. A script that registers its own
        gate then evaluates it is the normal pattern here, and such scripts get re-run --
        so a blind INSERT silently accumulates duplicate ledger rows for one gate, which
        destroys the one property the table exists to have: one row, one rule, one
        outcome. (Observed: five rows for qc_locus_heterogeneity, four of them pending.)

        Re-registering an identical rule is a no-op. Re-registering a DIFFERENT rule is
        an amendment, and is allowed only while the gate is still unevaluated: once an
        outcome exists, changing the rule is rewriting the prediction after seeing the
        answer, which is exactly the failure pre-registration prevents, so it raises.
        Permitted amendments are stamped into notes so they stay visible.
        """
        fields = (gate_family, rule_text, metric_name, threshold, comparator)
        with connect() as con:
            row = con.execute(
                """SELECT decision_id, gate_family, rule_text, metric_name, threshold,
                          comparator, evaluated_at, notes FROM prov_decision
                   WHERE gate_name = ? ORDER BY registered_at LIMIT 1""", [gate_name]
            ).fetchone()
            if row is not None:
                did, *existing, evaluated_at, old_notes = row
                if tuple(existing) == fields:
                    return did                      # identical re-registration: no-op
                if evaluated_at is not None:
                    raise ValueError(
                        f"gate '{gate_name}' already has an outcome recorded at "
                        f"{evaluated_at}; its rule cannot be changed now. Register a "
                        f"NEW, superseding gate instead (as G5a supersedes G5)."
                    )
                stamp = f"[amended {_now()} by {self.run_id}]"
                con.execute(
                    """UPDATE prov_decision SET gate_family=?, rule_text=?, metric_name=?,
                           threshold=?, comparator=?, notes=?
                       WHERE decision_id=?""",
                    [*fields, " ".join(x for x in (old_notes, notes, stamp) if x), did])
                return did
            did = ulid()
            con.execute(
                """INSERT INTO prov_decision
                   (decision_id, gate_name, gate_family, rule_text, metric_name,
                    threshold, comparator, registered_at, registered_by, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [did, gate_name, gate_family, rule_text, metric_name,
                 threshold, comparator, _now(), self.run_id, notes],
            )
        return did

    def evaluate_gate(
        self, gate_name: str, metric_value: float | None, passed: bool | None,
        *, detail: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the outcome. Refuses if the gate was never registered.

        `passed=None` means UNEVALUABLE -- the test could not be run at all (e.g.
        its input data turned out to be inaccessible). That is deliberately
        distinct from `passed=False`, which means the test ran and the data failed
        it. Conflating the two would misreport the evidence.
        """
        with connect() as con:
            row = con.execute(
                """SELECT decision_id, registered_at FROM prov_decision
                   WHERE gate_name = ? ORDER BY registered_at LIMIT 1""", [gate_name]
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"gate '{gate_name}' was never registered -- pre-registration violated"
                )
            did, registered_at = row
            now = _now()
            if registered_at >= now:
                raise ValueError(f"gate '{gate_name}': registered_at >= evaluated_at")
            con.execute(
                """UPDATE prov_decision
                   SET metric_value=?, passed=?, metric_detail=?, evaluated_at=?, evaluated_by=?
                   WHERE decision_id=?""",
                [metric_value, passed, json.dumps(detail or {}, default=str), now,
                 self.run_id, did],
            )
        print(f"  [gate] {gate_name}: value={metric_value} passed={passed}")

    # ------------------------------------------------------------------ staging
    def staging_dir(self) -> Path:
        d = STAGING_ROOT / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ingest_staging(self, domain: str, *, inputs: Iterable[str] = (),
                       upstream: Iterable[str] = ()) -> str:
        """Promote db/staging/<run_id>/<domain>/ into the Parquet store, then clean up."""
        src = self.staging_dir() / domain
        if not src.exists():
            raise FileNotFoundError(src)
        out_id = self.archive_parquet(domain, src, inputs=inputs, upstream=upstream)
        shutil.rmtree(src)
        if not any(self.staging_dir().iterdir()):
            shutil.rmtree(self.staging_dir())
        return out_id


@contextmanager
def run_context(script_path: str | Path, *, env_name: str | None = None,
                software: Mapping[str, str] | None = None, notes: str | None = None):
    """Wrap a script. Opens the prov_script_run row, closes it with exit status."""
    script_path = Path(script_path).resolve()
    run = Run(ulid(), script_path, env_name)
    with connect() as con:
        con.execute(
            """INSERT INTO prov_script_run
               (run_id, script_path, script_sha256, git_commit, cmdline, env_name,
                env_lock_sha256, software_versions, host, gpu_model, started_at, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [run.run_id, str(script_path),
             sha256_file(script_path) if script_path.exists() else "missing",
             _git_commit(), " ".join(sys.argv), env_name, _env_lock_sha256(env_name),
             _software_versions(software), socket.gethostname(), _gpu_model(),
             run.started_at, notes],
        )
    print(f"[run {run.run_id}] {script_path.name}")
    status = 0
    try:
        yield run
    except BaseException:
        status = 1
        raise
    finally:
        with connect() as con:
            con.execute(
                """UPDATE prov_script_run
                   SET finished_at=?, exit_status=?, cpu_seconds=?, peak_rss_gb=?
                   WHERE run_id=?""",
                [_now(), status, time.process_time() - run._t0, _peak_rss_gb(), run.run_id],
            )
        print(f"[run {run.run_id}] exit={status}")


# ------------------------------------------------------------------ views + audit

def ensure_views() -> list[str]:
    """Create a view per Parquet domain that actually has files.

    A view over an empty glob errors in DuckDB, so views are created lazily rather
    than in schema.sql. Absolute paths, so the view works from any cwd.
    """
    made: list[str] = []
    with connect() as con:
        domains = con.execute(
            "SELECT domain, relpath, view_name FROM ref_parquet_domain"
        ).fetchall()
        for _, relpath, view_name in domains:
            d = ROOT / relpath
            if d.exists() and any(d.rglob("*.parquet")):
                con.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM "
                    f"read_parquet('{d}/**/*.parquet', hive_partitioning=true)"
                )
                made.append(view_name)
    return made


def verify(strict: bool = True) -> int:
    """Re-hash every registered output; report drift, gaps, build-mixing, prereg breaks.

    Returns a nonzero count if anything is wrong (the CLI exits with it).
    """
    problems = 0
    with connect(read_only=True) as con:
        rows = con.execute(
            """SELECT object_name, storage, relpath, file_sha256 FROM (
                 -- a file has one content at a time: its latest registration is authoritative, whatever the
                 -- object name it was registered under (renamed objects would otherwise report stale drift)
                 SELECT *, row_number() OVER (PARTITION BY relpath ORDER BY created_at DESC) AS rn
                 FROM prov_output_asset WHERE relpath IS NOT NULL
               ) o WHERE rn = 1 AND NOT EXISTS (
                 SELECT 1 FROM prov_retired r WHERE r.relpath = o.relpath AND r.retired_at > o.created_at)"""
        ).fetchall()
        for name, storage, relpath, sha in rows:
            p = ROOT / relpath
            if not p.exists():
                print(f"MISSING   {name}: {relpath}")
                problems += 1
                continue
            if sha is None:
                continue
            cur = sha256_tree(p) if p.is_dir() else sha256_file(p)
            if cur != sha:
                print(f"DRIFT     {name}: {relpath}")
                problems += 1

        for label, macro in (("GAP", "manifest_gaps()"),
                             ("BUILD-MIX", "manifest_builds()"),
                             ("PREREG", "prereg_violations()")):
            for r in con.execute(f"SELECT * FROM {macro}").fetchall():
                print(f"{label:9s} {r}")
                problems += 1

        unresolved = con.execute(
            """SELECT asset_id FROM prov_input_asset
               WHERE status='present' AND genome_build='unknown'"""
        ).fetchall()
        for (a,) in unresolved:
            print(f"BUILD-UNK {a}: genome_build unresolved")
            problems += 1

    print(f"\n{'FAIL' if problems else 'OK'}: {problems} problem(s)")
    return problems if strict else 0


if __name__ == "__main__":
    if "--init" in sys.argv:
        init_db(force="--force" in sys.argv)
        print(f"initialised {DB_PATH}")
    elif "--verify" in sys.argv:
        sys.exit(1 if verify() else 0)
    elif "--views" in sys.argv:
        print("views:", ensure_views())
    else:
        print(__doc__)
        print("usage: python -m oajoint.provenance [--init [--force] | --verify | --views]")
