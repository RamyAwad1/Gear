"""
generate_test_data.py
=====================
Produces a `htu_enrollments.csv` + `htu_students.csv` pair you can feed
straight into the `/api/predict/upload/` endpoint.

It re-uses the existing notebook simulator (htu_full_generator_fixed.ipynb)
so the data shape, plan logic, grade scale, and prerequisite handling all
match exactly what the trained ML models saw — only the cutoff semester
and the number of students are exposed as knobs.

Why two CSVs and not one?
-------------------------
The ML pipeline needs *historical* enrollments per student to compute
lag features (cum_gpa, course_lag_1, gpa_trend, etc.). A single semester
of data is not enough to predict the next semester. This script simulates
N students from their admission year through the cutoff you specify, then
writes everything up to and including that cutoff. The endpoint then
predicts the *following* semester.

Why 3500 students by default?
-----------------------------
HTU has roughly 5,000 students total and admits ~500 students per year.
The default cohort is 3,500 = 500 students/year × 7 admission years
(2019-2025), which gives a realistic active population (~2,000-2,500
in any given semester) and a believable demand distribution: 5-10
sections for popular intro courses, 2-4 for mid-level, 1 for late-year
electives.

This script also flattens the notebook's default ADM_YEAR_WEIGHTS at
load time so each admission year gets ~1/7 of the cohort instead of
the humped distribution the notebook ships with — that distribution
biases students toward middle years and inflates demand for the
courses those cohorts are about to take.

Going below ~2000 students will make the data look sparse (most
courses end up needing only 1 section because there aren't enough
students per course-year-major slice).

Usage
-----
    # Default: 3500 students (HTU-scale), cutoff = 2025_Fall, ~3-5s
    python generate_test_data.py

    # Quick smoke test — sparse output, but fast
    python generate_test_data.py --students 500

    # Bigger sample, simulate up through Spring 2026
    python generate_test_data.py --students 5000 --end-year 2026 --end-term Spring

    # Reproducible run for regression testing
    python generate_test_data.py --seed 42

    # Custom output directory
    python generate_test_data.py --output-dir test_data_small --students 500

After it runs, the next semester your /api/predict/upload/ endpoint will
forecast is the one immediately following the cutoff (e.g. cutoff
"2025_Fall"  ->  forecast "2025_Spring").

The script is read-only with respect to your real data; it writes to
`test_data/` (or whatever --output-dir you pass) and never touches
`ml/Data/output/`.
"""

import argparse
import json
import os
import random
import sys
import time


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_NOTEBOOK = os.path.join(
    "ml", "Data", "htu_full_generator_fixed.ipynb"
)
DEFAULT_OUTPUT_DIR = "test_data"
DEFAULT_STUDENTS = 3500
DEFAULT_END_YEAR = 2025
DEFAULT_END_TERM = "Fall"
VALID_TERMS = ("Fall", "Spring", "Summer")


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate test CSVs for the /api/predict/upload/ endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--notebook", default=DEFAULT_NOTEBOOK,
        help=f"Path to the generator notebook (default: {DEFAULT_NOTEBOOK})",
    )
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the two CSVs (default: {DEFAULT_OUTPUT_DIR}/)",
    )
    p.add_argument(
        "--students", type=int, default=DEFAULT_STUDENTS,
        help=(
            f"How many students to simulate (default: {DEFAULT_STUDENTS}, "
            f"matching HTU's ~500 admissions/year over a 7-year window). "
            f"Below ~2000 the demand distribution gets sparse."
        ),
    )
    p.add_argument(
        "--end-year", type=int, default=DEFAULT_END_YEAR,
        help=f"Last academic year to include (default: {DEFAULT_END_YEAR})",
    )
    p.add_argument(
        "--end-term", choices=VALID_TERMS, default=DEFAULT_END_TERM,
        help=f"Last term to include (default: {DEFAULT_END_TERM})",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Optional RNG seed for reproducible output.",
    )
    return p.parse_args()


# ── Notebook execution ────────────────────────────────────────────────────────
def load_notebook_namespace(notebook_path, end_year, end_term, seed):
    """
    Execute every code cell from the generator notebook *except* the
    final `if __name__ == '__main__'` driver blocks, then return the
    populated namespace (PLANS, CATALOG, run_simulation, etc.).
    """
    if not os.path.exists(notebook_path):
        sys.exit(
            f"ERROR: Generator notebook not found at:\n  {notebook_path}\n"
            f"Pass --notebook with the correct path."
        )

    # An .ipynb file is just JSON; we don't need nbformat as a dependency.
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]

    ns = {"__name__": "__notebook_loader__"}  # sentinel -> __main__ blocks skip

    for cell in code_cells:
        # In .ipynb JSON, `source` may be a string or a list of strings.
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if not src.strip():
            continue

        # The notebook has two driver cells guarded by `if __name__ == "__main__":`
        # Our sentinel above prevents them from executing, but *one* of them
        # contains the call to generate_all_csvs at module top-level — which we
        # also want to skip, since we'll invoke it ourselves with a custom path.
        # Both driver cells happen to start with a triple-quoted string or a
        # print banner; either way the `if __name__` guard is enough.
        try:
            exec(compile(src, "<notebook-cell>", "exec"), ns)
        except Exception as exc:
            raise RuntimeError(
                f"Failed while executing notebook cell:\n"
                f"---\n{src[:200]}...\n---\n{type(exc).__name__}: {exc}"
            ) from exc

    # Patch the cutoff semester. `make_semester_sequence` captures
    # SIM_END_YEAR / SIM_END_TERM as default-argument values at *function
    # definition* time, so reassigning the module globals afterwards has no
    # effect. We have to rewrite the function's __defaults__ tuple directly.
    ns["SIM_END_YEAR"] = end_year
    ns["SIM_END_TERM"] = end_term
    seq_fn = ns.get("make_semester_sequence")
    if seq_fn is not None and seq_fn.__defaults__ is not None:
        # signature: make_semester_sequence(start_year, end_year=..., end_term=...)
        seq_fn.__defaults__ = (end_year, end_term)

    # ── Flatten admission distribution ────────────────────────────────────
    # The notebook ships with a humped ADM_YEAR_WEIGHTS distribution
    # (e.g. [0.10, 0.12, 0.15, 0.18, 0.18, 0.15, 0.12]) that biases new
    # students into the middle years. For demo data we want a uniform
    # ~500-students-per-year intake to match HTU's actual admissions
    # pattern, so we replace the weights with a flat distribution and
    # also re-anchor ADMISSION_YEARS to the 7-year window ending at the
    # selected cutoff. `run_simulation` reads both as module globals at
    # call time, so this assignment is sufficient — no __defaults__
    # surgery needed here.
    if "ADM_YEAR_WEIGHTS" in ns and "ADMISSION_YEARS" in ns:
        n_years = len(ns["ADMISSION_YEARS"])
        ns["ADMISSION_YEARS"] = list(range(end_year - n_years + 1, end_year + 1))
        ns["ADM_YEAR_WEIGHTS"] = [1.0 / n_years] * n_years

    # The generator already does `random.seed()` (system-time seed). If the
    # user wants reproducibility, override after the namespace is built.
    if seed is not None:
        random.seed(seed)
        ns["random"].seed(seed)

    return ns


# ── Validation helpers ────────────────────────────────────────────────────────
def assert_required_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: Generated {name} is missing required columns: {missing}\n"
            f"Got columns: {list(df.columns)}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 60)
    print("Gear API — Test Data Generator")
    print("=" * 60)
    print(f"  Students:       {args.students:,}")
    print(f"  Cutoff:         {args.end_year}_{args.end_term}")
    print(f"  Output dir:     {os.path.abspath(args.output_dir)}")
    print(f"  Notebook:       {args.notebook}")
    if args.seed is not None:
        print(f"  Seed:           {args.seed}")
    print()

    t0 = time.time()
    print("[1/3] Loading simulator from notebook...")
    ns = load_notebook_namespace(
        args.notebook, args.end_year, args.end_term, args.seed
    )
    print(f"      Loaded in {time.time() - t0:.1f}s.")

    # Sanity-check the symbols we expect
    for name in ("run_simulation", "PLANS", "CATALOG"):
        if name not in ns:
            sys.exit(
                f"ERROR: Expected `{name}` in the notebook namespace but "
                f"it wasn't defined. Did the notebook structure change?"
            )

    print(f"[2/3] Simulating {args.students} students through "
          f"{args.end_year}_{args.end_term}...")
    t1 = time.time()
    df_students, df_enrollments = ns["run_simulation"](args.students)
    print(f"      Simulation finished in {time.time() - t1:.1f}s.")
    print(f"      Students:    {len(df_students):,}")
    print(f"      Enrollments: {len(df_enrollments):,}")

    # Quick column sanity check — these are what the API expects
    assert_required_columns(
        df_enrollments,
        ["student_id", "course_code", "semester_label", "grade", "credit_hours"],
        "htu_enrollments.csv",
    )
    assert_required_columns(
        df_students, ["student_id", "major"], "htu_students.csv",
    )

    print("[3/3] Writing CSVs...")
    os.makedirs(args.output_dir, exist_ok=True)
    enr_path = os.path.join(args.output_dir, "htu_enrollments.csv")
    stu_path = os.path.join(args.output_dir, "htu_students.csv")
    df_enrollments.to_csv(enr_path, index=False)
    df_students.to_csv(stu_path, index=False)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  {enr_path}  ({os.path.getsize(enr_path) // 1024} KB)")
    print(f"  {stu_path}  ({os.path.getsize(stu_path) // 1024} KB)")
    print()
    print("  Quick stats:")
    if "status" in df_students.columns:
        print("    Student status:")
        for k, v in df_students["status"].value_counts().items():
            print(f"      {k:<12} {v:,}")
    last_sem = df_enrollments["semester_label"].max()
    n_last = (df_enrollments["semester_label"] == last_sem).sum()
    print(f"    Latest semester present: {last_sem} "
          f"({n_last:,} enrollment rows)")
    print()
    print("Next steps:")
    print(f"  1. Make sure your Django server is running:")
    print(f"       python manage.py runserver")
    print(f"  2. POST these two files to /api/predict/upload/:")
    print(f"       enrollments_file -> {enr_path}")
    print(f"       students_file    -> {stu_path}")
    print(f"  3. The predicted semester will be the one *after* "
          f"{args.end_year}_{args.end_term}.")


if __name__ == "__main__":
    main()