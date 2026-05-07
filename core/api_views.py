"""
Stateless prediction API.
=========================
Accepts two CSV uploads (enrollments + students), runs the trained
ML pipeline in-memory, and returns the per-course "sections needed"
recommendation as JSON.

Endpoints
---------
GET  /api/health/            — simple liveness check
POST /api/predict/upload/    — multipart form with two files:
                                 - enrollments_file  (htu_enrollments.csv)
                                 - students_file     (htu_students.csv)
                               Optional form fields:
                                 - semester  (e.g. "2026_Fall")  — overrides
                                                                    auto-detect
                                 - section_cap   (int)
                                 - buffer_pct    (float, e.g. 0.10)

Response shape (200 OK)
-----------------------
{
  "semester_predicted": "2026_Fall",
  "section_cap":        30,
  "buffer_pct":         0.10,
  "totals": {
    "courses":           87,
    "predicted_enrollment": 2847,
    "sections_needed":   94,
    "students_processed": 1284
  },
  "courses": [
    {
      "course_code":          "0010203210",
      "course_title":         "Data Structures",
      "predicted_enrollment": 240,
      "sections_needed":      8,
      "last_semester_actual": 235,
      "status":               "optimal"   // optimal | warning | critical
    },
    ...
  ]
}
"""

import io
import traceback

import pandas as pd
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from ml.inference import predict_next_semester


# ── Required CSV columns (mirrors load_csv_data.py) ──────────────────────────
REQUIRED_ENROLLMENT_COLS = {
    "student_id", "course_code", "semester_label",
    "grade", "credit_hours",
}
REQUIRED_STUDENT_COLS = {
    "student_id", "major",
}

# ── Status thresholds (gap between predicted and last-semester actual) ───────
STATUS_WARNING_PCT = 0.20    # >20% increase ⇒ warning
STATUS_CRITICAL_PCT = 0.50   # >50% increase ⇒ critical


# ── Helpers ──────────────────────────────────────────────────────────────────

def _err(message, code=status.HTTP_400_BAD_REQUEST, **extra):
    """Return a uniform JSON error response."""
    body = {"error": message}
    body.update(extra)
    return Response(body, status=code)


def _parse_semester_label(label):
    """
    Parse '2024_Fall' → (2024, 'Fall').  Mirrors the loader's logic so
    the API tolerates the same variations users may have in their files.
    """
    s = str(label).strip()
    if "_" in s:
        year_str, term = s.split("_", 1)
        if year_str.isdigit():
            return int(year_str), term
    for term_name in ("Fall", "Spring", "Summer"):
        if term_name in s:
            digits = "".join(c for c in s if c.isdigit())
            year = int(digits) if digits else 2024
            return year, term_name
    return 2024, "Fall"


def _read_uploaded_csv(uploaded_file, dtype=None):
    """Read a Django InMemoryUploadedFile into a DataFrame."""
    raw = uploaded_file.read()
    return pd.read_csv(io.BytesIO(raw), dtype=dtype)


def _coerce_enrollments(df):
    """
    Ensure the enrollments DataFrame has the columns predict_next_semester
    expects: student_id, course_code, semester_label, year, term, grade,
    credit_hours.  Adds year/term if only semester_label is present.
    """
    df = df.copy()
    df["student_id"] = df["student_id"].astype(str)
    df["course_code"] = df["course_code"].astype(str).str.strip().str.zfill(10)

    if "year" not in df.columns or "term" not in df.columns:
        parsed = df["semester_label"].apply(_parse_semester_label)
        df["year"] = [p[0] for p in parsed]
        df["term"] = [p[1] for p in parsed]
    else:
        df["year"] = df["year"].astype(int)
        df["term"] = df["term"].astype(str)

    df["grade"] = df["grade"].fillna("").astype(str)
    df["credit_hours"] = df.get("credit_hours", 3).fillna(3).astype(int)
    return df


def _coerce_students(df):
    """
    Ensure the students DataFrame has the columns predict_next_semester needs:
    student_id, major, degree_type, admission_year,
    needs_rem_arabic/english/math.  Fills sensible defaults for optional cols.
    """
    df = df.copy()
    df["student_id"] = df["student_id"].astype(str)
    df["major"] = df["major"].astype(str).str.strip()

    if "degree_type" not in df.columns:
        df["degree_type"] = "Bachelor"
    else:
        df["degree_type"] = df["degree_type"].fillna("Bachelor").astype(str)
        df.loc[~df["degree_type"].isin(["Bachelor", "Technical"]),
               "degree_type"] = "Bachelor"

    if "admission_year" not in df.columns:
        df["admission_year"] = df.get("start_year", 2020)
    df["admission_year"] = (
        pd.to_numeric(df["admission_year"], errors="coerce")
        .fillna(2020).astype(int)
    )

    for col in ("needs_rem_arabic", "needs_rem_english", "needs_rem_math"):
        if col not in df.columns:
            df[col] = 0
        df[col] = (
            pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        )
    return df


def _last_semester_actuals(enrolls_df):
    """
    Compute the actual number of students that took each course in the
    most recent semester present in the uploaded enrollments — used as
    the 'LAST SEMESTER' column in the mockup.
    """
    if "sem_sort_key" not in enrolls_df.columns:
        sort_keys = enrolls_df["year"].astype(int) * 10 + enrolls_df["term"].map(
            {"Spring": 1, "Summer": 2, "Fall": 3}
        ).fillna(0).astype(int)
    else:
        sort_keys = enrolls_df["sem_sort_key"]

    df = enrolls_df.assign(_sort=sort_keys)
    last_key = df["_sort"].max()
    last_df = df[df["_sort"] == last_key]

    counts = (
        last_df.groupby("course_code")["student_id"]
        .nunique().to_dict()
    )
    return counts


def _course_titles_lookup(enrolls_df):
    """course_code → most common course_name from the enrollments file."""
    if "course_name" not in enrolls_df.columns:
        return {}
    return (
        enrolls_df.dropna(subset=["course_name"])
        .groupby("course_code")["course_name"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
        .to_dict()
    )


def _classify_status(predicted, actual):
    """Heuristic for the colored pill in the UI."""
    if actual is None or actual == 0:
        return "warning"
    growth = (predicted - actual) / actual
    if growth >= STATUS_CRITICAL_PCT:
        return "critical"
    if growth >= STATUS_WARNING_PCT:
        return "warning"
    return "optimal"


# ── Views ────────────────────────────────────────────────────────────────────

@api_view(["GET"])
def health(request):
    """Quick check that the API process is alive and the ML module imports."""
    return Response({"status": "ok", "service": "gear-api"})


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def predict_upload(request):
    """
    Accept enrollments + students CSVs, run the pipeline, return JSON.
    Stateless — nothing is written to the database.
    """
    enrollments_file = request.FILES.get("enrollments_file")
    students_file = request.FILES.get("students_file")

    if enrollments_file is None or students_file is None:
        return _err(
            "Both 'enrollments_file' and 'students_file' must be provided "
            "as multipart file fields."
        )

    # 1. Read the CSVs
    try:
        enrolls_raw = _read_uploaded_csv(
            enrollments_file,
            dtype={"student_id": str, "course_code": str},
        )
        students_raw = _read_uploaded_csv(
            students_file, dtype={"student_id": str},
        )
    except Exception as exc:
        return _err(f"Failed to read CSV: {exc}")

    # 2. Validate required columns
    missing_e = REQUIRED_ENROLLMENT_COLS - set(enrolls_raw.columns)
    if missing_e:
        return _err(
            "Enrollments CSV is missing required columns.",
            missing_columns=sorted(missing_e),
        )
    missing_s = REQUIRED_STUDENT_COLS - set(students_raw.columns)
    if missing_s:
        return _err(
            "Students CSV is missing required columns.",
            missing_columns=sorted(missing_s),
        )

    # 3. Normalize column types/defaults
    try:
        enrolls_df = _coerce_enrollments(enrolls_raw)
        students_df = _coerce_students(students_raw)
    except Exception as exc:
        return _err(f"Failed to normalize CSVs: {exc}")

    # 4. Optional knobs from the form
    next_sem_label = request.data.get("semester") or None
    section_cap = request.data.get("section_cap")
    buffer_pct = request.data.get("buffer_pct")
    try:
        section_cap = int(section_cap) if section_cap not in (None, "") else None
        buffer_pct = float(buffer_pct) if buffer_pct not in (None, "") else None
    except (TypeError, ValueError):
        return _err("section_cap must be int and buffer_pct must be float.")

    # 5. Run the pipeline
    try:
        student_preds_df, demand_df, sections_df, _latest_snap = (
            predict_next_semester(
                enrolls_df, students_df,
                next_sem_label=next_sem_label,
                section_cap=section_cap,
                buffer_pct=buffer_pct,
            )
        )
    except FileNotFoundError as exc:
        return _err(
            "Trained ML artifacts are missing on the server. "
            "Make sure ml/artifacts/ is populated.",
            detail=str(exc),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    except Exception as exc:
        return _err(
            "Pipeline failed.",
            detail=str(exc),
            traceback=traceback.format_exc(),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if len(sections_df) == 0:
        return _err(
            "Pipeline returned zero predictions. Check that the uploaded "
            "data covers students with enrollment history.",
            code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    # 6. Build the response
    last_actuals = _last_semester_actuals(enrolls_df)
    titles = _course_titles_lookup(enrolls_df)

    predicted_sem = (
        student_preds_df["next_sem"].iloc[0]
        if "next_sem" in student_preds_df.columns and len(student_preds_df) > 0
        else next_sem_label or ""
    )

    courses_payload = []
    for _, row in sections_df.sort_values(
        "predicted_count", ascending=False
    ).iterrows():
        cc = str(row["course_code"])
        predicted = int(row["predicted_count"])
        actual = last_actuals.get(cc)
        courses_payload.append({
            "course_code":           cc,
            "course_title":          titles.get(cc, ""),
            "predicted_enrollment":  predicted,
            "sections_needed":       int(row["predicted_sections"]),
            "last_semester_actual":  actual,  # may be None
            "status":                _classify_status(predicted, actual),
        })

    response = {
        "semester_predicted": predicted_sem,
        "section_cap":        section_cap,
        "buffer_pct":         buffer_pct,
        "totals": {
            "courses":              len(courses_payload),
            "predicted_enrollment": int(sections_df["predicted_count"].sum()),
            "sections_needed":      int(sections_df["predicted_sections"].sum()),
            "students_processed":   int(students_df["student_id"].nunique()),
        },
        "courses": courses_payload,
    }
    return Response(response)
