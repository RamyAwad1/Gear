"""
load_csv_data.py
================
Clears the database and reloads all data from the generated CSVs.
Reads from ml/Data/output/ (the canonical data location inside the repo).

Usage:
    python load_csv_data.py
"""

import os
import sys
import django
import pandas as pd
from decimal import Decimal, InvalidOperation

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "htu_registration.settings")
django.setup()

from core.models import (
    Department,
    Course,
    Student,
    EnrollmentHistory,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ml", "Data", "output")

ENROLLMENTS_CSV = os.path.join(DATA_DIR, "htu_enrollments.csv")
STUDENTS_CSV    = os.path.join(DATA_DIR, "htu_students.csv")

# Plan-level course CSVs (for lecture/lab hours)
PLAN_CSVS = [
    os.path.join(DATA_DIR, "CS_Bachelor_Courses.csv"),
    os.path.join(DATA_DIR, "CS_Technical_Courses.csv"),
    os.path.join(DATA_DIR, "AI_Bachelor_Courses.csv"),
    os.path.join(DATA_DIR, "AI_Technical_Courses.csv"),
    os.path.join(DATA_DIR, "Cyber_Bachelor_Courses.csv"),
    os.path.join(DATA_DIR, "Cyber_Technical_Courses.csv"),
]

# ── Department map ────────────────────────────────────────────────────────────
DEPT_DEFINITIONS = {
    "CS":    ("Computer Science", "CS"),
    "AI":    ("Data Science & AI", "AI"),
    "Cyber": ("Cybersecurity", "Cyber"),
    "UNI":   ("University Requirements", "UNI"),
}

# Remedial course codes — used to flag students
REMEDIAL_ARABIC  = {"0030301110"}
REMEDIAL_ENGLISH = {"0030301120"}
REMEDIAL_MATH    = {"0030303110"}


def safe_decimal(val, default="0.00"):
    try:
        return Decimal(str(val))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def parse_semester_label(label):
    """
    Parse a semester_label string like '2024_Fall' into (year, term).
    Handles multiple known formats gracefully.
    """
    label = str(label).strip()

    # Expected format: "YYYY_Term"
    if "_" in label:
        parts = label.split("_", 1)
        if len(parts) == 2:
            year_str, term = parts
            if year_str.isdigit():
                return int(year_str), term, label

    # Fallback: "Fall 2024" or "Spring 2025"
    for term_name in ["Fall", "Spring", "Summer"]:
        if term_name in label:
            digits = "".join(c for c in label if c.isdigit())
            year = int(digits) if digits else 2024
            normalized = f"{year}_{term_name}"
            return year, term_name, normalized

    # Last resort default
    return 2024, "Fall", "2024_Fall"


def load_course_metadata():
    """
    Read all plan CSVs and build a dict: course_code → {lecture_hours, lab_hours}.
    The plan CSVs are the authoritative source of lecture/lab hours.
    """
    meta = {}
    for path in PLAN_CSVS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={"course_code": str})
        for _, row in df.iterrows():
            code = str(row["course_code"]).strip().zfill(10)
            if code not in meta:
                meta[code] = {
                    "lecture_hours": int(row.get("lecture_hours", 3)),
                    "lab_hours":     int(row.get("lab_hours", 0)),
                    "title":         str(row.get("course_name_en", "")),
                }
    return meta


def load_data():
    print("=" * 60)
    print("HTU Registration System — Data Loader")
    print("=" * 60)

    # ── Validate files exist ───────────────────────────────────────────────
    for path in [ENROLLMENTS_CSV, STUDENTS_CSV]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    # ── Clear existing data ────────────────────────────────────────────────
    print("Clearing existing database records...")
    EnrollmentHistory.objects.all().delete()
    Student.objects.all().delete()
    Course.objects.all().delete()
    Department.objects.all().delete()
    print("  Done.")

    # ── Departments ────────────────────────────────────────────────────────
    print("Creating departments...")
    dept_map = {}
    for key, (name, code) in DEPT_DEFINITIONS.items():
        dept_map[key] = Department.objects.create(name=name, code=code)
    print(f"  Created {len(dept_map)} departments.")

    # ── Load CSVs ──────────────────────────────────────────────────────────
    print("Reading CSVs...")
    enrollments_df = pd.read_csv(
        ENROLLMENTS_CSV, dtype={"course_code": str, "student_id": str}
    )
    students_df = pd.read_csv(
        STUDENTS_CSV, dtype={"student_id": str}
    )

    print(f"  Enrollments: {len(enrollments_df):,} rows")
    print(f"  Students:    {len(students_df):,} rows")

    # Print actual columns so mismatches are visible
    print(f"  Enrollment columns: {list(enrollments_df.columns)}")
    print(f"  Student columns:    {list(students_df.columns)}")

    # ── Parse semester labels ──────────────────────────────────────────────
    print("Parsing semester labels...")
    parsed = enrollments_df["semester_label"].apply(parse_semester_label)
    enrollments_df["year_parsed"]           = [p[0] for p in parsed]
    enrollments_df["term_parsed"]           = [p[1] for p in parsed]
    enrollments_df["semester_label_clean"]  = [p[2] for p in parsed]

    unique_labels = enrollments_df["semester_label_clean"].unique()
    print(f"  Found {len(unique_labels)} unique semesters: {sorted(unique_labels)[:6]}...")

    # ── Load course metadata (lecture/lab hours) ───────────────────────────
    course_meta = load_course_metadata()
    print(f"  Loaded metadata for {len(course_meta)} unique courses from plan CSVs.")

    # ── Courses ────────────────────────────────────────────────────────────
    print("Populating courses...")
    unique_courses = (
        enrollments_df[["course_code", "course_name", "credit_hours"]]
        .drop_duplicates(subset=["course_code"])
    )

    courses_to_create = []
    for _, row in unique_courses.iterrows():
        code = str(row["course_code"]).strip().zfill(10)
        meta = course_meta.get(code, {})

        # Assign department by major prefix in course code or name
        dept = dept_map["UNI"]
        name_lower = str(row.get("course_name", "")).lower()
        if code.startswith("0000") or code.startswith("0010"):
            dept = dept_map["CS"]
        elif code.startswith("0040"):
            dept = dept_map["AI"]
        elif code.startswith("0030"):
            dept = dept_map["UNI"]

        courses_to_create.append(Course(
            course_code=code,
            title=meta.get("title") or str(row.get("course_name", code)),
            credit_hours=int(row.get("credit_hours", 3)),
            lecture_hours=meta.get("lecture_hours", 3),
            lab_hours=meta.get("lab_hours", 0),
            is_elective=False,
            department=dept,
        ))

    Course.objects.bulk_create(courses_to_create, ignore_conflicts=True)
    course_obj_map = {c.course_code: c for c in Course.objects.all()}
    print(f"  Created {len(course_obj_map)} courses.")

    # ── Students ───────────────────────────────────────────────────────────
    print("Populating students...")

    # Determine which students took remedial courses (for the flag fields)
    remedial_enrollment = enrollments_df.copy()
    remedial_enrollment["course_code_padded"] = (
        remedial_enrollment["course_code"].astype(str).str.zfill(10)
    )
    rem_arabic_students = set(
        remedial_enrollment[
            remedial_enrollment["course_code_padded"].isin(REMEDIAL_ARABIC)
        ]["student_id"].tolist()
    )
    rem_english_students = set(
        remedial_enrollment[
            remedial_enrollment["course_code_padded"].isin(REMEDIAL_ENGLISH)
        ]["student_id"].tolist()
    )
    rem_math_students = set(
        remedial_enrollment[
            remedial_enrollment["course_code_padded"].isin(REMEDIAL_MATH)
        ]["student_id"].tolist()
    )

    students_to_create = []
    for _, row in students_df.iterrows():
        sid = str(row["student_id"])
        major_str = str(row.get("major", "CS")).strip()
        dept = dept_map.get(major_str, dept_map["CS"])

        # degree_type: prefer column if present, else default Bachelor
        degree_type = str(row.get("degree_type", "Bachelor")).strip()
        if degree_type not in ("Bachelor", "Technical"):
            degree_type = "Bachelor"

        # admission_year: prefer column if present
        admission_year = int(row.get("admission_year", row.get("start_year", 2020)))

        students_to_create.append(Student(
            student_university_id=sid,
            name=f"Student {sid}",
            major=dept,
            degree_type=degree_type,
            admission_year=admission_year,
            cumulative_gpa=safe_decimal(
                row.get("final_cum_gpa", row.get("cum_gpa", 2.5))
            ),
            earned_credit_hours=int(
                row.get("total_passed_hrs", row.get("earned_credit_hours", 0))
            ),
            is_graduating_flag=(
                str(row.get("status", "")).strip() == "Graduated"
            ),
            needs_rem_arabic=(sid in rem_arabic_students),
            needs_rem_english=(sid in rem_english_students),
            needs_rem_math=(sid in rem_math_students),
        ))

    Student.objects.bulk_create(students_to_create, batch_size=1000)
    student_obj_map = {
        s.student_university_id: s for s in Student.objects.all()
    }
    print(f"  Created {len(student_obj_map)} students.")

    # ── Enrollments ────────────────────────────────────────────────────────
    print(f"Populating {len(enrollments_df):,} enrollment records...")
    enrollments_to_create = []
    skipped = 0

    for _, row in enrollments_df.iterrows():
        sid = str(row["student_id"])
        code = str(row["course_code"]).strip().zfill(10)
        stu = student_obj_map.get(sid)
        crs = course_obj_map.get(code)

        if stu is None or crs is None:
            skipped += 1
            continue

        grade = str(row.get("grade", "")).strip()
        year, term, sem_label = (
            int(row["year_parsed"]),
            str(row["term_parsed"]),
            str(row["semester_label_clean"]),
        )

        enrollments_to_create.append(EnrollmentHistory(
            student=stu,
            course=crs,
            semester_label=sem_label,
            year=year,
            term=term,
            grade=grade if grade else None,
            credit_hours=int(row.get("credit_hours", crs.credit_hours)),
            status=(
                "Completed" if grade in {"D", "M", "P"}
                else "Failed/Withdrawn" if grade in {"U", "W"}
                else "Enrolled"
            ),
        ))

    EnrollmentHistory.objects.bulk_create(
        enrollments_to_create, batch_size=5000
    )
    print(f"  Created {len(enrollments_to_create):,} enrollment records.")
    if skipped:
        print(f"  Skipped {skipped} rows (student or course not found in DB).")

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Data load complete!")
    print(f"  Departments:  {Department.objects.count()}")
    print(f"  Courses:      {Course.objects.count()}")
    print(f"  Students:     {Student.objects.count()}")
    print(f"  Enrollments:  {EnrollmentHistory.objects.count():,}")
    print("=" * 60)


if __name__ == "__main__":
    load_data()