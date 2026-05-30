"""
Django management command: run_pipeline
=======================================
Fetches current enrollment + student data from PostgreSQL,
runs the Pipeline v3 inference, and writes results back to the DB.

Usage:
    python manage.py run_pipeline
    python manage.py run_pipeline --semester "2026_Fall"
    python manage.py run_pipeline --skip-explanations
"""

import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import (
    Course,
    Student,
    EnrollmentHistory,
    PredictionRun,
    CoursePrediction,
    StudentPrediction,
    PredictionExplanation,
    FlaggedStudent,
)
from ml.inference import (
    predict_next_semester,
    explain_live_course_demand,
)

# Credit-hour threshold below which a student is flagged as near-graduation
NEAR_GRADUATION_THRESHOLD = 33


class Command(BaseCommand):
    help = "Execute Pipeline v3 inference and save predictions to PostgreSQL."

    def add_arguments(self, parser):
        parser.add_argument(
            "--semester",
            type=str,
            default=None,
            help='Override the predicted semester label (e.g. "2026_Fall").',
        )
        parser.add_argument(
            "--skip-explanations",
            action="store_true",
            help="Skip SHAP explanation generation (faster).",
        )
        parser.add_argument(
            "--model-version",
            type=str,
            default="v3",
            help="Model version tag for the PredictionRun record.",
        )

    def handle(self, *args, **options):
        self.stdout.write("=" * 60)
        self.stdout.write("HTU Intelligent Registration Pipeline v3")
        self.stdout.write("=" * 60)

        # ── 1. Fetch live data from PostgreSQL ─────────────────────────────
        self.stdout.write("Step 1: Loading data from database...")

        enrollments_qs = EnrollmentHistory.objects.select_related(
            "student", "course"
        ).all()

        if not enrollments_qs.exists():
            self.stderr.write(self.style.ERROR(
                "No enrollment records found in the database. "
                "Load your CSV data first (python manage.py load_csv_data)."
            ))
            return

        enrolls_records = []
        for e in enrollments_qs:
            enrolls_records.append({
                "student_id":     e.student.student_university_id,
                "course_code":    e.course.course_code,
                "semester_label": e.semester_label,
                "year":           e.year,
                "term":           e.term,
                "grade":          e.grade or "",
                "credit_hours":   e.credit_hours,
            })
        current_enrolls_df = pd.DataFrame(enrolls_records)

        students_qs = Student.objects.select_related("major").all()
        student_records = []
        for s in students_qs:
            student_records.append({
                "student_id":       s.student_university_id,
                "major":            s.major.code,
                "degree_type":      s.degree_type,
                "admission_year":   s.admission_year,
                "needs_rem_arabic": int(s.needs_rem_arabic),
                "needs_rem_english": int(s.needs_rem_english),
                "needs_rem_math":   int(s.needs_rem_math),
                "plan_key":         s.plan_key,
            })
        current_students_df = pd.DataFrame(student_records)

        self.stdout.write(self.style.SUCCESS(
            f"  Loaded {len(current_enrolls_df)} enrollment records, "
            f"{len(current_students_df)} students."
        ))

        # ── 2. Execute the pipeline ────────────────────────────────────────
        self.stdout.write("Step 2: Running inference pipeline...")

        student_preds_df, demand_df, sections_df, latest_snap = (
            predict_next_semester(
                current_enrolls_df,
                current_students_df,
                next_sem_label=options["semester"],
            )
        )

        if len(student_preds_df) == 0:
            self.stderr.write(self.style.ERROR(
                "Pipeline returned zero predictions. "
                "Check that your artifacts/ folder contains trained models."
            ))
            return

        predicted_sem = student_preds_df["next_sem"].iloc[0]
        self.stdout.write(self.style.SUCCESS(
            f"  Predicted semester: {predicted_sem}"
        ))
        self.stdout.write(self.style.SUCCESS(
            f"  {len(demand_df)} courses, "
            f"{len(student_preds_df)} student-course predictions."
        ))

        # ── 3. Write results to database (atomic) ─────────────────────────
        self.stdout.write("Step 3: Saving results to database...")

        with transaction.atomic():
            # 3a. Create PredictionRun record
            run = PredictionRun.objects.create(
                model_version=options["model_version"],
                semester_predicted=predicted_sem,
                total_courses_predicted=len(demand_df),
                total_students_processed=current_students_df.shape[0],
            )
            self.stdout.write(f"  Created PredictionRun #{run.id}")

            # 3b. Build course lookup (course_code → Course object)
            course_lookup = {
                c.course_code: c
                for c in Course.objects.all()
            }

            # 3c. Save CoursePrediction records
            course_pred_lookup = {}  # course_code → CoursePrediction
            skipped_courses = 0

            for _, row in demand_df.iterrows():
                cc = str(row["course_code"])
                course_obj = course_lookup.get(cc)

                if course_obj is None:
                    skipped_courses += 1
                    continue

                cp = CoursePrediction.objects.create(
                    run=run,
                    course=course_obj,
                    predicted_demand=int(row["predicted_count"]),
                    suggested_sections=int(row["predicted_sections"]),
                    confidence_score=0.0,
                )
                course_pred_lookup[cc] = cp

            self.stdout.write(self.style.SUCCESS(
                f"  Saved {len(course_pred_lookup)} CoursePrediction records"
                + (f" ({skipped_courses} courses not in DB, skipped)"
                   if skipped_courses else "")
            ))

            # 3d. Save StudentPrediction records (bulk)
            student_lookup = {
                s.student_university_id: s
                for s in Student.objects.all()
            }

            student_pred_objects = []
            for _, row in student_preds_df.iterrows():
                sid = str(row["student_id"])
                cc = str(row["course_code"])
                student_obj = student_lookup.get(sid)
                course_obj = course_lookup.get(cc)

                if student_obj is None or course_obj is None:
                    continue

                student_pred_objects.append(StudentPrediction(
                    run=run,
                    student=student_obj,
                    course=course_obj,
                    probability=round(float(row["prob_enroll"]), 4),
                    predicted_enroll=bool(row["pred_enroll"]),
                ))

            if student_pred_objects:
                StudentPrediction.objects.bulk_create(
                    student_pred_objects, batch_size=1000
                )
            self.stdout.write(self.style.SUCCESS(
                f"  Saved {len(student_pred_objects)} StudentPrediction records"
            ))

            # 3e. Save SHAP explanations (per course, top 10 features)
            if not options["skip_explanations"]:
                self.stdout.write("  Generating SHAP explanations...")
                explanation_count = 0
                for cc, cp in course_pred_lookup.items():
                    explanation_df = explain_live_course_demand(
                        cc, latest_snap, student_preds_df, top_k=10
                    )
                    if explanation_df is None:
                        continue
                    for _, erow in explanation_df.iterrows():
                        PredictionExplanation.objects.create(
                            prediction=cp,
                            feature_name=str(erow["feature"]),
                            impact_value=round(
                                float(erow["mean_abs_contribution"]), 4
                            ),
                        )
                        explanation_count += 1
                self.stdout.write(self.style.SUCCESS(
                    f"  Saved {explanation_count} PredictionExplanation records"
                ))

            # 3f. Flag at-risk (near-graduation) students
            flagged_count = 0
            for _, row in latest_snap.iterrows():
                sid = str(row["student_id"])
                student_obj = student_lookup.get(sid)
                if student_obj is None:
                    continue

                hrs_remaining = row.get("extra_hrs_remaining", None)
                if hrs_remaining is not None and hrs_remaining <= NEAR_GRADUATION_THRESHOLD:
                    FlaggedStudent.objects.create(
                        student=student_obj,
                        run=run,
                        prediction=None,
                        reason=(
                            f"Near graduation: {int(hrs_remaining)} credit "
                            f"hours remaining (threshold: "
                            f"{NEAR_GRADUATION_THRESHOLD})"
                        ),
                        is_resolved=False,
                    )
                    flagged_count += 1

            if flagged_count:
                self.stdout.write(self.style.SUCCESS(
                    f"  Flagged {flagged_count} near-graduation students"
                ))

        # ── Done ───────────────────────────────────────────────────────────
        self.stdout.write("=" * 60)
        self.stdout.write(self.style.SUCCESS(
            f"Pipeline complete! Run #{run.id} saved to database."
        ))
        self.stdout.write("=" * 60)