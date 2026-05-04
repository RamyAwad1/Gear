from django.db import models
from django.contrib.auth.models import AbstractUser


class Department(models.Model):
    name = models.CharField(max_length=255)
    code = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return f"{self.code} - {self.name}"


class AppUser(AbstractUser):
    # AbstractUser has built-in first/last name and email
    role = models.CharField(max_length=100)  # e.g., 'Department Head', 'Advisor'
    department = models.ForeignKey(
        Department, on_delete=models.SET_NULL, null=True, blank=True
    )

    def __str__(self):
        return f"{self.username} ({self.role})"


class Course(models.Model):
    course_code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=255)
    credit_hours = models.IntegerField()
    lecture_hours = models.IntegerField(default=3)
    lab_hours = models.IntegerField(default=0)
    is_elective = models.BooleanField(default=False)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.course_code} - {self.title}"


class StudyPlan(models.Model):
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    recommended_semester = models.IntegerField()

    def __str__(self):
        return f"{self.department.code} - {self.course.course_code} (Sem {self.recommended_semester})"


class Student(models.Model):
    student_university_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    major = models.ForeignKey(Department, on_delete=models.CASCADE)
    degree_type = models.CharField(
        max_length=20,
        choices=[("Bachelor", "Bachelor"), ("Technical", "Technical")],
        default="Bachelor",
    )
    admission_year = models.IntegerField(default=2020)
    cumulative_gpa = models.DecimalField(max_digits=4, decimal_places=2)
    earned_credit_hours = models.IntegerField()
    is_graduating_flag = models.BooleanField(default=False)
    needs_rem_arabic = models.BooleanField(default=False)
    needs_rem_english = models.BooleanField(default=False)
    needs_rem_math = models.BooleanField(default=False)

    @property
    def plan_key(self):
        """Returns the plan key used by the ML pipeline, e.g. 'CS_Bachelor'."""
        return f"{self.major.code}_{self.degree_type}"

    def __str__(self):
        return f"{self.student_university_id} - {self.name}"


class EnrollmentHistory(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    semester_label = models.CharField(max_length=20)   # e.g. "2024_Fall"
    year = models.IntegerField()
    term = models.CharField(max_length=10)             # "Fall", "Spring", "Summer"
    grade = models.CharField(max_length=5, blank=True, null=True)
    credit_hours = models.IntegerField(default=3)
    status = models.CharField(max_length=50, default="Completed")

    class Meta:
        indexes = [
            models.Index(fields=["student", "year", "term"]),
        ]

    def __str__(self):
        return f"{self.student.student_university_id} - {self.course.course_code} ({self.semester_label})"


class PredictionRun(models.Model):
    run_timestamp = models.DateTimeField(auto_now_add=True)
    triggered_by = models.ForeignKey(
        AppUser, on_delete=models.SET_NULL, null=True, blank=True
    )
    model_version = models.CharField(max_length=50)
    semester_predicted = models.CharField(max_length=20, blank=True)
    total_courses_predicted = models.IntegerField(default=0)
    total_students_processed = models.IntegerField(default=0)

    def __str__(self):
        return f"Run {self.id} at {self.run_timestamp.strftime('%Y-%m-%d %H:%M')}"


class CoursePrediction(models.Model):
    run = models.ForeignKey(PredictionRun, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    predicted_demand = models.IntegerField()
    suggested_sections = models.IntegerField()
    confidence_score = models.DecimalField(
        max_digits=5, decimal_places=4, default=0.0
    )

    class Meta:
        unique_together = ("run", "course")

    def __str__(self):
        return f"{self.course.course_code} Prediction (Run {self.run.id})"


class StudentPrediction(models.Model):
    """Per-student per-course prediction from a pipeline run."""
    run = models.ForeignKey(PredictionRun, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    probability = models.DecimalField(max_digits=6, decimal_places=4)
    predicted_enroll = models.BooleanField()

    class Meta:
        unique_together = ("run", "student", "course")
        indexes = [
            models.Index(fields=["run", "student"]),
        ]

    def __str__(self):
        return f"{self.student.student_university_id} → {self.course.course_code} ({self.probability})"


class PredictionExplanation(models.Model):
    prediction = models.ForeignKey(CoursePrediction, on_delete=models.CASCADE)
    feature_name = models.CharField(max_length=255)
    impact_value = models.DecimalField(max_digits=10, decimal_places=4)

    def __str__(self):
        return f"{self.feature_name} impact on {self.prediction.course.course_code}"


class FlaggedStudent(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    run = models.ForeignKey(PredictionRun, on_delete=models.CASCADE)
    prediction = models.ForeignKey(
        CoursePrediction, on_delete=models.CASCADE, null=True, blank=True
    )
    reason = models.CharField(max_length=255)
    is_resolved = models.BooleanField(default=False)

    def __str__(self):
        return f"Flagged: {self.student.student_university_id} — {self.reason}"