from django.db import models
from django.contrib.auth.models import AbstractUser

class Department(models.Model):
    name = models.CharField(max_length=255)
    code = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return f"{self.code} - {self.name}"

class AppUser(AbstractUser):
    
    #note:abstract user has a built in first/last name and email so we dont need to add them
    role = models.CharField(max_length=100) # e.g., 'Department Head', 'Advisor'
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.username} ({self.role})"

class Course(models.Model):
    course_code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=255)
    credit_hours = models.IntegerField()
    is_elective = models.BooleanField(default=False)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.course_code} - {self.title}"

class StudyPlan(models.Model):
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    recommended_semester = models.IntegerField() # e.g., 1 through 8

    def __str__(self):
        return f"{self.department.code} - {self.course.course_code} (Sem {self.recommended_semester})"

class Student(models.Model):
    student_university_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    major = models.ForeignKey(Department, on_delete=models.CASCADE)
    cumulative_gpa = models.DecimalField(max_digits=4, decimal_places=2)
    earned_credit_hours = models.IntegerField()
    is_graduating_flag = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.student_university_id} - {self.name}"

class EnrollmentHistory(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    semester_year = models.CharField(max_length=50) # e.g., "Fall 2025"
    grade = models.CharField(max_length=5, blank=True, null=True)
    status = models.CharField(max_length=50) # e.g., "Completed", "Withdrawn"

    def __str__(self):
        return f"{self.student.student_university_id} - {self.course.course_code} ({self.semester_year})"

class PredictionRun(models.Model):
    run_timestamp = models.DateTimeField(auto_now_add=True)
    triggered_by = models.ForeignKey(AppUser, on_delete=models.SET_NULL, null=True)
    model_version = models.CharField(max_length=50)

    def __str__(self):
        return f"Run {self.id} at {self.run_timestamp.strftime('%Y-%m-%d %H:%M')}"

class CoursePrediction(models.Model):
    run = models.ForeignKey(PredictionRun, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    predicted_demand = models.IntegerField()
    suggested_sections = models.IntegerField()
    confidence_score = models.DecimalField(max_digits=5, decimal_places=4)

    def __str__(self):
        return f"{self.course.course_code} Prediction (Run {self.run.id})"

class PredictionExplanation(models.Model):
    prediction = models.ForeignKey(CoursePrediction, on_delete=models.CASCADE)
    feature_name = models.CharField(max_length=255) # e.g., "Past Enrollment"
    impact_value = models.DecimalField(max_digits=10, decimal_places=4) # SHAP/LIME value

    def __str__(self):
        return f"{self.feature_name} impact on {self.prediction.course.course_code}"

class FlaggedStudent(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    prediction = models.ForeignKey(CoursePrediction, on_delete=models.CASCADE)
    reason = models.CharField(max_length=255)
    is_resolved = models.BooleanField(default=False)

    def __str__(self):
        return f"Flagged: {self.student.student_university_id} for {self.prediction.course.course_code}"
# Create your models here.
