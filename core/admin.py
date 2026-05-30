from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import (
    Department, AppUser, Course, StudyPlan, Student, 
    EnrollmentHistory, PredictionRun, CoursePrediction, 
    PredictionExplanation, FlaggedStudent
)

#Register your models here.
admin.site.register(Department)
admin.site.register(AppUser)
admin.site.register(Course)
admin.site.register(StudyPlan)
admin.site.register(Student)
admin.site.register(EnrollmentHistory)
admin.site.register(PredictionRun)
admin.site.register(CoursePrediction)
admin.site.register(PredictionExplanation)
admin.site.register(FlaggedStudent)