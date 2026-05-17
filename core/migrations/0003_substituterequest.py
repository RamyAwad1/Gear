import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0002_studentprediction_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="SubstituteRequest",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("student_id", models.CharField(max_length=50)),
                ("original_course_code", models.CharField(max_length=20)),
                ("original_course_title", models.CharField(blank=True, max_length=255)),
                ("substitute_course_code", models.CharField(max_length=20)),
                ("substitute_course_title", models.CharField(blank=True, max_length=255)),
                ("reason", models.TextField()),
                ("status", models.CharField(
                    choices=[("pending", "Pending"), ("approved", "Approved"), ("rejected", "Rejected")],
                    default="pending",
                    max_length=20,
                )),
                ("reviewer_notes", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("reviewed_at", models.DateTimeField(blank=True, null=True)),
                ("reviewer", models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name="reviewed_substitute_requests",
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="substituterequest",
            index=models.Index(fields=["student_id", "status"], name="core_substi_student_8c5e1f_idx"),
        ),
        migrations.AddIndex(
            model_name="substituterequest",
            index=models.Index(fields=["status", "-created_at"], name="core_substi_status_a7b3c2_idx"),
        ),
    ]
