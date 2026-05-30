"""
Remove the SubstituteRequest model from the schema.

The substitute-request workflow was prototyped during development but
descoped from the final scope (see Future Work in the capstone report).
This migration drops the corresponding table so the schema matches the
final delivered system.

If your database is still on migration 0002 (you never ran 0003), you
can simply delete this file along with 0003_substituterequest.py.
Otherwise, run:

    python manage.py migrate core

to apply this deletion.
"""
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0003_substituterequest"),
    ]

    operations = [
        migrations.DeleteModel(
            name="SubstituteRequest",
        ),
    ]
