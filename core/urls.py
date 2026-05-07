"""
URL routes for the `core` app.
Mounted at /api/ from the project urls.py.
"""
from django.urls import path

from . import api_views

app_name = "core"

urlpatterns = [
    path("health/",          api_views.health,         name="health"),
    path("predict/upload/",  api_views.predict_upload, name="predict_upload"),
]
