"""
URL routes for the `core` app.
Mounted at /api/ from the project urls.py.
"""
from django.urls import path

from . import api_views
from . import substitute_views

app_name = "core"

urlpatterns = [
    path("health/",         api_views.health,         name="health"),
    path("predict/upload/", api_views.predict_upload, name="predict_upload"),
    # Substitute requests
    path(
        "substitute-requests/",
        substitute_views.substitute_requests_list,
        name="substitute_requests_list",
    ),
    path(
        "substitute-requests/<int:pk>/",
        substitute_views.substitute_request_detail,
        name="substitute_request_detail",
    ),
]