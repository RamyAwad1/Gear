"""
Substitute-request CRUD endpoints.
==================================
Students submit a request to substitute a required graduation course
with a different course. Reviewers (advisors / department heads) list
pending requests and approve or reject them with optional notes.

Endpoints
---------
GET  /api/substitute-requests/                       list all (admin)
GET  /api/substitute-requests/?student_id=<id>       list for one student
POST /api/substitute-requests/                       create
GET  /api/substitute-requests/<int:pk>/              detail
PATCH /api/substitute-requests/<int:pk>/             update status / notes

Payload (POST):
{
  "student_id": "202400001",
  "original_course_code": "0040303121",
  "original_course_title": "Maths for Computing",
  "substitute_course_code": "0030303111",
  "substitute_course_title": "Functional Math",
  "reason": "Already passed Functional Math at another institution."
}

Payload (PATCH, reviewer action):
{
  "status": "approved" | "rejected",
  "reviewer_notes": "..."
}
"""
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status as drf_status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import SubstituteRequest


REQUIRED_CREATE_FIELDS = {
    "student_id",
    "original_course_code",
    "substitute_course_code",
    "reason",
}

UPDATABLE_FIELDS = {"status", "reviewer_notes"}
VALID_STATUSES = {"pending", "approved", "rejected"}


def _serialize(req: SubstituteRequest) -> dict:
    return {
        "id":                      req.id,
        "student_id":              req.student_id,
        "original_course_code":    req.original_course_code,
        "original_course_title":   req.original_course_title,
        "substitute_course_code":  req.substitute_course_code,
        "substitute_course_title": req.substitute_course_title,
        "reason":                  req.reason,
        "status":                  req.status,
        "reviewer_notes":          req.reviewer_notes,
        "reviewer":                req.reviewer.username if req.reviewer else None,
        "created_at":              req.created_at.isoformat(),
        "updated_at":              req.updated_at.isoformat(),
        "reviewed_at":             req.reviewed_at.isoformat() if req.reviewed_at else None,
    }


@api_view(["GET", "POST"])
def substitute_requests_list(request):
    """GET to list, POST to create."""
    if request.method == "GET":
        qs = SubstituteRequest.objects.all()
        student_id = request.query_params.get("student_id")
        status_filter = request.query_params.get("status")
        if student_id:
            qs = qs.filter(student_id=str(student_id).strip())
        if status_filter:
            qs = qs.filter(status=status_filter)
        return Response({"results": [_serialize(r) for r in qs]})

    # POST: create
    data = request.data
    missing = REQUIRED_CREATE_FIELDS - set(k for k, v in data.items() if v not in (None, ""))
    if missing:
        return Response(
            {"error": "Missing required fields.",
             "missing_fields": sorted(missing)},
            status=drf_status.HTTP_400_BAD_REQUEST,
        )

    orig_code = str(data["original_course_code"]).strip()
    sub_code = str(data["substitute_course_code"]).strip()
    if orig_code == sub_code:
        return Response(
            {"error": "Original and substitute courses must be different."},
            status=drf_status.HTTP_400_BAD_REQUEST,
        )

    req = SubstituteRequest.objects.create(
        student_id=str(data["student_id"]).strip(),
        original_course_code=orig_code,
        original_course_title=str(data.get("original_course_title", "")).strip(),
        substitute_course_code=sub_code,
        substitute_course_title=str(data.get("substitute_course_title", "")).strip(),
        reason=str(data["reason"]).strip(),
    )
    return Response(_serialize(req), status=drf_status.HTTP_201_CREATED)


@api_view(["GET", "PATCH"])
def substitute_request_detail(request, pk):
    """GET single, PATCH to update status / reviewer notes."""
    req = get_object_or_404(SubstituteRequest, pk=pk)

    if request.method == "GET":
        return Response(_serialize(req))

    # PATCH: only allow whitelisted fields
    data = request.data
    unknown = set(data.keys()) - UPDATABLE_FIELDS
    if unknown:
        return Response(
            {"error": "Some fields are not updatable.",
             "unknown_fields": sorted(unknown)},
            status=drf_status.HTTP_400_BAD_REQUEST,
        )

    if "status" in data:
        new_status = str(data["status"]).strip().lower()
        if new_status not in VALID_STATUSES:
            return Response(
                {"error": f"Invalid status. Must be one of {sorted(VALID_STATUSES)}."},
                status=drf_status.HTTP_400_BAD_REQUEST,
            )
        req.status = new_status
        # If moving out of pending, stamp reviewed_at and (optionally) reviewer
        if new_status in ("approved", "rejected"):
            req.reviewed_at = timezone.now()
            if request.user and request.user.is_authenticated:
                req.reviewer = request.user

    if "reviewer_notes" in data:
        req.reviewer_notes = str(data["reviewer_notes"]).strip()

    req.save()
    return Response(_serialize(req))
