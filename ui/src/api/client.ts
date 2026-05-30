// API client for the Django backend.

import type {
  CoursePrediction,
  PredictedStudent,
  PredictionResponse,
  PredictionTotals,
  ApiError,
  SubstituteRequest,
  SubstituteRequestCreate,
  SubstituteRequestUpdate,
  SubstituteRequestStatus,
} from "@/types";

export type {
  CoursePrediction,
  PredictedStudent,
  PredictionResponse,
  PredictionTotals,
  ApiError,
  SubstituteRequest,
  SubstituteRequestCreate,
  SubstituteRequestUpdate,
};

const API_BASE: string =
  (typeof window !== "undefined" &&
    (window as unknown as { __API_BASE__?: string }).__API_BASE__) ||
  "http://localhost:8000";

async function raiseFor(res: Response): Promise<never> {
  const err = (await res.json().catch(() => ({}))) as Partial<ApiError>;
  throw new Error(err.error || err.detail || `HTTP ${res.status}`);
}

// ─── Predictions ─────────────────────────────────────────────────────────────

export type PredictUploadOptions = {
  semester?: string;
  sectionCap?: number;
  bufferPct?: number;
};

export async function predictUpload(
  enrollmentsFile: File,
  studentsFile: File,
  options: PredictUploadOptions = {}
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("enrollments_file", enrollmentsFile);
  form.append("students_file", studentsFile);
  if (options.semester) form.append("semester", options.semester);
  if (options.sectionCap !== undefined)
    form.append("section_cap", String(options.sectionCap));
  if (options.bufferPct !== undefined)
    form.append("buffer_pct", String(options.bufferPct));

  const res = await fetch(`${API_BASE}/api/predict/upload/`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) await raiseFor(res);
  return res.json();
}

export async function healthCheck(): Promise<{ status: string; service: string }> {
  const res = await fetch(`${API_BASE}/api/health/`);
  if (!res.ok) await raiseFor(res);
  return res.json();
}

// ─── Substitute Requests ─────────────────────────────────────────────────────

type ListResponse = { results: SubstituteRequest[] };

export async function listSubstituteRequests(filter?: {
  studentId?: string;
  status?: SubstituteRequestStatus;
}): Promise<SubstituteRequest[]> {
  const params = new URLSearchParams();
  if (filter?.studentId) params.set("student_id", filter.studentId);
  if (filter?.status) params.set("status", filter.status);
  const qs = params.toString();
  const url = `${API_BASE}/api/substitute-requests/${qs ? "?" + qs : ""}`;
  const res = await fetch(url);
  if (!res.ok) await raiseFor(res);
  const body = (await res.json()) as ListResponse;
  return body.results;
}

export async function getSubstituteRequest(id: number): Promise<SubstituteRequest> {
  const res = await fetch(`${API_BASE}/api/substitute-requests/${id}/`);
  if (!res.ok) await raiseFor(res);
  return res.json();
}

export async function createSubstituteRequest(
  payload: SubstituteRequestCreate
): Promise<SubstituteRequest> {
  const res = await fetch(`${API_BASE}/api/substitute-requests/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) await raiseFor(res);
  return res.json();
}

export async function updateSubstituteRequest(
  id: number,
  payload: SubstituteRequestUpdate
): Promise<SubstituteRequest> {
  const res = await fetch(`${API_BASE}/api/substitute-requests/${id}/`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) await raiseFor(res);
  return res.json();
}