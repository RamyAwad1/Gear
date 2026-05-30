/**
 * Types mirroring the response shape of the /api/predict/upload/ endpoint
 * defined in core/api_views.py. Keep these in sync with the backend if the
 * response changes.
 */

/** Identifier for the active tab in the navigation strip. */
export type TabKey =
  | 'dashboard'
  | 'predictions'
  | 'data-management'
  | 'flagged-students'
  | 'elective-requests'

export type CourseStatus = 'optimal' | 'warning' | 'critical'

/** Why an individual student is in the predicted-enrolment list for a course. */
export type PredictedStudentReason =
  | 'passed_predecessor'
  | 'retake_candidate'
  | 'plan_match'

/** A single named student the model expects to enrol in a given course. */
export interface PredictedStudent {
  student_id: string
  major: string
  degree_type: string
  plan_key: string
  admission_year: number | null
  current_year: number
  cum_hrs: number
  gpa: number | null
  reason: PredictedStudentReason
  /** 0..1 — historical conversion rate behind this candidate's signal. */
  confidence: number
}

export interface CoursePrediction {
  course_code: string
  course_title: string
  predicted_enrollment: number
  sections_needed: number
  /** null when no last-semester record exists for this course */
  last_semester_actual: number | null
  status: CourseStatus
  /**
   * Students the model can name as likely enrollees. May be shorter than
   * predicted_enrollment for courses whose forecast is dominated by the
   * time-series component (which doesn't identify specific students).
   */
  predicted_students: PredictedStudent[]
  /** Length of predicted_students; populated by the backend for convenience. */
  identified_count: number
}

export interface PredictionTotals {
  courses: number
  predicted_enrollment: number
  sections_needed: number
  students_processed: number
}

export interface PredictionResponse {
  semester_predicted: string
  /** null when not provided in the request and the backend used its default */
  section_cap: number | null
  /** null when not provided in the request and the backend used its default */
  buffer_pct: number | null
  totals: PredictionTotals
  courses: CoursePrediction[]
}

/** Shape of a 4xx/5xx error body from the API. */
export interface ApiError {
  error: string
  detail?: string
  missing_columns?: string[]
  missing_fields?: string[]
  unknown_fields?: string[]
  traceback?: string
}

// ─── Flagged Students ────────────────────────────────────────────────────────

/** Reason categories the advisor sees for a flagged student. */
export type FlagReason =
  | 'low_gpa'
  | 'missing_prereq'
  | 'graduation_risk'
  | 'excess_credits'
  | 'repeating_failure'
  | 'probation'

export interface FlaggedStudent {
  id: number
  student_id: string
  student_name: string
  major: string
  degree_type: 'Bachelor' | 'Technical'
  reason: FlagReason
  reason_detail: string
  flagged_at: string
  is_resolved: boolean
}

// ─── Elective Requests ───────────────────────────────────────────────────────

export type ElectiveRequestStatus = 'pending' | 'approved' | 'rejected'

/** Aggregated demand for a single elective course. */
export interface ElectiveRequest {
  id: number
  course_code: string
  course_title: string
  department: string
  request_count: number
  status: ElectiveRequestStatus
  last_requested_at: string
}