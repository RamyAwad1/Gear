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

export interface CoursePrediction {
  course_code: string
  course_title: string
  predicted_enrollment: number
  sections_needed: number
  /** null when no last-semester record exists for this course */
  last_semester_actual: number | null
  status: CourseStatus
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
  traceback?: string
}
