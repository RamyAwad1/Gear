/**
 * Tiny axios wrapper around the Gear API.
 *
 * Base URL is read from VITE_API_BASE (set in `.env`). When unset, falls
 * back to the Django dev server's default address so a fresh clone works
 * without configuration.
 */

import axios, { AxiosError } from 'axios'
import type { ApiError, PredictionResponse } from '@/types'

const baseURL = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

export const apiClient = axios.create({
  baseURL,
  // The first prediction request loads ~94 model .pkl files into memory and
  // can take 30-60s on a cold server. Give it a generous ceiling so axios
  // doesn't abort the request before Django finishes.
  timeout: 5 * 60 * 1000, // 5 minutes
})

export interface PredictUploadOptions {
  semester?: string
  sectionCap?: number
  bufferPct?: number
}

/**
 * POST both CSVs to /api/predict/upload/ and return the parsed response.
 *
 * Throws an Error with a human-readable `.message` derived from the
 * backend's error body (or a generic network error message) when the
 * request fails — components only need to catch one thing.
 */
export async function predictUpload(
  enrollmentsFile: File,
  studentsFile: File,
  opts: PredictUploadOptions = {},
): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('enrollments_file', enrollmentsFile)
  formData.append('students_file', studentsFile)
  if (opts.semester) formData.append('semester', opts.semester)
  if (opts.sectionCap !== undefined)
    formData.append('section_cap', String(opts.sectionCap))
  if (opts.bufferPct !== undefined)
    formData.append('buffer_pct', String(opts.bufferPct))

  try {
    const { data } = await apiClient.post<PredictionResponse>(
      '/api/predict/upload/',
      formData,
    )
    return data
  } catch (err) {
    throw new Error(formatAxiosError(err))
  }
}

function formatAxiosError(err: unknown): string {
  if (!axios.isAxiosError(err)) {
    return err instanceof Error ? err.message : 'Unknown error'
  }
  const axiosErr = err as AxiosError<ApiError>
  if (axiosErr.response?.data) {
    const body = axiosErr.response.data
    let msg = body.error ?? 'Request failed'
    if (body.missing_columns?.length) {
      msg += ` Missing columns: ${body.missing_columns.join(', ')}.`
    }
    if (body.detail) {
      msg += ` Detail: ${body.detail}`
    }
    return msg
  }
  if (axiosErr.code === 'ECONNABORTED') {
    return 'The server took too long to respond. The first prediction can take up to a minute as models load — try once more.'
  }
  if (axiosErr.code === 'ERR_NETWORK') {
    return `Could not reach the API at ${baseURL}. Is the Django server running?`
  }
  return axiosErr.message
}
