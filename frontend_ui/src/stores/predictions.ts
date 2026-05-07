/**
 * Predictions store.
 *
 * Holds the most recent prediction result so the Data Management view can
 * trigger a run and the Predictions view can render the table without
 * either component owning the state.
 */

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { predictUpload, type PredictUploadOptions } from '@/api/client'
import type { PredictionResponse } from '@/types'

export const usePredictionsStore = defineStore('predictions', () => {
  // ── State ───────────────────────────────────────────────────────────────
  const result = ref<PredictionResponse | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  /** Wall-clock seconds the most recent successful run took. */
  const lastRunSeconds = ref<number | null>(null)

  // ── Getters ─────────────────────────────────────────────────────────────
  const hasResult = computed(() => result.value !== null)

  const coursesSortedByDemand = computed(() => {
    if (!result.value) return []
    return [...result.value.courses].sort(
      (a, b) => b.predicted_enrollment - a.predicted_enrollment,
    )
  })

  // ── Actions ─────────────────────────────────────────────────────────────
  async function runPrediction(
    enrollmentsFile: File,
    studentsFile: File,
    opts: PredictUploadOptions = {},
  ) {
    loading.value = true
    error.value = null
    const startedAt = performance.now()
    try {
      const data = await predictUpload(enrollmentsFile, studentsFile, opts)
      result.value = data
      lastRunSeconds.value = (performance.now() - startedAt) / 1000
      return data
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
      throw err
    } finally {
      loading.value = false
    }
  }

  function clear() {
    result.value = null
    error.value = null
    lastRunSeconds.value = null
  }

  return {
    // state
    result,
    loading,
    error,
    lastRunSeconds,
    // getters
    hasResult,
    coursesSortedByDemand,
    // actions
    runPrediction,
    clear,
  }
})
