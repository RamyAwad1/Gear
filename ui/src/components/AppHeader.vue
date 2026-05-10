<script setup lang="ts">
import { computed } from 'vue'
import { usePredictionsStore } from '@/stores/predictions'

const store = usePredictionsStore()

/**
 * Show the predicted semester once we have a result; otherwise fall back
 * to a static "Current Semester" placeholder so the header isn't empty.
 */
const semesterDisplay = computed(() => {
  if (store.result) return formatSemester(store.result.semester_predicted)
  return null
})

function formatSemester(label: string): string {
  // "2025_Fall" -> "Fall 2025"
  const [year, term] = label.split('_')
  if (!year || !term) return label
  return `${term} ${year}`
}
</script>

<template>
  <header class="border-b border-slate-200 bg-white">
    <div class="mx-auto flex max-w-7xl items-center justify-between px-8 py-6">
      <div>
        <h1 class="text-2xl font-semibold tracking-tight text-slate-900">
          HTU Enrollment Prediction System
        </h1>
        <p class="mt-1 text-sm text-slate-500">
          AI-Powered Course Planning &amp; Resource Allocation
        </p>
      </div>

      <div class="flex items-center gap-4">
        <div v-if="semesterDisplay" class="text-right">
          <div class="text-xs uppercase tracking-wider text-slate-400">
            Predicting
          </div>
          <div class="text-sm font-medium text-slate-900">
            {{ semesterDisplay }}
          </div>
        </div>
        <button
          type="button"
          aria-label="Settings"
          class="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
        >
          <!-- Lightweight inline gear icon -->
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="h-5 w-5"
          >
            <circle cx="12" cy="12" r="3" />
            <path
              d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
            />
          </svg>
        </button>
      </div>
    </div>
  </header>
</template>
