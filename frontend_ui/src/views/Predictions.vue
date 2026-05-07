<script setup lang="ts">
/**
 * Predictions view.
 *
 * Renders the per-course prediction table. Pulls everything from the
 * store, so this component is purely presentational. When there's no
 * result yet, shows a friendly empty state pointing the user to the
 * Data Management tab.
 */

import { computed } from 'vue'
import SummaryCard from '@/components/SummaryCard.vue'
import { usePredictionsStore } from '@/stores/predictions'
import type { CourseStatus } from '@/types'

const emit = defineEmits<{ 'go-upload': [] }>()
const store = usePredictionsStore()

// Colored pill classes per status — using ring-1 inset to keep the
// pill outline crisp without adding a heavy border.
const statusStyles: Record<CourseStatus, string> = {
  optimal: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20',
  warning: 'bg-amber-50 text-amber-700 ring-amber-600/20',
  critical: 'bg-red-50 text-red-700 ring-red-600/20',
}

// Pretty-print "2025_Fall" -> "Fall 2025" for the heading.
const semesterDisplay = computed(() => {
  if (!store.result) return ''
  const [year, term] = store.result.semester_predicted.split('_')
  return year && term ? `${term} ${year}` : store.result.semester_predicted
})

function formatActual(n: number | null): string {
  return n === null ? 'N/A' : `${n}`
}
</script>

<template>
  <!-- Empty state -->
  <div
    v-if="!store.hasResult"
    class="rounded-xl border border-slate-200 bg-white p-12 text-center shadow-sm"
  >
    <div
      class="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-slate-100"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
        class="h-6 w-6 text-slate-400"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          d="M3 13.5 12 3l9 10.5M5 11v9a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1v-9"
        />
      </svg>
    </div>
    <h3 class="mt-4 text-base font-semibold text-slate-900">
      No predictions yet
    </h3>
    <p class="mt-1 text-sm text-slate-500">
      Upload the enrollment and student CSVs to forecast next semester's
      course demand.
    </p>
    <button
      type="button"
      class="mt-6 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700"
      @click="emit('go-upload')"
    >
      Go to Data Management
    </button>
  </div>

  <!-- Result -->
  <div v-else class="space-y-6">
    <!-- Summary row -->
    <div class="grid gap-4 md:grid-cols-4">
      <SummaryCard
        label="Predicted Enrollment"
        :value="store.result!.totals.predicted_enrollment.toLocaleString()"
      />
      <SummaryCard
        label="Sections Needed"
        :value="store.result!.totals.sections_needed.toString()"
      />
      <SummaryCard
        label="Courses"
        :value="store.result!.totals.courses.toString()"
      />
      <SummaryCard
        label="Students Processed"
        :value="store.result!.totals.students_processed.toLocaleString()"
      />
    </div>

    <!-- Table card -->
    <div class="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div
        class="flex items-center justify-between border-b border-slate-100 px-6 py-5"
      >
        <div>
          <h3 class="text-base font-semibold text-slate-900">
            Course Predictions — {{ semesterDisplay }}
          </h3>
          <p class="mt-0.5 text-sm text-slate-500">
            AI-powered enrollment forecasting
          </p>
        </div>
        <div class="text-xs text-slate-400">
          <span v-if="store.lastRunSeconds !== null">
            Generated in {{ store.lastRunSeconds.toFixed(1) }}s
          </span>
        </div>
      </div>

      <div class="overflow-x-auto">
        <table class="min-w-full">
          <thead>
            <tr class="border-b border-slate-100 bg-slate-50/50">
              <th
                class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
              >
                Course
              </th>
              <th
                class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
              >
                Predicted Enrollment
              </th>
              <th
                class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
              >
                Sections Needed
              </th>
              <th
                class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
              >
                Last Semester
              </th>
              <th
                class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
              >
                Status
              </th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            <tr
              v-for="course in store.coursesSortedByDemand"
              :key="course.course_code"
              class="transition hover:bg-slate-50/50"
            >
              <td class="px-6 py-4">
                <div class="font-medium text-slate-900">
                  {{ course.course_code }}
                </div>
                <div v-if="course.course_title" class="text-xs text-slate-500">
                  {{ course.course_title }}
                </div>
              </td>
              <td class="px-6 py-4 text-sm text-slate-700">
                {{ course.predicted_enrollment }} students
              </td>
              <td class="px-6 py-4 text-sm text-slate-700">
                {{ course.sections_needed }}
                {{ course.sections_needed === 1 ? 'section' : 'sections' }}
              </td>
              <td class="px-6 py-4 text-sm text-slate-700">
                {{ formatActual(course.last_semester_actual) }}
              </td>
              <td class="px-6 py-4">
                <span
                  class="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset"
                  :class="statusStyles[course.status]"
                >
                  {{ course.status }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
