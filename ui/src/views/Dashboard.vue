<script setup lang="ts">
/**
 * Dashboard view.
 *
 * High-level overview built entirely on top of the predictions store —
 * no separate fetch. Renders an empty-state pointer to Data Management
 * when there's no result yet; otherwise breaks the prediction down into
 * status mix, biggest growth courses, and largest sections.
 */

import { computed } from 'vue'
import SummaryCard from '@/components/SummaryCard.vue'
import { usePredictionsStore } from '@/stores/predictions'
import type { CoursePrediction, CourseStatus } from '@/types'

const emit = defineEmits<{
  'go-upload': []
  'go-predictions': []
}>()

const store = usePredictionsStore()

// "2025_Fall" -> "Fall 2025"
const semesterDisplay = computed(() => {
  if (!store.result) return ''
  const [year, term] = store.result.semester_predicted.split('_')
  return year && term ? `${term} ${year}` : store.result.semester_predicted
})

// Status mix counts.
const statusCounts = computed(() => {
  const counts: Record<CourseStatus, number> = {
    optimal: 0,
    warning: 0,
    critical: 0,
  }
  if (!store.result) return counts
  for (const c of store.result.courses) counts[c.status]++
  return counts
})

const totalCourses = computed(() => store.result?.totals.courses ?? 0)

// Pct of courses that need attention (warning + critical).
const attentionPct = computed(() => {
  if (!totalCourses.value) return 0
  return Math.round(
    ((statusCounts.value.warning + statusCounts.value.critical) /
      totalCourses.value) *
      100,
  )
})

// Total enrollment growth vs last semester (only over courses where we
// actually have a last-semester actual to compare against).
const lastSemesterTotal = computed(() => {
  if (!store.result) return 0
  return store.result.courses.reduce(
    (sum, c) => sum + (c.last_semester_actual ?? 0),
    0,
  )
})

const growthPct = computed(() => {
  if (!store.result || lastSemesterTotal.value === 0) return null
  const predicted = store.result.totals.predicted_enrollment
  return ((predicted - lastSemesterTotal.value) / lastSemesterTotal.value) * 100
})

// Top 5 courses by predicted demand.
const topByDemand = computed(() => {
  if (!store.result) return []
  return [...store.result.courses]
    .sort((a, b) => b.predicted_enrollment - a.predicted_enrollment)
    .slice(0, 5)
})

// Top 5 courses by absolute growth from last semester. Skip courses
// without a last-semester actual since growth is undefined for those.
const topByGrowth = computed(() => {
  if (!store.result) return []
  return store.result.courses
    .filter((c) => c.last_semester_actual !== null)
    .map((c) => ({
      course: c,
      delta: c.predicted_enrollment - (c.last_semester_actual as number),
    }))
    .sort((a, b) => b.delta - a.delta)
    .slice(0, 5)
})

const statusStyles: Record<CourseStatus, string> = {
  optimal: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20',
  warning: 'bg-amber-50 text-amber-700 ring-amber-600/20',
  critical: 'bg-red-50 text-red-700 ring-red-600/20',
}

const statusBars: Record<CourseStatus, string> = {
  optimal: 'bg-emerald-500',
  warning: 'bg-amber-500',
  critical: 'bg-red-500',
}

const statusOrder: CourseStatus[] = ['optimal', 'warning', 'critical']

function statusPct(s: CourseStatus): number {
  if (!totalCourses.value) return 0
  return (statusCounts.value[s] / totalCourses.value) * 100
}

function deltaLabel(course: CoursePrediction): string {
  if (course.last_semester_actual === null) return ''
  const delta = course.predicted_enrollment - course.last_semester_actual
  const sign = delta >= 0 ? '+' : ''
  return `${sign}${delta}`
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
          d="M3 3v18h18M7 14l4-4 4 4 5-6"
        />
      </svg>
    </div>
    <h3 class="mt-4 text-base font-semibold text-slate-900">
      Dashboard is empty
    </h3>
    <p class="mt-1 text-sm text-slate-500">
      Run a prediction to see enrollment forecasts, capacity warnings, and
      growth trends here.
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
    <!-- Header -->
    <div class="flex items-end justify-between">
      <div>
        <h2 class="text-xl font-semibold text-slate-900">Dashboard</h2>
        <p class="mt-1 text-sm text-slate-500">
          Overview for {{ semesterDisplay }} — generated from
          {{ store.result!.totals.students_processed.toLocaleString() }} student
          records.
        </p>
      </div>
      <button
        type="button"
        class="text-sm font-medium text-blue-600 transition hover:text-blue-700"
        @click="emit('go-predictions')"
      >
        View all predictions →
      </button>
    </div>

    <!-- KPI row -->
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
        label="Needs Attention"
        :value="`${statusCounts.warning + statusCounts.critical} (${attentionPct}%)`"
      />
      <SummaryCard
        label="Growth vs Last Sem"
        :value="
          growthPct === null
            ? '—'
            : `${growthPct >= 0 ? '+' : ''}${growthPct.toFixed(1)}%`
        "
      />
    </div>

    <!-- Two-column block: status mix + critical alerts -->
    <div class="grid gap-6 lg:grid-cols-5">
      <!-- Status mix -->
      <div
        class="rounded-xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2"
      >
        <h3 class="text-base font-semibold text-slate-900">
          Capacity status mix
        </h3>
        <p class="mt-0.5 text-sm text-slate-500">
          {{ totalCourses }} courses across all majors.
        </p>

        <!-- Stacked bar -->
        <div
          class="mt-5 flex h-2 w-full overflow-hidden rounded-full bg-slate-100"
        >
          <div
            v-for="s in statusOrder"
            :key="s"
            :class="statusBars[s]"
            :style="{ width: `${statusPct(s)}%` }"
          />
        </div>

        <!-- Legend rows -->
        <ul class="mt-5 space-y-3">
          <li
            v-for="s in statusOrder"
            :key="s"
            class="flex items-center justify-between text-sm"
          >
            <span class="flex items-center gap-2">
              <span
                class="inline-block h-2 w-2 rounded-full"
                :class="statusBars[s]"
              />
              <span class="capitalize text-slate-700">{{ s }}</span>
            </span>
            <span class="font-medium text-slate-900">
              {{ statusCounts[s] }}
              <span class="ml-1 text-xs font-normal text-slate-400">
                ({{ statusPct(s).toFixed(0) }}%)
              </span>
            </span>
          </li>
        </ul>
      </div>

      <!-- Critical capacity alerts -->
      <div
        class="rounded-xl border border-slate-200 bg-white shadow-sm lg:col-span-3"
      >
        <div class="border-b border-slate-100 px-6 py-5">
          <h3 class="text-base font-semibold text-slate-900">
            Largest growth vs last semester
          </h3>
          <p class="mt-0.5 text-sm text-slate-500">
            Courses where forecasted demand jumped the most.
          </p>
        </div>
        <ul class="divide-y divide-slate-100">
          <li
            v-for="row in topByGrowth"
            :key="row.course.course_code"
            class="flex items-center justify-between gap-4 px-6 py-3.5"
          >
            <div class="min-w-0 flex-1">
              <div class="font-medium text-slate-900">
                {{ row.course.course_code }}
              </div>
              <div
                v-if="row.course.course_title"
                class="truncate text-xs text-slate-500"
              >
                {{ row.course.course_title }}
              </div>
            </div>
            <div class="flex items-center gap-3">
              <span class="text-sm text-slate-700">
                {{ row.course.last_semester_actual }} →
                <span class="font-semibold text-slate-900">
                  {{ row.course.predicted_enrollment }}
                </span>
              </span>
              <span
                class="inline-flex w-16 justify-center rounded-md bg-blue-50 px-2 py-0.5 text-xs font-semibold text-blue-700 ring-1 ring-inset ring-blue-600/20"
              >
                {{ deltaLabel(row.course) }}
              </span>
              <span
                class="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset"
                :class="statusStyles[row.course.status]"
              >
                {{ row.course.status }}
              </span>
            </div>
          </li>
        </ul>
      </div>
    </div>

    <!-- Top demand list -->
    <div class="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div class="border-b border-slate-100 px-6 py-5">
        <h3 class="text-base font-semibold text-slate-900">
          Top courses by predicted demand
        </h3>
        <p class="mt-0.5 text-sm text-slate-500">
          Where the bulk of section planning will land for {{ semesterDisplay }}.
        </p>
      </div>
      <ul class="divide-y divide-slate-100">
        <li
          v-for="(course, idx) in topByDemand"
          :key="course.course_code"
          class="flex items-center gap-4 px-6 py-3.5"
        >
          <span
            class="flex h-7 w-7 flex-none items-center justify-center rounded-md bg-slate-50 text-xs font-semibold text-slate-600"
          >
            {{ idx + 1 }}
          </span>
          <div class="min-w-0 flex-1">
            <div class="font-medium text-slate-900">{{ course.course_code }}</div>
            <div
              v-if="course.course_title"
              class="truncate text-xs text-slate-500"
            >
              {{ course.course_title }}
            </div>
          </div>
          <div class="text-sm text-slate-700">
            <span class="font-semibold text-slate-900">
              {{ course.predicted_enrollment }}
            </span>
            students
          </div>
          <div class="text-sm text-slate-500">
            {{ course.sections_needed }}
            {{ course.sections_needed === 1 ? 'section' : 'sections' }}
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>
