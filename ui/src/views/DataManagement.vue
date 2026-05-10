<script setup lang="ts">
/**
 * Data Management view.
 *
 * Lets the user pick the two CSVs (htu_enrollments.csv + htu_students.csv),
 * trigger the prediction pipeline, and watch a friendly progress indicator
 * while the (slow, cold-start) request runs. On success, the parent App
 * switches the active tab to Predictions automatically.
 */

import { computed, ref } from 'vue'
import FileSlot from '@/components/FileSlot.vue'
import { usePredictionsStore } from '@/stores/predictions'

const emit = defineEmits<{ done: [] }>()
const store = usePredictionsStore()

const enrollmentsFile = ref<File | null>(null)
const studentsFile = ref<File | null>(null)

const enrollmentsInput = ref<HTMLInputElement | null>(null)
const studentsInput = ref<HTMLInputElement | null>(null)

const canRun = computed(
  () => !!enrollmentsFile.value && !!studentsFile.value && !store.loading,
)

function handleFile(event: Event, target: 'enrollments' | 'students') {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0] ?? null
  if (target === 'enrollments') enrollmentsFile.value = file
  else studentsFile.value = file
}

function clearEnrollments() {
  enrollmentsFile.value = null
  if (enrollmentsInput.value) enrollmentsInput.value.value = ''
}

function clearStudents() {
  studentsFile.value = null
  if (studentsInput.value) studentsInput.value.value = ''
}

async function run() {
  if (!enrollmentsFile.value || !studentsFile.value) return
  try {
    await store.runPrediction(enrollmentsFile.value, studentsFile.value)
    emit('done')
  } catch {
    // Error is already in the store and surfaced in the template.
  }
}

function reset() {
  clearEnrollments()
  clearStudents()
  store.clear()
}
</script>

<template>
  <div class="space-y-6">
    <div>
      <h2 class="text-xl font-semibold text-slate-900">
        Data Upload &amp; Management
      </h2>
      <p class="mt-1 text-sm text-slate-500">
        Upload last semester's records to forecast next semester's enrollment.
      </p>
    </div>

    <div class="rounded-xl border border-slate-200 bg-white p-8 shadow-sm">
      <div class="grid gap-6 md:grid-cols-2">
        <FileSlot
          label="Enrollment history"
          hint="htu_enrollments.csv"
          :file="enrollmentsFile"
          :disabled="store.loading"
          @pick="enrollmentsInput?.click()"
          @clear="clearEnrollments"
        />
        <input
          ref="enrollmentsInput"
          type="file"
          accept=".csv"
          class="hidden"
          @change="handleFile($event, 'enrollments')"
        />

        <FileSlot
          label="Student records"
          hint="htu_students.csv"
          :file="studentsFile"
          :disabled="store.loading"
          @pick="studentsInput?.click()"
          @clear="clearStudents"
        />
        <input
          ref="studentsInput"
          type="file"
          accept=".csv"
          class="hidden"
          @change="handleFile($event, 'students')"
        />
      </div>

      <div
        class="mt-8 flex items-center justify-between border-t border-slate-100 pt-6"
      >
        <div class="text-sm text-slate-500">
          <span v-if="store.loading">
            Running predictions — this may take 30–60 seconds on first run
            while models load.
          </span>
          <span v-else-if="store.error" class="text-red-600">
            {{ store.error }}
          </span>
          <span v-else-if="store.hasResult" class="text-emerald-600">
            Predictions ready for {{ store.result!.semester_predicted }}.
          </span>
          <span v-else>
            Both files are required. Pick them above to enable “Run Predictions”.
          </span>
        </div>

        <div class="flex gap-3">
          <button
            type="button"
            class="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            :disabled="store.loading"
            @click="reset"
          >
            Reset
          </button>
          <button
            type="button"
            class="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-300"
            :disabled="!canRun"
            @click="run"
          >
            <svg
              v-if="store.loading"
              xmlns="http://www.w3.org/2000/svg"
              class="h-4 w-4 animate-spin"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="3"
                stroke-opacity="0.25"
              />
              <path
                d="M22 12a10 10 0 0 1-10 10"
                stroke="currentColor"
                stroke-width="3"
                stroke-linecap="round"
              />
            </svg>
            {{ store.loading ? 'Running…' : 'Run Predictions' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
