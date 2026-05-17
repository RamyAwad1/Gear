<script setup lang="ts">
/**
 * Substitute Requests view.
 *
 * A single page with a role toggle at the top — Student / Reviewer.
 *
 *   Student view:
 *     1. Enter your student ID.
 *     2. See all your past substitute requests with status badges.
 *     3. Submit a new request (original course, substitute course, reason).
 *
 *   Reviewer view:
 *     1. See all requests (filterable by status).
 *     2. Approve or reject a pending request with optional notes.
 *
 * Auth is intentionally absent — for the capstone demo, anyone can
 * "be" any student or reviewer. In production this view would sit
 * behind Django session auth or JWT.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useSubstituteRequestsStore } from '@/stores/substituteRequests'
import { usePredictionsStore } from '@/stores/predictions'
import type { SubstituteRequest, SubstituteRequestStatus } from '@/types'

type Role = 'student' | 'reviewer'

const store = useSubstituteRequestsStore()
const predictions = usePredictionsStore()

const role = ref<Role>('student')

// ── Student-side state ──────────────────────────────────────────────
const studentIdInput = ref('')
const showForm = ref(false)
const formOriginalCode = ref('')
const formSubstituteCode = ref('')
const formReason = ref('')
const formError = ref('')

// Use the most recent prediction's course list to power dropdowns.
const courseOptions = computed(() => {
  if (!predictions.result) return [] as { code: string; title: string }[]
  return predictions.result.courses
    .map((c) => ({ code: c.course_code, title: c.course_title }))
    .sort((a, b) => a.code.localeCompare(b.code))
})

function findTitle(code: string): string {
  return courseOptions.value.find((c) => c.code === code)?.title || ''
}

async function loadStudent() {
  formError.value = ''
  await store.loadStudentRequests(studentIdInput.value.trim())
}

function openForm() {
  formOriginalCode.value = ''
  formSubstituteCode.value = ''
  formReason.value = ''
  formError.value = ''
  showForm.value = true
}

async function submitForm() {
  formError.value = ''
  if (!studentIdInput.value.trim()) {
    formError.value = 'Enter your student ID first.'
    return
  }
  if (!formOriginalCode.value.trim()) {
    formError.value = 'Pick the course you need to substitute.'
    return
  }
  if (!formSubstituteCode.value.trim()) {
    formError.value = 'Pick the course you want to take instead.'
    return
  }
  if (formOriginalCode.value === formSubstituteCode.value) {
    formError.value = 'Original and substitute must be different courses.'
    return
  }
  if (!formReason.value.trim()) {
    formError.value = 'Briefly explain why.'
    return
  }

  try {
    await store.submitRequest({
      student_id:               studentIdInput.value.trim(),
      original_course_code:     formOriginalCode.value.trim(),
      original_course_title:    findTitle(formOriginalCode.value.trim()),
      substitute_course_code:   formSubstituteCode.value.trim(),
      substitute_course_title:  findTitle(formSubstituteCode.value.trim()),
      reason:                   formReason.value.trim(),
    })
    showForm.value = false
  } catch (e: unknown) {
    formError.value = e instanceof Error ? e.message : 'Submission failed.'
  }
}

// ── Reviewer-side state ─────────────────────────────────────────────
const reviewerFilter = ref<'all' | SubstituteRequestStatus>('pending')
const reviewerNotesDraft = ref<Record<number, string>>({})

async function reloadReviewer() {
  if (reviewerFilter.value === 'all') {
    await store.loadAllRequests()
  } else {
    await store.loadAllRequests(reviewerFilter.value)
  }
}

async function approve(req: SubstituteRequest) {
  await store.reviewRequest(req.id, {
    status: 'approved',
    reviewer_notes: reviewerNotesDraft.value[req.id] || '',
  })
  reviewerNotesDraft.value[req.id] = ''
  reloadReviewer()
}

async function reject(req: SubstituteRequest) {
  await store.reviewRequest(req.id, {
    status: 'rejected',
    reviewer_notes: reviewerNotesDraft.value[req.id] || '',
  })
  reviewerNotesDraft.value[req.id] = ''
  reloadReviewer()
}

watch(role, async (r) => {
  if (r === 'reviewer') {
    await reloadReviewer()
  }
})
watch(reviewerFilter, reloadReviewer)

onMounted(() => {
  if (role.value === 'reviewer') reloadReviewer()
})

// ── Display helpers ─────────────────────────────────────────────────
const statusStyles: Record<SubstituteRequestStatus, string> = {
  pending:  'bg-amber-50 text-amber-700 ring-amber-600/20',
  approved: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20',
  rejected: 'bg-red-50 text-red-700 ring-red-600/20',
}

function fmtDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  })
}
</script>

<template>
  <div class="space-y-6">
    <!-- Header + role toggle -->
    <div class="flex items-end justify-between">
      <div>
        <h2 class="text-xl font-semibold text-slate-900">Substitute Requests</h2>
        <p class="mt-1 text-sm text-slate-500">
          Submit and track requests to substitute one required graduation
          course with another.
        </p>
      </div>
      <div class="inline-flex rounded-lg bg-slate-100 p-1 text-sm">
        <button
          @click="role = 'student'"
          :class="[
            'rounded-md px-4 py-1.5 font-medium transition',
            role === 'student' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900',
          ]"
        >
          Student
        </button>
        <button
          @click="role = 'reviewer'"
          :class="[
            'rounded-md px-4 py-1.5 font-medium transition',
            role === 'reviewer' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900',
          ]"
        >
          Reviewer
          <span
            v-if="store.pendingCount > 0"
            class="ml-1.5 rounded-full bg-amber-500 px-1.5 py-0.5 text-xs font-semibold text-white"
          >
            {{ store.pendingCount }}
          </span>
        </button>
      </div>
    </div>

    <!-- ─────────────────────────── STUDENT ROLE ───────────────────── -->
    <template v-if="role === 'student'">
      <!-- ID input -->
      <div class="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <label class="block text-sm font-medium text-slate-700">
          Your student ID
        </label>
        <div class="mt-2 flex gap-2">
          <input
            v-model="studentIdInput"
            type="text"
            placeholder="e.g. 202400001"
            class="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            @keyup.enter="loadStudent"
          />
          <button
            @click="loadStudent"
            :disabled="store.isLoading"
            class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-slate-300"
          >
            {{ store.isLoading ? 'Loading…' : 'Load my requests' }}
          </button>
          <button
            @click="openForm"
            :disabled="!studentIdInput.trim()"
            class="rounded-md border border-blue-600 px-4 py-2 text-sm font-medium text-blue-600 hover:bg-blue-50 disabled:cursor-not-allowed disabled:border-slate-300 disabled:text-slate-400"
          >
            + New request
          </button>
        </div>
        <p v-if="store.error" class="mt-2 text-sm text-red-600">{{ store.error }}</p>
      </div>

      <!-- Existing requests -->
      <div
        v-if="store.currentStudentId"
        class="rounded-xl border border-slate-200 bg-white shadow-sm"
      >
        <div class="border-b border-slate-100 px-6 py-5">
          <h3 class="text-base font-semibold text-slate-900">
            Requests for {{ store.currentStudentId }}
          </h3>
          <p class="mt-0.5 text-sm text-slate-500">
            {{ store.studentRequests.length }} total —
            {{ store.studentPendingCount }} pending review.
          </p>
        </div>

        <ul v-if="store.studentRequests.length > 0" class="divide-y divide-slate-100">
          <li
            v-for="req in store.studentRequests"
            :key="req.id"
            class="px-6 py-4"
          >
            <div class="flex items-start justify-between gap-4">
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2">
                  <span class="font-mono text-xs text-slate-500">#{{ req.id }}</span>
                  <span
                    class="inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset"
                    :class="statusStyles[req.status]"
                  >
                    {{ req.status }}
                  </span>
                  <span class="text-xs text-slate-400">
                    submitted {{ fmtDate(req.created_at) }}
                  </span>
                </div>
                <div class="mt-2 text-sm text-slate-700">
                  <span class="font-medium text-slate-900">
                    {{ req.original_course_code }}
                  </span>
                  {{ req.original_course_title ? '— ' + req.original_course_title : '' }}
                  <span class="px-2 text-slate-400">→</span>
                  <span class="font-medium text-slate-900">
                    {{ req.substitute_course_code }}
                  </span>
                  {{ req.substitute_course_title ? '— ' + req.substitute_course_title : '' }}
                </div>
                <p class="mt-1 text-sm text-slate-600 whitespace-pre-wrap">
                  {{ req.reason }}
                </p>
                <p
                  v-if="req.reviewer_notes"
                  class="mt-2 rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-700"
                >
                  <span class="font-semibold">Reviewer note:</span>
                  {{ req.reviewer_notes }}
                </p>
              </div>
            </div>
          </li>
        </ul>

        <div v-else class="p-12 text-center text-sm text-slate-500">
          No substitute requests on file. Click <strong>+ New request</strong>
          to submit one.
        </div>
      </div>

      <!-- New-request modal -->
      <div
        v-if="showForm"
        class="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
        @click.self="showForm = false"
      >
        <div class="w-full max-w-lg rounded-xl border border-slate-200 bg-white p-6 shadow-xl">
          <h3 class="text-lg font-semibold text-slate-900">New substitute request</h3>
          <p class="mt-1 text-sm text-slate-500">
            Filing for student <span class="font-mono">{{ studentIdInput }}</span>.
          </p>

          <div class="mt-5 space-y-4">
            <div>
              <label class="block text-sm font-medium text-slate-700">
                Course you need to graduate
              </label>
              <!-- If we have a course list from predictions, use a select.
                   Otherwise fall back to a free-text input. -->
              <select
                v-if="courseOptions.length > 0"
                v-model="formOriginalCode"
                class="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="">— Pick a course —</option>
                <option v-for="c in courseOptions" :key="c.code" :value="c.code">
                  {{ c.code }} — {{ c.title }}
                </option>
              </select>
              <input
                v-else
                v-model="formOriginalCode"
                type="text"
                placeholder="e.g. 0040303121"
                class="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-slate-700">
                Course you want to take instead
              </label>
              <select
                v-if="courseOptions.length > 0"
                v-model="formSubstituteCode"
                class="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="">— Pick a course —</option>
                <option v-for="c in courseOptions" :key="c.code" :value="c.code">
                  {{ c.code }} — {{ c.title }}
                </option>
              </select>
              <input
                v-else
                v-model="formSubstituteCode"
                type="text"
                placeholder="e.g. 0030303111"
                class="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-slate-700">Reason</label>
              <textarea
                v-model="formReason"
                rows="4"
                placeholder="Explain why you're asking for this substitution."
                class="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>

            <p v-if="formError" class="text-sm text-red-600">{{ formError }}</p>
          </div>

          <div class="mt-6 flex justify-end gap-2">
            <button
              @click="showForm = false"
              class="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              @click="submitForm"
              :disabled="store.isSubmitting"
              class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-slate-300"
            >
              {{ store.isSubmitting ? 'Submitting…' : 'Submit request' }}
            </button>
          </div>
        </div>
      </div>
    </template>

    <!-- ─────────────────────────── REVIEWER ROLE ──────────────────── -->
    <template v-else>
      <div class="rounded-xl border border-slate-200 bg-white shadow-sm">
        <div
          class="flex items-center justify-between border-b border-slate-100 px-6 py-5"
        >
          <div>
            <h3 class="text-base font-semibold text-slate-900">
              All substitute requests
            </h3>
            <p class="mt-0.5 text-sm text-slate-500">
              Approve or reject pending requests from students.
            </p>
          </div>
          <select
            v-model="reviewerFilter"
            class="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
            <option value="all">All</option>
          </select>
        </div>

        <div v-if="store.isLoading" class="p-12 text-center text-sm text-slate-500">
          Loading…
        </div>
        <div
          v-else-if="store.allRequests.length === 0"
          class="p-12 text-center text-sm text-slate-500"
        >
          No requests match the current filter.
        </div>
        <ul v-else class="divide-y divide-slate-100">
          <li v-for="req in store.allRequests" :key="req.id" class="p-6">
            <div class="flex items-start justify-between gap-4">
              <div class="min-w-0 flex-1">
                <div class="flex items-center gap-2">
                  <span class="font-mono text-xs text-slate-500">#{{ req.id }}</span>
                  <span class="font-mono text-sm font-medium text-slate-900">
                    {{ req.student_id }}
                  </span>
                  <span
                    class="inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset"
                    :class="statusStyles[req.status]"
                  >
                    {{ req.status }}
                  </span>
                  <span class="text-xs text-slate-400">
                    {{ fmtDate(req.created_at) }}
                  </span>
                </div>
                <div class="mt-2 text-sm text-slate-700">
                  <span class="font-medium text-slate-900">
                    {{ req.original_course_code }}
                  </span>
                  <span v-if="req.original_course_title" class="text-slate-500">
                    — {{ req.original_course_title }}
                  </span>
                  <span class="px-2 text-slate-400">→</span>
                  <span class="font-medium text-slate-900">
                    {{ req.substitute_course_code }}
                  </span>
                  <span v-if="req.substitute_course_title" class="text-slate-500">
                    — {{ req.substitute_course_title }}
                  </span>
                </div>
                <p class="mt-1 text-sm text-slate-600 whitespace-pre-wrap">
                  {{ req.reason }}
                </p>
                <p
                  v-if="req.reviewer_notes && req.status !== 'pending'"
                  class="mt-2 rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-700"
                >
                  <span class="font-semibold">Note ({{ req.reviewer || 'reviewer' }}):</span>
                  {{ req.reviewer_notes }}
                </p>
              </div>
            </div>

            <!-- Action row for pending only -->
            <div
              v-if="req.status === 'pending'"
              class="mt-4 flex flex-col gap-2 rounded-md bg-slate-50 p-3 sm:flex-row sm:items-end"
            >
              <div class="flex-1">
                <label class="block text-xs font-medium uppercase tracking-wide text-slate-500">
                  Reviewer notes (optional)
                </label>
                <input
                  v-model="reviewerNotesDraft[req.id]"
                  type="text"
                  placeholder="Comment visible to the student"
                  class="mt-1 w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div class="flex gap-2">
                <button
                  @click="approve(req)"
                  :disabled="store.isSubmitting"
                  class="rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:bg-slate-300"
                >
                  Approve
                </button>
                <button
                  @click="reject(req)"
                  :disabled="store.isSubmitting"
                  class="rounded-md bg-red-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-red-700 disabled:bg-slate-300"
                >
                  Reject
                </button>
              </div>
            </div>
          </li>
        </ul>
      </div>
    </template>
  </div>
</template>
