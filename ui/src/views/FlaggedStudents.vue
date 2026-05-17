<script setup lang="ts">
/**
 * Flagged Students view.
 *
 * Lists students that the prediction pipeline (or an advisor) flagged
 * for review. Supports search, status filter (all / unresolved /
 * resolved), and a one-click resolve toggle. Mock-backed today; the
 * store swaps to a real fetch once the endpoint exists.
 */

import { computed, onMounted, ref } from 'vue'
import { useFlaggedStudentsStore } from '@/stores/flaggedStudents'
import type { FlagReason } from '@/types'

const store = useFlaggedStudentsStore()

const search = ref('')
const filter = ref<'all' | 'unresolved' | 'resolved'>('unresolved')

onMounted(() => store.load())

const reasonLabels: Record<FlagReason, string> = {
  low_gpa: 'Low GPA',
  missing_prereq: 'Missing prerequisite',
  graduation_risk: 'Graduation risk',
  excess_credits: 'Excess credit hours',
  repeating_failure: 'Repeating failed course',
  probation: 'Academic probation',
}

const reasonStyles: Record<FlagReason, string> = {
  low_gpa: 'bg-red-50 text-red-700 ring-red-600/20',
  missing_prereq: 'bg-amber-50 text-amber-700 ring-amber-600/20',
  graduation_risk: 'bg-red-50 text-red-700 ring-red-600/20',
  excess_credits: 'bg-amber-50 text-amber-700 ring-amber-600/20',
  repeating_failure: 'bg-red-50 text-red-700 ring-red-600/20',
  probation: 'bg-red-50 text-red-700 ring-red-600/20',
}

const visible = computed(() => {
  const q = search.value.trim().toLowerCase()
  return store.students.filter((s) => {
    if (filter.value === 'unresolved' && s.is_resolved) return false
    if (filter.value === 'resolved' && !s.is_resolved) return false
    if (!q) return true
    return (
      s.student_id.toLowerCase().includes(q) ||
      s.student_name.toLowerCase().includes(q) ||
      s.major.toLowerCase().includes(q) ||
      reasonLabels[s.reason].toLowerCase().includes(q)
    )
  })
})

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

const filterTabs = [
  { key: 'unresolved' as const, label: 'Unresolved' },
  { key: 'resolved' as const, label: 'Resolved' },
  { key: 'all' as const, label: 'All' },
]
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-end justify-between">
      <div>
        <h2 class="text-xl font-semibold text-slate-900">Flagged Students</h2>
        <p class="mt-1 text-sm text-slate-500">
          {{ store.unresolved.length }} unresolved ·
          {{ store.resolved.length }} resolved
        </p>
      </div>
    </div>

    <!-- Filter strip -->
    <div
      class="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm md:flex-row md:items-center md:justify-between"
    >
      <!-- Search -->
      <div class="relative flex-1 md:max-w-sm">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          class="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
        >
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-3.5-3.5" />
        </svg>
        <input
          v-model="search"
          type="search"
          placeholder="Search by ID, name, major, or reason"
          class="w-full rounded-lg border border-slate-300 bg-white py-2 pl-9 pr-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      <!-- Filter pills -->
      <div class="flex gap-1 rounded-lg bg-slate-100 p-1">
        <button
          v-for="tab in filterTabs"
          :key="tab.key"
          type="button"
          class="rounded-md px-3 py-1.5 text-sm font-medium transition"
          :class="
            filter === tab.key
              ? 'bg-white text-slate-900 shadow-sm'
              : 'text-slate-500 hover:text-slate-700'
          "
          @click="filter = tab.key"
        >
          {{ tab.label }}
        </button>
      </div>
    </div>

    <!-- Empty state for the current filter -->
    <div
      v-if="visible.length === 0"
      class="rounded-xl border border-slate-200 bg-white p-12 text-center shadow-sm"
    >
      <div
        class="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-emerald-50"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.8"
          class="h-6 w-6 text-emerald-600"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            d="M5 13l4 4L19 7"
          />
        </svg>
      </div>
      <h3 class="mt-4 text-base font-semibold text-slate-900">
        Nothing to review
      </h3>
      <p class="mt-1 text-sm text-slate-500">
        No students match the current filter.
      </p>
    </div>

    <!-- Table -->
    <div
      v-else
      class="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
    >
      <table class="min-w-full">
        <thead>
          <tr class="border-b border-slate-100 bg-slate-50/50">
            <th
              class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
            >
              Student
            </th>
            <th
              class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
            >
              Major
            </th>
            <th
              class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
            >
              Reason
            </th>
            <th
              class="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500"
            >
              Flagged
            </th>
            <th
              class="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-slate-500"
            >
              Action
            </th>
          </tr>
        </thead>
        <tbody class="divide-y divide-slate-100">
          <tr
            v-for="s in visible"
            :key="s.id"
            class="transition hover:bg-slate-50/50"
            :class="s.is_resolved ? 'opacity-60' : ''"
          >
            <td class="px-6 py-4">
              <div class="font-medium text-slate-900">
                {{ s.student_name }}
              </div>
              <div class="text-xs text-slate-500">{{ s.student_id }}</div>
            </td>
            <td class="px-6 py-4">
              <div class="text-sm text-slate-700">{{ s.major }}</div>
              <div class="text-xs text-slate-500">{{ s.degree_type }}</div>
            </td>
            <td class="px-6 py-4">
              <span
                class="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset"
                :class="reasonStyles[s.reason]"
              >
                {{ reasonLabels[s.reason] }}
              </span>
              <div class="mt-1 max-w-md text-xs text-slate-500">
                {{ s.reason_detail }}
              </div>
            </td>
            <td class="px-6 py-4 text-sm text-slate-600">
              {{ formatDate(s.flagged_at) }}
            </td>
            <td class="px-6 py-4 text-right">
              <button
                type="button"
                class="text-xs font-medium transition"
                :class="
                  s.is_resolved
                    ? 'text-slate-500 hover:text-slate-700'
                    : 'text-blue-600 hover:text-blue-700'
                "
                @click="store.toggleResolved(s.id)"
              >
                {{ s.is_resolved ? 'Reopen' : 'Mark resolved' }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
