<script setup lang="ts">
/**
 * Elective Requests view.
 *
 * Department heads see which non-plan electives students are asking for,
 * sorted by demand. Approve/reject is a one-click pill swap. Mock-backed;
 * the store will swap to a real fetch when the endpoint exists.
 */

import { computed, onMounted, ref } from 'vue'
import SummaryCard from '@/components/SummaryCard.vue'
import { useElectiveRequestsStore } from '@/stores/electiveRequests'
import type { ElectiveRequestStatus } from '@/types'

const store = useElectiveRequestsStore()

const filter = ref<'all' | ElectiveRequestStatus>('pending')

onMounted(() => store.load())

const filterTabs = [
  { key: 'pending' as const, label: 'Pending' },
  { key: 'approved' as const, label: 'Approved' },
  { key: 'rejected' as const, label: 'Rejected' },
  { key: 'all' as const, label: 'All' },
]

const statusStyles: Record<ElectiveRequestStatus, string> = {
  pending: 'bg-amber-50 text-amber-700 ring-amber-600/20',
  approved: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20',
  rejected: 'bg-slate-100 text-slate-600 ring-slate-500/20',
}

const statusLabels: Record<ElectiveRequestStatus, string> = {
  pending: 'Pending',
  approved: 'Approved',
  rejected: 'Rejected',
}

const visible = computed(() => {
  const list =
    filter.value === 'all'
      ? store.requests
      : store.requests.filter((r) => r.status === filter.value)
  return [...list].sort((a, b) => b.request_count - a.request_count)
})

const approvedCount = computed(
  () => store.requests.filter((r) => r.status === 'approved').length,
)
const pendingCount = computed(() => store.pending.length)

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  })
}
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-xl font-semibold text-slate-900">Elective Requests</h2>
      <p class="mt-1 text-sm text-slate-500">
        Non-plan elective course demand from student requests.
      </p>
    </div>

    <!-- KPI row -->
    <div class="grid gap-4 md:grid-cols-3">
      <SummaryCard
        label="Total Requests"
        :value="store.totalRequests.toLocaleString()"
      />
      <SummaryCard label="Pending" :value="pendingCount.toString()" />
      <SummaryCard label="Approved" :value="approvedCount.toString()" />
    </div>

    <!-- Filter strip -->
    <div
      class="flex items-center justify-between rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
    >
      <div class="text-sm text-slate-500">
        Showing {{ visible.length }}
        {{ visible.length === 1 ? 'course' : 'courses' }}
      </div>
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

    <!-- Empty -->
    <div
      v-if="visible.length === 0"
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
            d="M9 12h6m-6 4h6M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1z"
          />
        </svg>
      </div>
      <h3 class="mt-4 text-base font-semibold text-slate-900">
        No requests in this view
      </h3>
      <p class="mt-1 text-sm text-slate-500">
        Try a different filter or check back after the next prediction run.
      </p>
    </div>

    <!-- Cards grid -->
    <div v-else class="grid gap-4 lg:grid-cols-2">
      <div
        v-for="r in visible"
        :key="r.id"
        class="flex flex-col rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-slate-300"
      >
        <div class="flex items-start justify-between gap-4">
          <div class="min-w-0 flex-1">
            <div class="font-mono text-xs text-slate-500">
              {{ r.course_code }}
            </div>
            <h4 class="mt-1 truncate text-base font-semibold text-slate-900">
              {{ r.course_title }}
            </h4>
            <div class="mt-0.5 text-xs text-slate-500">{{ r.department }}</div>
          </div>
          <span
            class="inline-flex flex-none items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset"
            :class="statusStyles[r.status]"
          >
            {{ statusLabels[r.status] }}
          </span>
        </div>

        <div
          class="mt-5 flex items-end justify-between border-t border-slate-100 pt-4"
        >
          <div>
            <div class="text-2xl font-semibold text-slate-900">
              {{ r.request_count }}
            </div>
            <div class="text-xs text-slate-500">
              {{ r.request_count === 1 ? 'student request' : 'student requests' }}
              · last {{ formatDate(r.last_requested_at) }}
            </div>
          </div>

          <div class="flex gap-2">
            <button
              v-if="r.status !== 'rejected'"
              type="button"
              class="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
              @click="store.setStatus(r.id, 'rejected')"
            >
              Reject
            </button>
            <button
              v-if="r.status !== 'approved'"
              type="button"
              class="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white shadow-sm transition hover:bg-blue-700"
              @click="store.setStatus(r.id, 'approved')"
            >
              Approve
            </button>
            <button
              v-if="r.status !== 'pending'"
              type="button"
              class="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
              @click="store.setStatus(r.id, 'pending')"
            >
              Reopen
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
