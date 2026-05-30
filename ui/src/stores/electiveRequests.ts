/**
 * Elective requests store.
 *
 * Aggregates per-elective student demand. Mock-seeded for now; replace
 * `MOCK` with a fetch from `/api/elective-requests/` when the endpoint
 * exists. Approve/reject mutations are local-only until that point.
 */

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import type { ElectiveRequest, ElectiveRequestStatus } from '@/types'

const MOCK: ElectiveRequest[] = [
  {
    id: 1,
    course_code: '0010204350',
    course_title: 'Machine Learning',
    department: 'Data Science & AI',
    request_count: 87,
    status: 'pending',
    last_requested_at: '2026-04-29T12:30:00Z',
  },
  {
    id: 2,
    course_code: '0010203362',
    course_title: 'Ethical Hacking',
    department: 'Cybersecurity',
    request_count: 64,
    status: 'pending',
    last_requested_at: '2026-04-29T10:11:00Z',
  },
  {
    id: 3,
    course_code: '0010204351',
    course_title: 'Natural Language Processing',
    department: 'Data Science & AI',
    request_count: 52,
    status: 'approved',
    last_requested_at: '2026-04-28T16:48:00Z',
  },
  {
    id: 4,
    course_code: '0040201362',
    course_title: 'Games Engine and Scripting',
    department: 'Computer Science',
    request_count: 41,
    status: 'pending',
    last_requested_at: '2026-04-28T09:25:00Z',
  },
  {
    id: 5,
    course_code: '0010203361',
    course_title: 'Forensics',
    department: 'Cybersecurity',
    request_count: 38,
    status: 'approved',
    last_requested_at: '2026-04-27T14:02:00Z',
  },
  {
    id: 6,
    course_code: '0000204311',
    course_title: 'Big Data Analytics and Visualization',
    department: 'Data Science & AI',
    request_count: 33,
    status: 'pending',
    last_requested_at: '2026-04-27T11:18:00Z',
  },
  {
    id: 7,
    course_code: '0010204330',
    course_title: 'Modeling and Simulation',
    department: 'Data Science & AI',
    request_count: 22,
    status: 'pending',
    last_requested_at: '2026-04-26T15:54:00Z',
  },
  {
    id: 8,
    course_code: '0000203420',
    course_title: 'Secure Coding',
    department: 'Cybersecurity',
    request_count: 19,
    status: 'rejected',
    last_requested_at: '2026-04-26T08:33:00Z',
  },
  {
    id: 9,
    course_code: '0040201430',
    course_title: 'Database Programming',
    department: 'Computer Science',
    request_count: 17,
    status: 'pending',
    last_requested_at: '2026-04-25T13:40:00Z',
  },
  {
    id: 10,
    course_code: '0000203400',
    course_title: 'Risk Analysis and Systems Testing',
    department: 'Cybersecurity',
    request_count: 14,
    status: 'pending',
    last_requested_at: '2026-04-25T10:07:00Z',
  },
  {
    id: 11,
    course_code: '0010203300',
    course_title: 'Information Security Management',
    department: 'Cybersecurity',
    request_count: 11,
    status: 'rejected',
    last_requested_at: '2026-04-24T16:22:00Z',
  },
  {
    id: 12,
    course_code: '0000204430',
    course_title: 'Data Mining',
    department: 'Data Science & AI',
    request_count: 9,
    status: 'pending',
    last_requested_at: '2026-04-24T09:50:00Z',
  },
]

export const useElectiveRequestsStore = defineStore('electiveRequests', () => {
  const requests = ref<ElectiveRequest[]>(MOCK)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const pending = computed(() =>
    requests.value.filter((r) => r.status === 'pending'),
  )
  const totalRequests = computed(() =>
    requests.value.reduce((sum, r) => sum + r.request_count, 0),
  )

  async function load() {
    // no-op until the backend endpoint exists
  }

  function setStatus(id: number, status: ElectiveRequestStatus) {
    const target = requests.value.find((r) => r.id === id)
    if (target) target.status = status
  }

  return {
    requests,
    loading,
    error,
    pending,
    totalRequests,
    load,
    setStatus,
  }
})
