/**
 * Flagged students store.
 *
 * Currently seeded with mock data so the view is meaningful before the
 * `/api/flagged-students/` endpoint exists. When the backend lands,
 * replace `MOCK` with a fetch in `load()` — the view component doesn't
 * change.
 */

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import type { FlaggedStudent } from '@/types'

const MOCK: FlaggedStudent[] = [
  {
    id: 1,
    student_id: '20210042',
    student_name: 'Layla Haddad',
    major: 'Computer Science',
    degree_type: 'Bachelor',
    reason: 'graduation_risk',
    reason_detail:
      'Missing 12 credit hours of major requirements with 1 semester remaining.',
    flagged_at: '2026-04-28T09:14:00Z',
    is_resolved: false,
  },
  {
    id: 2,
    student_id: '20220133',
    student_name: 'Omar Khalil',
    major: 'Cybersecurity',
    degree_type: 'Bachelor',
    reason: 'missing_prereq',
    reason_detail:
      'Registered for Network Security without passing Networking (0010203180).',
    flagged_at: '2026-04-27T15:02:00Z',
    is_resolved: false,
  },
  {
    id: 3,
    student_id: '20230091',
    student_name: 'Yasmeen Saleh',
    major: 'Data Science & AI',
    degree_type: 'Bachelor',
    reason: 'low_gpa',
    reason_detail: 'Cumulative GPA 1.84 — below 2.0 threshold for two terms.',
    flagged_at: '2026-04-27T11:48:00Z',
    is_resolved: false,
  },
  {
    id: 4,
    student_id: '20210178',
    student_name: 'Tariq Al-Najjar',
    major: 'Computer Science',
    degree_type: 'Bachelor',
    reason: 'repeating_failure',
    reason_detail:
      'Third attempt at Data Structures and Algorithms (0040201201).',
    flagged_at: '2026-04-26T14:30:00Z',
    is_resolved: false,
  },
  {
    id: 5,
    student_id: '20240056',
    student_name: 'Noor Abu-Rashid',
    major: 'AI Technical',
    degree_type: 'Technical',
    reason: 'excess_credits',
    reason_detail: '21 credits requested — exceeds 18-credit max for the term.',
    flagged_at: '2026-04-26T08:15:00Z',
    is_resolved: false,
  },
  {
    id: 6,
    student_id: '20220210',
    student_name: 'Dana Mansour',
    major: 'Cybersecurity',
    degree_type: 'Bachelor',
    reason: 'probation',
    reason_detail:
      'Academic probation — must achieve term GPA ≥ 2.5 to continue.',
    flagged_at: '2026-04-25T16:42:00Z',
    is_resolved: false,
  },
  {
    id: 7,
    student_id: '20210301',
    student_name: 'Hamza Odeh',
    major: 'Computer Science',
    degree_type: 'Bachelor',
    reason: 'graduation_risk',
    reason_detail:
      'Capstone Project I prerequisite (Computing Research Project) not yet passed.',
    flagged_at: '2026-04-25T10:20:00Z',
    is_resolved: true,
  },
  {
    id: 8,
    student_id: '20230145',
    student_name: 'Sara Qasem',
    major: 'Data Science & AI',
    degree_type: 'Bachelor',
    reason: 'missing_prereq',
    reason_detail:
      'Registered for Machine Learning without AI and Intelligent Systems.',
    flagged_at: '2026-04-24T13:55:00Z',
    is_resolved: false,
  },
  {
    id: 9,
    student_id: '20220088',
    student_name: 'Khaled Issa',
    major: 'CS Technical',
    degree_type: 'Technical',
    reason: 'low_gpa',
    reason_detail: 'Cumulative GPA 1.92 — below 2.0 threshold.',
    flagged_at: '2026-04-24T09:08:00Z',
    is_resolved: false,
  },
  {
    id: 10,
    student_id: '20210067',
    student_name: 'Rana Bishara',
    major: 'Cybersecurity',
    degree_type: 'Bachelor',
    reason: 'repeating_failure',
    reason_detail:
      'Second attempt at Operating Systems (0040201341) after withdrawal.',
    flagged_at: '2026-04-23T17:30:00Z',
    is_resolved: false,
  },
  {
    id: 11,
    student_id: '20240002',
    student_name: 'Anas Saleem',
    major: 'Computer Science',
    degree_type: 'Bachelor',
    reason: 'missing_prereq',
    reason_detail:
      'Registered for Advanced Programming without passing Programming.',
    flagged_at: '2026-04-23T11:14:00Z',
    is_resolved: true,
  },
  {
    id: 12,
    student_id: '20220166',
    student_name: 'Lina Al-Tamimi',
    major: 'Data Science & AI',
    degree_type: 'Bachelor',
    reason: 'graduation_risk',
    reason_detail:
      'On track for 9th semester — Bachelor plan limit is 8 with extension.',
    flagged_at: '2026-04-22T14:46:00Z',
    is_resolved: false,
  },
]

export const useFlaggedStudentsStore = defineStore('flaggedStudents', () => {
  const students = ref<FlaggedStudent[]>(MOCK)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const unresolved = computed(() =>
    students.value.filter((s) => !s.is_resolved),
  )
  const resolved = computed(() =>
    students.value.filter((s) => s.is_resolved),
  )

  /** When the backend ships, replace with a real fetch. */
  async function load() {
    // no-op for now — data is preloaded
  }

  function toggleResolved(id: number) {
    const target = students.value.find((s) => s.id === id)
    if (target) target.is_resolved = !target.is_resolved
  }

  return {
    students,
    loading,
    error,
    unresolved,
    resolved,
    load,
    toggleResolved,
  }
})
