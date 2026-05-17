/**
 * Pinia store for substitute-request CRUD.
 *
 * Holds two parallel lists in state — the student's own requests
 * (whatever student_id is currently being viewed) and the reviewer's
 * full list. They're loaded independently so a reviewer toggling roles
 * doesn't lose either side's state.
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import {
  listSubstituteRequests,
  createSubstituteRequest,
  updateSubstituteRequest,
} from '@/api/client'
import type {
  SubstituteRequest,
  SubstituteRequestCreate,
  SubstituteRequestUpdate,
  SubstituteRequestStatus,
} from '@/types'

export const useSubstituteRequestsStore = defineStore('substituteRequests', () => {
  // ── State ───────────────────────────────────────────────────────────
  const studentRequests = ref<SubstituteRequest[]>([])
  const allRequests = ref<SubstituteRequest[]>([])
  const isLoading = ref(false)
  const isSubmitting = ref(false)
  const error = ref<string | null>(null)
  const currentStudentId = ref('')

  // ── Getters ─────────────────────────────────────────────────────────
  const pendingCount = computed(
    () => allRequests.value.filter((r) => r.status === 'pending').length
  )
  const studentPendingCount = computed(
    () => studentRequests.value.filter((r) => r.status === 'pending').length
  )

  function bucketByStatus(list: SubstituteRequest[]) {
    return {
      pending:  list.filter((r) => r.status === 'pending'),
      approved: list.filter((r) => r.status === 'approved'),
      rejected: list.filter((r) => r.status === 'rejected'),
    }
  }

  // ── Actions ─────────────────────────────────────────────────────────
  async function loadStudentRequests(studentId: string) {
    if (!studentId.trim()) {
      studentRequests.value = []
      currentStudentId.value = ''
      return
    }
    isLoading.value = true
    error.value = null
    try {
      currentStudentId.value = studentId.trim()
      studentRequests.value = await listSubstituteRequests({
        studentId: currentStudentId.value,
      })
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : 'Failed to load requests.'
      studentRequests.value = []
    } finally {
      isLoading.value = false
    }
  }

  async function loadAllRequests(statusFilter?: SubstituteRequestStatus) {
    isLoading.value = true
    error.value = null
    try {
      allRequests.value = await listSubstituteRequests(
        statusFilter ? { status: statusFilter } : undefined
      )
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : 'Failed to load requests.'
      allRequests.value = []
    } finally {
      isLoading.value = false
    }
  }

  async function submitRequest(payload: SubstituteRequestCreate) {
    isSubmitting.value = true
    error.value = null
    try {
      const created = await createSubstituteRequest(payload)
      // Push to the front of student's list if it's for the current student
      if (created.student_id === currentStudentId.value) {
        studentRequests.value = [created, ...studentRequests.value]
      }
      return created
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : 'Failed to submit request.'
      throw e
    } finally {
      isSubmitting.value = false
    }
  }

  async function reviewRequest(id: number, payload: SubstituteRequestUpdate) {
    isSubmitting.value = true
    error.value = null
    try {
      const updated = await updateSubstituteRequest(id, payload)
      // Patch both lists in place if the request appears in them
      const replace = (list: SubstituteRequest[]) =>
        list.map((r) => (r.id === updated.id ? updated : r))
      allRequests.value = replace(allRequests.value)
      studentRequests.value = replace(studentRequests.value)
      return updated
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : 'Failed to update request.'
      throw e
    } finally {
      isSubmitting.value = false
    }
  }

  function clearError() {
    error.value = null
  }

  return {
    // state
    studentRequests,
    allRequests,
    isLoading,
    isSubmitting,
    error,
    currentStudentId,
    // getters
    pendingCount,
    studentPendingCount,
    bucketByStatus,
    // actions
    loadStudentRequests,
    loadAllRequests,
    submitRequest,
    reviewRequest,
    clearError,
  }
})
