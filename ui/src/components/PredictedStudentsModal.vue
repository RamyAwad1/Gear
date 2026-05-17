<script setup lang="ts">
import { ref, computed, watch } from "vue";
import type { CoursePrediction, PredictedStudent } from "@/api/client";

const props = defineProps<{
  course: CoursePrediction | null;
  isOpen: boolean;
}>();

const emit = defineEmits<{ (e: "close"): void }>();

const search = ref("");
const reasonFilter = ref<"all" | "passed_predecessor" | "retake_candidate">("all");

watch(
  () => props.isOpen,
  (open) => {
    if (!open) {
      search.value = "";
      reasonFilter.value = "all";
    }
  }
);

const filteredStudents = computed<PredictedStudent[]>(() => {
  if (!props.course) return [];
  let rows = props.course.predicted_students || [];

  if (reasonFilter.value !== "all") {
    rows = rows.filter((s) => s.reason === reasonFilter.value);
  }
  const q = search.value.trim().toLowerCase();
  if (q) {
    rows = rows.filter((s) => {
      return (
        s.student_id.toLowerCase().includes(q) ||
        s.major.toLowerCase().includes(q) ||
        s.plan_key.toLowerCase().includes(q)
      );
    });
  }
  return rows;
});

const predecessorCount = computed(() =>
  props.course
    ? props.course.predicted_students.filter(
        (s) => s.reason === "passed_predecessor"
      ).length
    : 0
);

const retakeCount = computed(() =>
  props.course
    ? props.course.predicted_students.filter(
        (s) => s.reason === "retake_candidate"
      ).length
    : 0
);

const remainderCount = computed(() => {
  if (!props.course) return 0;
  return Math.max(
    0,
    props.course.predicted_enrollment - props.course.identified_count
  );
});

function reasonLabel(reason: string) {
  if (reason === "passed_predecessor") return "Prereq just passed";
  if (reason === "retake_candidate") return "Likely retake";
  return reason;
}

function reasonClass(reason: string) {
  if (reason === "passed_predecessor")
    return "bg-blue-50 text-blue-700 border-blue-200";
  if (reason === "retake_candidate")
    return "bg-amber-50 text-amber-700 border-amber-200";
  return "bg-slate-50 text-slate-700 border-slate-200";
}

function exportCSV() {
  if (!props.course) return;
  const rows = filteredStudents.value;
  const header = [
    "student_id",
    "major",
    "degree_type",
    "plan_key",
    "current_year",
    "cum_hrs",
    "gpa",
    "reason",
    "confidence",
  ];
  const csv = [
    header.join(","),
    ...rows.map((r) =>
      header
        .map((k) => {
          const v = r[k as keyof PredictedStudent];
          return v === null || v === undefined ? "" : String(v);
        })
        .join(",")
    ),
  ].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `predicted_${props.course.course_code}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
</script>

<template>
  <div
    v-if="isOpen && course"
    class="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-4"
    @click.self="emit('close')"
  >
    <div
      class="flex h-[85vh] w-full max-w-4xl flex-col rounded-xl border border-slate-200 bg-white shadow-xl"
    >
      <!-- Header -->
      <div class="flex items-start justify-between border-b border-slate-200 p-6">
        <div class="min-w-0 flex-1">
          <div class="font-mono text-xs uppercase tracking-wide text-slate-500">
            {{ course.course_code }}
          </div>
          <h2 class="mt-1 truncate text-xl font-semibold text-slate-900">
            {{ course.course_title || "Untitled course" }}
          </h2>
          <p class="mt-2 text-sm text-slate-600">
            Predicted to have
            <span class="font-semibold text-slate-900">{{ course.predicted_enrollment }}</span>
            students next semester.
            <span class="font-semibold text-slate-900">{{ course.identified_count }}</span>
            individually identified below.
            <template v-if="remainderCount > 0">
              The remaining
              <span class="font-semibold text-slate-900">{{ remainderCount }}</span>
              are projected from historical enrolment patterns
              (incoming cohorts, term-share trends) where individual
              students can't yet be named.
            </template>
          </p>
        </div>
        <button
          @click="emit('close')"
          class="ml-4 rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
        >
          <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <!-- Filters bar -->
      <div
        class="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-slate-50 px-6 py-3"
      >
        <input
          v-model="search"
          type="text"
          placeholder="Search ID, major, or plan…"
          class="min-w-[200px] flex-1 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <select
          v-model="reasonFilter"
          class="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="all">All reasons ({{ course.identified_count }})</option>
          <option value="passed_predecessor">
            Prereq just passed ({{ predecessorCount }})
          </option>
          <option value="retake_candidate">
            Likely retake ({{ retakeCount }})
          </option>
        </select>
        <button
          @click="exportCSV"
          :disabled="filteredStudents.length === 0"
          class="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Export CSV
        </button>
      </div>

      <!-- Body: student list -->
      <div class="flex-1 overflow-y-auto">
        <table v-if="filteredStudents.length > 0" class="w-full text-sm">
          <thead class="sticky top-0 z-10 bg-slate-50 text-xs uppercase tracking-wide text-slate-600">
            <tr>
              <th class="px-6 py-3 text-left font-medium">Student ID</th>
              <th class="px-6 py-3 text-left font-medium">Major</th>
              <th class="px-6 py-3 text-left font-medium">Degree</th>
              <th class="px-6 py-3 text-left font-medium">Year</th>
              <th class="px-6 py-3 text-left font-medium">Hrs</th>
              <th class="px-6 py-3 text-left font-medium">GPA</th>
              <th class="px-6 py-3 text-left font-medium">Reason</th>
              <th class="px-6 py-3 text-right font-medium">Confidence</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            <tr v-for="s in filteredStudents" :key="s.student_id" class="hover:bg-slate-50">
              <td class="px-6 py-2.5 font-mono text-slate-900">{{ s.student_id }}</td>
              <td class="px-6 py-2.5 text-slate-700">{{ s.major || "—" }}</td>
              <td class="px-6 py-2.5 text-slate-700">{{ s.degree_type || "—" }}</td>
              <td class="px-6 py-2.5 text-slate-700">Y{{ s.current_year }}</td>
              <td class="px-6 py-2.5 text-slate-700">{{ s.cum_hrs }}</td>
              <td class="px-6 py-2.5 text-slate-700">
                {{ s.gpa !== null ? s.gpa.toFixed(2) : "—" }}
              </td>
              <td class="px-6 py-2.5">
                <span
                  :class="reasonClass(s.reason)"
                  class="inline-flex rounded-full border px-2 py-0.5 text-xs font-medium"
                >
                  {{ reasonLabel(s.reason) }}
                </span>
              </td>
              <td class="px-6 py-2.5 text-right font-mono text-slate-700">
                {{ (s.confidence * 100).toFixed(0) }}%
              </td>
            </tr>
          </tbody>
        </table>

        <div v-else class="flex h-full items-center justify-center p-12 text-center">
          <div class="max-w-md">
            <p class="text-sm font-medium text-slate-900">
              No students match the current filter.
            </p>
            <p v-if="course.identified_count === 0" class="mt-2 text-sm text-slate-600">
              This course's forecast is driven entirely by historical
              enrolment patterns. No individual students can yet be named —
              the predicted enrolment of
              <span class="font-semibold">{{ course.predicted_enrollment }}</span>
              comes from this course's own past Spring/Fall/Summer numbers,
              not from a tracked cohort of current students.
            </p>
            <p v-else class="mt-2 text-sm text-slate-600">
              Try clearing the search box or changing the reason filter.
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
