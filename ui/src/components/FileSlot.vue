<script setup lang="ts">
/**
 * One file picker slot. Displays as a dashed dropzone when empty and
 * switches to a green "selected" state with file size + Remove button
 * once a file is picked. Pure click-to-select; no drag-and-drop yet —
 * keeping the slice small.
 */

defineProps<{
  label: string
  hint: string
  file: File | null
  disabled?: boolean
}>()

const emit = defineEmits<{
  pick: []
  clear: []
}>()

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}
</script>

<template>
  <!-- Selected state -->
  <div
    v-if="file"
    class="flex items-center justify-between rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3"
  >
    <div class="min-w-0 flex-1">
      <div class="text-xs uppercase tracking-wider text-emerald-700">
        {{ label }}
      </div>
      <div class="mt-0.5 truncate text-sm font-medium text-emerald-900">
        {{ file.name }}
      </div>
      <div class="text-xs text-emerald-700/70">{{ formatBytes(file.size) }}</div>
    </div>
    <button
      type="button"
      :disabled="disabled"
      class="ml-4 text-xs font-medium text-emerald-700 transition hover:text-emerald-900 disabled:opacity-40"
      @click="emit('clear')"
    >
      Remove
    </button>
  </div>

  <!-- Empty state -->
  <button
    v-else
    type="button"
    :disabled="disabled"
    class="flex w-full flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 px-4 py-8 text-center transition hover:border-blue-400 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-60"
    @click="emit('pick')"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.5"
      stroke-linecap="round"
      stroke-linejoin="round"
      class="mb-3 h-8 w-8 text-slate-400"
    >
      <path d="M12 16V4" />
      <path d="m6 10 6-6 6 6" />
      <path d="M4 20h16" />
    </svg>
    <div class="text-sm font-medium text-slate-700">{{ label }}</div>
    <div class="mt-1 text-xs text-slate-500">Choose {{ hint }}</div>
  </button>
</template>
