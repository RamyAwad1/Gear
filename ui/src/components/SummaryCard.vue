<script setup lang="ts">
/**
 * Single statistic card used in the Predictions summary row.
 */

defineProps<{
  label: string
  value: string
}>()
</script>

<template>
  <div class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
    <div class="text-xs font-medium uppercase tracking-wider text-slate-500">
      {{ label }}
    </div>
    <div class="mt-2 text-2xl font-semibold text-slate-900">{{ value }}</div>
  </div>
</template>
