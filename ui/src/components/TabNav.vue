<script setup lang="ts">
/**
 * Tab navigation strip. All five tabs are now wired up to real views.
 * Disabled state remains in the prop shape so individual tabs can be
 * temporarily turned off for maintenance without removing the entry.
 */

import type { TabKey } from '@/types'

interface TabDef {
  key: TabKey
  label: string
  enabled: boolean
}

const tabs: TabDef[] = [
  { key: 'dashboard', label: 'Dashboard', enabled: true },
  { key: 'predictions', label: 'Predictions', enabled: true },
  { key: 'data-management', label: 'Data Management', enabled: true },
  { key: 'flagged-students', label: 'Flagged Students', enabled: true },
  { key: 'elective-requests', label: 'Elective Requests', enabled: true },
  { key: 'substitute-requests', label: 'Substitute Requests', enabled: true},
]

defineProps<{ modelValue: TabKey }>()
const emit = defineEmits<{ 'update:modelValue': [value: TabKey] }>()

function selectTab(tab: TabDef) {
  if (tab.enabled) emit('update:modelValue', tab.key)
}
</script>

<template>
  <nav class="border-b border-slate-200 bg-white">
    <div class="mx-auto max-w-7xl px-8">
      <div class="flex gap-1">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          type="button"
          :disabled="!tab.enabled"
          :title="tab.enabled ? '' : 'Coming soon'"
          class="relative px-4 py-3.5 text-sm font-medium transition focus:outline-none"
          :class="[
            tab.enabled
              ? 'cursor-pointer hover:text-slate-900'
              : 'cursor-not-allowed text-slate-300',
            modelValue === tab.key
              ? 'text-blue-600'
              : tab.enabled
                ? 'text-slate-500'
                : '',
          ]"
          @click="selectTab(tab)"
        >
          {{ tab.label }}
          <span
            v-if="modelValue === tab.key"
            class="absolute inset-x-3 -bottom-px h-0.5 bg-blue-600"
          />
        </button>
      </div>
    </div>
  </nav>
</template>