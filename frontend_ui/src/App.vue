<script setup lang="ts">
/**
 * Application shell.
 *
 * Owns the active tab and switches between the two functional views.
 * Adding more tabs later means: enable them in TabNav.vue and add a
 * v-if branch below.
 */

import { ref } from 'vue'
import AppHeader from '@/components/AppHeader.vue'
import TabNav from '@/components/TabNav.vue'
import DataManagement from '@/views/DataManagement.vue'
import Predictions from '@/views/Predictions.vue'
import type { TabKey } from '@/types'

const activeTab = ref<TabKey>('data-management')
</script>

<template>
  <div class="min-h-screen bg-slate-50">
    <AppHeader />
    <TabNav v-model="activeTab" />

    <main class="mx-auto max-w-7xl px-8 py-8">
      <DataManagement
        v-if="activeTab === 'data-management'"
        @done="activeTab = 'predictions'"
      />
      <Predictions
        v-else-if="activeTab === 'predictions'"
        @go-upload="activeTab = 'data-management'"
      />
    </main>
  </div>
</template>