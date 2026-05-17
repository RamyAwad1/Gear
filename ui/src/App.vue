<script setup lang="ts">
/**
 * Application shell.
 *
 * Owns the active tab and switches between the five functional views.
 * Default landing tab is the dashboard so users see system status first;
 * Data Management is one click away from there.
 */

import { ref } from 'vue'
import AppHeader from '@/components/AppHeader.vue'
import TabNav from '@/components/TabNav.vue'
import Dashboard from '@/views/Dashboard.vue'
import DataManagement from '@/views/DataManagement.vue'
import Predictions from '@/views/Predictions.vue'
import FlaggedStudents from '@/views/FlaggedStudents.vue'
import ElectiveRequests from '@/views/ElectiveRequests.vue'
import SubstituteRequests from "@/views/SubstituteRequests.vue";
import type { TabKey } from '@/types'

const activeTab = ref<TabKey>('dashboard')
</script>

<template>
  <div class="min-h-screen bg-slate-50">
    <AppHeader />
    <TabNav v-model="activeTab" />

    <main class="mx-auto max-w-7xl px-8 py-8">
      <Dashboard
        v-if="activeTab === 'dashboard'"
        @go-upload="activeTab = 'data-management'"
        @go-predictions="activeTab = 'predictions'"
      />
      <DataManagement
        v-else-if="activeTab === 'data-management'"
        @done="activeTab = 'predictions'"
      />
      <Predictions
        v-else-if="activeTab === 'predictions'"
        @go-upload="activeTab = 'data-management'"
      />
      <FlaggedStudents v-else-if="activeTab === 'flagged-students'" />
      <ElectiveRequests v-else-if="activeTab === 'elective-requests'" />
      <SubstituteRequests v-else-if="activeTab === 'substitute-requests'" />
    </main>
  </div>
</template>