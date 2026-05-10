/**
 * Application entry point.
 *
 * Order matters: Tailwind CSS must be imported before component styles
 * so utility classes can override defaults consistently. Pinia is
 * installed before App is mounted so `useXxxStore()` works immediately
 * inside `<script setup>` blocks.
 */

import { createApp } from 'vue'
import { createPinia } from 'pinia'

import './style.css'
import App from './App.vue'

createApp(App).use(createPinia()).mount('#app')