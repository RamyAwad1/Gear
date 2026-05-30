# System Documentation

Architectural reference for the HTU Intelligent Registration System. Covers
the forecasting algorithm, API contracts, data formats, and per-module
responsibilities. For setup and quick-start instructions, see `README.md`.

## Architecture overview

The system has three layers and no hidden state. A single prediction
request flows top-to-bottom in one pass:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  FRONTEND  ·  Vue 3 + Pinia + Tailwind  ·  Vite dev server (port 5173)   │
│  Five tabs: Dashboard · Predictions · Data Management · Flagged Students │
│            · Elective Requests                                           │
└─────────────────────────────────────┬────────────────────────────────────┘
                          POST /api/predict/upload/
                       (multipart: 2 CSVs + form fields)
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BACKEND  ·  Django 6 + REST Framework  ·  port 8000                     │
│                                                                          │
│  core/api_views.py                                                       │
│    ├─ POST /api/predict/upload/   validates CSVs, dispatches to engine   │
│    └─ GET  /api/health/           liveness check                         │
│                                      │                                   │
│  ml/inference.py                     │                                   │
│    └─ predict_next_semester(...)  ◄──┘                                   │
│        ├─ Signal 1: time-series                                          │
│        ├─ Signal 2: cohort + retake                                      │
│        ├─ Signal 3: plan + year match                                    │
│        └─ combine: forecast = max(ts, cohort + retake)                   │
│           returns JSON: totals + per-course predictions + named students │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PERSISTENCE  ·  PostgreSQL                                              │
│  Dormant. The `AppUser` schema is defined for future authentication      │
│  but no working feature reads from or writes to the database.            │
└──────────────────────────────────────────────────────────────────────────┘
```

The forecasting engine has no trained model, no training step, and no
cached state. Identical input always produces identical output.

## Forecasting algorithm

For each course in the catalogue, three independent signals are computed
from the uploaded enrolment history, then combined.

### Signal 1 — Time-series

A weighted three-period average of past same-term enrolments, plus a
damped year-over-year trend term.

```
ts = w · last_three_same_term_avg + damped_trend
```

This captures the baseline that *something like the last few semesters'*
takers will appear. It produces an aggregate count but does not identify
specific students.

### Signal 2 — Cohort + retake

Two named-student signals:

- **Cohort**: students who just passed the inferred predecessor of the
  target course (the system infers predecessor relationships from temporal
  precedence in the data; about 73 of 97 courses get a clear predecessor
  with ≥30% precedence)
- **Retake**: students whose most recent attempt at the course was a fail
  and who haven't since passed it, scaled by the historical retake rate

```
cohort + retake = |ready_pool| · take_rate + |fail_pool| · retake_rate
```

This produces both a count and the specific student IDs behind that count.

### Signal 3 — Plan + year match

Broad cohort signal for courses without a clear predecessor (year-1
fundamentals like Programming or Functional Math). Groups students by
`(plan_key, current_academic_year)` and surfaces students whose plan
historically takes the course in that year.

### Combination

```
forecast = max( ts, cohort + retake )
```

Maximum, not sum, because the time-series already implicitly contains the
cohort and retake students from past terms — adding them again would
double-count. Whichever signal fires more strongly wins. Plan-match
contributes named students but doesn't inflate the count beyond what the
other two signals would predict.

### Sections needed

```
sections_needed = ceil( forecast × (1 + buffer_pct) / section_cap )
```

Defaults: `buffer_pct = 0.10`, `section_cap = 30`. Both can be overridden
per request.

## API endpoints

### `POST /api/predict/upload/`

Stateless prediction. Accepts two CSV uploads plus optional form fields.

**Multipart form fields:**

| Field              | Type     | Required | Notes                                  |
|--------------------|----------|----------|----------------------------------------|
| `enrollments_file` | file     | yes      | Enrolment history CSV                  |
| `students_file`    | file     | yes      | Student demographic / academic CSV     |
| `semester`         | string   | no       | Target term like `2025_Spring`         |
| `section_cap`      | integer  | no       | Max students per section (default 30)  |
| `buffer_pct`       | float    | no       | Capacity buffer (default 0.10)         |

**Response shape (200 OK):**

```json
{
  "semester_predicted": "2025_Spring",
  "section_cap": 30,
  "buffer_pct": 0.10,
  "totals": {
    "courses": 56,
    "predicted_enrollment": 8692,
    "sections_needed": 378,
    "students_processed": 3000
  },
  "courses": [
    {
      "course_code": "0040201310",
      "course_title": "Data Structures",
      "predicted_enrollment": 240,
      "sections_needed": 8,
      "last_semester_actual": 235,
      "status": "optimal",
      "predicted_students": [
        {
          "student_id": "22110448",
          "major": "CS",
          "degree_type": "Bachelor",
          "plan_key": "CS_Bachelor",
          "admission_year": 2022,
          "current_year": 3,
          "cum_hrs": 78,
          "gpa": 3.4,
          "reason": "passed_predecessor",
          "confidence": 0.85
        }
      ],
      "identified_count": 187
    }
  ]
}
```

**Error responses:** `400` for missing files or columns, `422` for empty
prediction results, `500` for internal errors (full traceback included for
development).

### `GET /api/health/`

Liveness check. Returns `{"status": "ok", "service": "htu-irs"}`.

## Data formats

### Enrolments CSV — required columns

| Column           | Type    | Notes                                       |
|------------------|---------|---------------------------------------------|
| `student_id`     | string  | Matches `student_id` in students CSV        |
| `course_code`    | string  | 10-digit zero-padded format preferred       |
| `semester_label` | string  | e.g. `2024_Fall`                            |
| `year`           | integer | Calendar year of the academic term          |
| `term`           | string  | `Fall`, `Spring`, or `Summer`               |
| `grade`          | string  | Letter grade, blank for withdrawn           |
| `credit_hours`   | integer | Course credit hours                         |
| `status`         | string  | `Completed`, `Withdrawn`, `Incomplete`, etc.|

Optional column: `course_name` — used by the dashboard to display human-
readable course titles.

### Students CSV — required columns

| Column          | Type    | Notes                                       |
|-----------------|---------|---------------------------------------------|
| `student_id`    | string  | Unique identifier                           |
| `major`         | string  | Department code (`CS`, `DSAI`, `CYB`)       |

Optional columns (all default to sensible values when absent):

| Column                | Default     |
|-----------------------|-------------|
| `name`                | empty       |
| `degree_type`         | `Bachelor`  |
| `admission_year`      | 2020        |
| `cumulative_gpa`      | 0.0         |
| `earned_credit_hours` | 0           |
| `is_graduating_flag`  | false       |
| `needs_rem_arabic`    | 0           |
| `needs_rem_english`   | 0           |
| `needs_rem_math`      | 0           |

### Realistic-data constraint

The data simulator (`generate_data.py`) enforces a minimum of 16 distinct
students per (course, semester) section. Any combination that falls below
that threshold is dropped from the output, because a section that small
wouldn't have opened in reality. Real-world uploads are expected to honour
the same constraint.

## Modules

### `ml/inference.py`

Single-file forecasting engine. Public entry point:

```python
predict_next_semester(enrolls_df, students_df, target_semester=None,
                      section_cap=30, buffer_pct=0.10) -> dict
```

Key helpers (all private):

| Function                              | Responsibility                         |
|---------------------------------------|----------------------------------------|
| `_build_predecessor_map`              | Infer predecessor course per code      |
| `_ts_forecast`                        | Time-series signal (Signal 1)          |
| `_compute_plan_eligibility_pools`     | Cohort flow signal (Signal 2 part a)   |
| `_compute_retake_pools`               | Retake signal (Signal 2 part b)        |
| `_compute_plan_affinity`              | Plan-match signal (Signal 3)           |
| `_first_time_takerate`                | Historical take-rate for ready pools   |
| `_next_semester_label`                | Auto-detect target term from data      |

### `core/api_views.py`

Two view functions: `health` and `predict_upload`. The latter validates
required columns, parses the CSVs with pandas, calls
`predict_next_semester`, and serializes the result. All errors return
structured JSON.

Also defines small response-shaping helpers:

| Function                  | Responsibility                                  |
|---------------------------|-------------------------------------------------|
| `_course_titles_lookup`   | Pull `course_name` → `course_code` mapping      |
| `_last_semester_actuals`  | Term-aware comparison against the matching past semester |
| `_classify_status`        | optimal / warning / critical heuristic          |

### `core/models.py`

Django models. Currently contains the `AppUser` model (extending
`AbstractUser`) plus supporting tables (`Department`, `Course`, etc.) that
are kept for the schema but are not read or written by any active feature.
The system is fully stateless at runtime.

### `core/urls.py`

Two routes:

```
path("health/",         api_views.health,         name="health"),
path("predict/upload/", api_views.predict_upload, name="predict_upload"),
```

### `ui/src/`

Vue 3 single-page app. Key files:

| Path                                 | Responsibility                          |
|--------------------------------------|-----------------------------------------|
| `App.vue`                            | Shell, owns the active tab              |
| `components/TabNav.vue`              | Tab navigation strip                    |
| `components/SummaryCard.vue`         | KPI card primitive                      |
| `views/Dashboard.vue`                | Overview — KPIs, needs attention, top demand |
| `views/Predictions.vue`              | Sortable per-course table               |
| `views/DataManagement.vue`           | CSV upload + run-prediction trigger     |
| `views/FlaggedStudents.vue`          | At-risk student list                    |
| `views/ElectiveRequests.vue`         | Aggregated elective demand              |
| `stores/predictions.ts`              | Pinia store for the active prediction   |
| `api/client.ts`                      | Fetch wrappers for the two endpoints    |
| `types.ts`                           | TypeScript types mirroring API shapes   |

## Frontend tabs

| Tab                | Purpose                                                              |
|--------------------|----------------------------------------------------------------------|
| Dashboard          | Overview: KPI strip, courses needing attention (prominent), largest growth, capacity status mix, top courses by demand |
| Predictions        | Sortable table of every course in the forecast; row click opens the predicted-students modal |
| Data Management    | CSV upload form, optional target-semester / cap / buffer overrides, and reset button |
| Flagged Students   | Students whose academic trajectory suggests graduation-delay risk    |
| Elective Requests  | Aggregated demand for elective courses across all students           |

## Performance characteristics

Measured on the simulator-generated test dataset (3,000 students,
~86,000 enrolment rows, ~56 distinct courses after the 16-student
section filter):

| Metric                        | Value                          |
|-------------------------------|--------------------------------|
| End-to-end response time      | ~4.4 seconds                   |
| Predecessor inference         | ~2.8 seconds                   |
| Forecasting computation       | ~0.9 seconds                   |
| CSV parsing & validation      | ~0.4 seconds                   |
| Network & serialisation       | ~0.3 seconds                   |
| Frontend table render         | < 200 ms                       |
| Peak memory                   | < 250 MB                       |
| Stability across 50 requests  | No degradation, no memory leak |

### Forecast quality

Measured via leave-one-term-out validation against same-term past actuals:

| Metric                                       | Value     |
|----------------------------------------------|-----------|
| Pearson correlation r                        | 0.996     |
| Spearman rank correlation                    | 0.994     |
| Coefficient of determination r²              | 0.991     |
| Mean absolute error                          | 4.6 students per course |
| Mean absolute percentage error               | 7.8%      |
| Courses predicted within ±20% of actual      | 88 / 97   |

## Out of scope (Future Work)

The following are intentionally not in the delivered system:

- **Substitute-course request workflow** — prototyped during development
  (models, endpoints, role-toggle view) but descoped to focus on the
  forecasting engine. An LSI-based auto-match enhancement was scoped as
  the next iteration.
- **Authentication and role-based access** — the `AppUser` schema is
  defined but no auth flow uses it.
- **Persistent prediction runs** — every upload produces a fresh
  prediction; nothing is saved. Saving runs would enable longitudinal
  comparison and drift tracking.
- **Schedule recommendation per student** — picking which courses an
  individual student should register for next. Logic could reuse the
  predecessor map and cohort signals.
- **Trained-model ensemble** — a transformer or gradient-boosted model
  could serve as a residual predictor on top of the statistical baseline
  while preserving interpretability.
- **Real-data validation** — all accuracy figures were measured on
  simulator-generated data; validation against actual HTU enrolment
  history is the most important step before any production deployment.
- **Automated test suites** — current testing is manual across six
  categories. pytest, Vitest, and Playwright suites would establish a
  regression baseline.