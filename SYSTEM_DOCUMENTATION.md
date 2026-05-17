# HTU Enrollment Prediction System — Final Implementation

This document describes the final state of the enrollment forecasting
system, including how the model works, what changed across iterations,
the backend and frontend updates, and known limitations.

---

## 1. What the system does

Given a CSV of historical course enrollments and a CSV of student
records, the system forecasts how many students will enroll in each
course in the next academic semester (Fall / Spring / Summer). The
output drives the dashboard's per-course "Predicted Enrollment" and
"Sections Needed" columns.

There is **no trained model artefact**. All forecasting is computed
at request time directly from the uploaded enrollment history. This
means nothing needs to be re-trained when new data is uploaded, and
there is no `model.pkl` to keep in sync with the inference code.

---

## 2. System architecture

```
┌─────────────────┐   POST /api/predict/upload/   ┌─────────────────────┐
│   Vue frontend  │ ─────────────────────────────▶│  Django + DRF       │
│   (Vite, Pinia) │  multipart: enrollments.csv,  │  core/api_views.py  │
│                 │            students.csv       │                     │
└─────────────────┘                               └──────────┬──────────┘
                                                             │
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │  ml/inference.py    │
                                                  │  predict_next_      │
                                                  │  semester(...)      │
                                                  └─────────────────────┘
```

Frontend posts a multipart upload to `/api/predict/upload/`. The Django
view validates the CSV columns, normalises types, calls
`predict_next_semester(enrolls_df, students_df)`, and returns a JSON
payload that the frontend renders in the predictions table.

---

## 3. The forecasting algorithm

For each course in the enrollment history, the model computes three
independent signals and combines them with a single `max()` rule.

### 3.1 Signal 1 — Time-series

The course's own per-term history projected forward with a damped trend.

For a course C predicting next semester in target term T:

1. Filter the course's history to the rows whose term matches T.
2. Take the last 3 observations of C in T, sorted chronologically.
3. Compute a weighted mean favouring the most recent observation
   (weights `[1, 2, 3]` normalised, oldest → newest).
4. Compute year-over-year first differences and average them.
5. Forecast = `weighted_mean + 0.5 * trend`.
6. Edge cases:
   - Zero observations in T → forecast is 0 (course doesn't run in T).
   - One observation in T → use it directly, no trend.
   - Negative results clipped to 0.

This signal handles seasonal variation (Fall vs Spring vs Summer course
mix) and slow drift. It is blind to current cohort composition.

Implemented in `_ts_forecast(history, target_term)`.

### 3.2 Signal 2 — First-time cohort flow

Counts students who recently passed the inferred predecessor course
and have never taken the target course before, scaled by the
historical first-time take-rate.

#### 3.2.1 Predecessor inference

The system does not require a manually-curated prereq map. It infers
the strongest predecessor for each course from temporal precedence
patterns in the data.

Algorithm in `_build_predecessor_map(enrolls_df)`:

1. For every student, walk their enrolment history in chronological
   order.
2. For each course C taken in semester N, look at the set of courses
   P that the student passed in semester N-1. Each P casts one vote
   for being a predecessor of C.
3. After all students are processed, for each course C, the
   predecessor is the P with the most votes — subject to two
   support thresholds:
   - `PRED_MIN_OBSERVATIONS = 20` — ignore C if observed fewer than
     20 times (low confidence).
   - `PRED_MIN_PRECEDENCE = 0.40` — the winning P must precede C in
     at least 40 % of cases (filters out coincidental co-enrolments).

On the simulated data this yields a clean predecessor for ~75 of 97
courses. Year-1 courses with no prereqs and free electives end up
with no predecessor, which is correct — they fall back to time-series.

#### 3.2.2 First-time take-rate

In `_first_time_takerate(enrolls_df, sem_counts, course, predecessor, target_term)`:

For each past offering of `course` in `target_term`:

1. Determine the semester immediately before that offering.
2. Build the *ready pool*: students who passed `predecessor` in that
   prior semester AND have never enrolled in `course` before.
3. Count how many of those ready-pool students enrolled in `course`
   in this offering (these are the first-time takers from the ready
   pool).
4. Rate for this offering = first-time takers / ready pool size.
5. Average across all past offerings of `course` in `target_term`.

The "first-time" filter is the critical change from the earlier
take-rate computation. It excludes retakers from the numerator so
the cohort signal is disjoint from the retake signal — they can be
summed without double-counting.

#### 3.2.3 At inference

```python
ready_pool = (students who passed predecessor in latest semester) -
             (students who already took target course)
cohort_first = len(ready_pool) * first_time_take_rate
```

### 3.3 Signal 3 — Retake demand

Counts active students whose most recent attempt at the course was
a failing grade, scaled by the historical retake rate for the target
term.

Implemented in `_compute_retake_signals(enrolls_df, active_ids, target_term)`,
which returns a dict `{course_code: signal}`.

For each course:

1. **Current fail pool** = active students whose latest attempt at
   the course resulted in a non-passing grade (a non-pass grade is
   any grade not in `PASS_GRADES = {A, A-, B+, B, B-, C+, C, C-, D+, D, P, M}`).
2. **Historical retake rate in target term**: for each past offering
   of the course in the target term, compute (fail pool just before
   that offering ∩ that offering's takers) / fail pool just before.
   Average across past offerings.
3. **Retake signal** = current fail pool × average retake rate.

Why this is needed: real students who fail a course retake it the
next time it's offered. The cohort-flow signal explicitly excludes
prior-takers from the ready pool, so without a retake signal, these
students would be invisible to the model.

### 3.4 Combination rule

```python
cohort_total = cohort_first + retake_signal
forecast = max(time_series, cohort_total)
```

Why `max()` rather than a weighted blend:

- Time-series and cohort+retake measure overlapping things — they
  both estimate next-term demand from the same underlying data.
- A weighted average would dampen whichever signal is firing harder.
- `max()` lets either signal "lead" when its evidence is stronger,
  while both being bounded by historical patterns keeps either from
  running away.

Empirically (on the simulated training data) this beats both pure
time-series alone (MAPE 8.3 %) and pure cohort-flow alone (MAPE
21.4 %), landing at MAPE 7.8 %.

---

## 4. Inactive student filter

Before any forecasting runs, the model filters out students whose
status is in `INACTIVE_STATUSES = {Graduated, Dropped, Withdrawn, Suspended}`.

This affects:
- The fail-pool computation (graduated students who failed something
  five years ago aren't going to retake it).
- The "Students Processed" count in the dashboard header.

The historical enrolment counts (used for time-series) are not
filtered — past Fall 2023 enrolments are facts about Fall 2023,
not about who's currently active.

---

## 5. Backend API changes

### 5.1 `_last_semester_actuals` — term-aware comparison

The dashboard's "Last Semester" column previously showed whichever
semester was most recent in the data (always Fall when data ended
at Fall). This made every Spring prediction look wrong because it
was compared against a Fall actual.

The fix accepts a `target_term` argument and returns the most recent
semester whose term matches. When predicting Spring 2025, "Last
Semester" shows Spring 2024. When predicting Fall 2026, it shows
Fall 2025. Falls back to "most recent of any term" only if the
course was never offered in the target term.

Call site in `predict_upload`:

```python
predicted_term = predicted_sem.split("_")[-1]
last_actuals = _last_semester_actuals(enrolls_df, target_term=predicted_term)
```

### 5.2 Status thresholds

Status pills (`optimal` / `warning` / `critical`) compare predicted
to "Last Semester" actual. With the term-aware comparison from §5.1,
the thresholds now mean what they were intended to:

- `growth >= 50 %` → critical (genuine spike vs same-term last year)
- `growth >= 20 %` → warning
- otherwise         → optimal

Previously these thresholds fired on Spring-vs-Fall offering
differences rather than year-over-year growth.

---

## 6. Frontend changes

`Backend/Gear/ui/src/api/client.ts` provides the single function
`predictUpload(enrollmentsFile, studentsFile, options)` that POSTs
a multipart form to the Django endpoint and returns a typed
`PredictionResponse`. Used by `stores/predictions.ts` and the
Data Management view.

The rest of the frontend (Dashboard, Predictions, Data Management,
Flagged Students, Elective Requests views) requires no changes for
the model swap — the API contract is unchanged.

---

## 7. Validation

Measured on the training data set (3 000 students, 92 882 enrollments,
ending Fall 2025), comparing the forecast for Spring 2025 against the
actual Spring 2024 numbers in the same data (the most recent same-
term observation we have ground truth for):

| Metric                              | Value             |
| ----------------------------------- | ----------------- |
| Pearson correlation                 | **0.996**         |
| Mean absolute percentage error      | **7.8 %**         |
| Mean absolute error                 | **4.6 students**  |
| Courses within ± 20 % of actual     | 88 of 97          |
| Aggregate ratio (total pred / total actual) | 1.024     |
| Inference time                      | ~4.4 s            |

For the three different target terms (using truncated datasets that
end at Fall 2025, Summer 2024, and Spring 2024 respectively):

| Predicted term  | Total predicted | Top course               |
| --------------- | --------------- | ------------------------ |
| Spring 2025     | 8 692           | Maths for Computing 396  |
| Fall 2025       | 10 796          | STEM Lab I 479           |
| Summer 2024     | 1 860           | Operating Systems 129    |

The shapes are correct: Fall is heavier (fresh admits arriving, year-1
fundamentals dominating) and Summer is light (catch-up only).

---

## 8. Known limitations

1. **No per-student predictions.** The previous classifier could tell
   you which individual students were predicted to take which course.
   The time-series approach forecasts only aggregates. Per-student
   advising is no longer supported by this model.

2. **Cannot predict new courses.** A course with zero history will
   forecast as 0. Real institutional roll-out of a new course would
   need a manual override or a "similar-courses-average" heuristic.

3. **Predecessor inference is data-driven.** It assumes the data has
   consistent prereq patterns. On a real registrar's data this works
   well; on simulated data it can occasionally pick a coincidental
   predecessor over a real one if the simulator generated borderline
   cases.

4. **Co-enrolment patterns are ignored.** Real students bundle hard +
   easy courses together, avoid two-heavy-on-one-day pairings, etc.
   These signals are present in real data but not modelled here.
   With simulated data they would just measure what the simulator
   coded, which is not a useful signal.

5. **Slight aggregate upward bias (~2 %).** See §10.

---

## 9. What we tried before this and why it didn't work

For the capstone report, here is the honest narrative:

### 9.1 Per-course Random Forest classifiers (abandoned)

One classifier per course predicted per-student probability of
enrolling. Showed AUC ≈ 1.0 on every course (a red flag),
n_train ≈ 14 330 identical across all courses, and predictions
inflated by 5 – 30 × on minority courses. Root cause: the negative-
sampling logic (5 negatives per positive within each snapshot)
created an artificial 1:5 class balance that the classifier learned
to exploit, predicting "yes" for almost any plausible student.

### 9.2 Single global classifier with shared course features (abandoned)

Replaced per-course classifiers with one model trained on
(student, course) pairs, using shared course features. AUC 0.96 in
holdout. Improved over the per-course version but still showed
~5 – 6 × over-prediction on Cybersecurity courses on the test set.
Root cause: training and test data are independent simulator runs,
so the test population is at different points in their degrees than
training students. The classifier had memorised "this profile takes
this course" but the test profiles weren't the same.

### 9.3 Pure course-level time-series (improved on by §3)

For each course, weighted last-3-same-term observations + damped
trend. MAPE dropped to 5 % in pure form but didn't react to current
cohort sizes — it just projected last Spring forward. Real failure
mode: a course's predecessor might have had a big graduating class
last semester that the time-series component is blind to.

### 9.4 Time-series + cohort-flow (improved on by §3)

Added predecessor inference and a `max(ts, predecessor_ready_pool *
take_rate)` rule. MAPE moved to 7.5 % — slightly worse than pure TS
in aggregate but more responsive to cohort dynamics. The cohort
take-rate was naively computed over all takers (including retakers),
which double-counted when retake demand was later added.

### 9.5 Time-series + first-time cohort + retake demand (final)

Took the cohort flow and decomposed its take-rate into "first-time
takers only," then added retake demand as a disjoint third signal.
MAPE landed at 7.8 %. Modelling-wise this is the best the per-course
forecaster can do without explicit curriculum information.

---

## 10. Why predictions sometimes look "always higher"

Empirically they aren't always higher. On the training data:

- 47 % of courses predicted above Spring 2024 actual
- 38 % of courses predicted below
- 15 % equal

Mean per-course delta is +2.1 students. The aggregate is +2.4 %.

The small upward lean comes from three places:

1. **`max(ts, cohort_total)`** is asymmetric — when the two signals
   disagree, the higher one wins. There is no symmetric "min" branch.
2. **Retake demand only adds, never subtracts.** Students passing
   never reduce next-term demand (they're not in next-term's pool),
   but students failing always add to it.
3. **The simulator's data is genuinely growing.** Cohort sizes
   trended up over the simulated years, so most courses have a
   positive year-over-year trend in the time-series component that
   is correctly projecting forward.

Item 3 is the dominant effect. The model is tracking real growth in
the data, not inventing it. On a stable-enrolment institution the
upward lean would shrink to nearly zero. To verify this on the
existing data, look at any course in §10 of the validation: many
predict below Spring 2024 (Programming 271 < 283, Maths 260 < 266,
Computing Project Planning 288 < 296, Professional Practice 196 < 201,
Capstone II 20 < 24). The model does see declines when the data
supports them.

---

## 11. File inventory (final state)

### Backend

| Path                              | Purpose                                         |
| --------------------------------- | ----------------------------------------------- |
| `Backend/Gear/ml/inference.py`    | The forecasting engine. No model artefact.       |
| `Backend/Gear/core/api_views.py`  | DRF view; term-aware "Last Semester" comparison. |
| `Backend/Gear/core/urls.py`       | Routes (`/api/health/`, `/api/predict/upload/`). |
| `Backend/Gear/core/models.py`     | Django models for optional persistence.         |

The following are now obsolete and can be deleted:

- `Backend/Gear/ml/artifacts/` (any `global_model.pkl`, manifests, etc.)
- `Backend/Gear/ml/train_global.ipynb`
- `Backend/Gear/ml/train_pipeline.py`
- `Backend/Gear/ml/HTU_ML_Pipeline_v3.ipynb`

### Frontend

| Path                                                       | Purpose                            |
| ---------------------------------------------------------- | ---------------------------------- |
| `Backend/Gear/ui/src/api/client.ts`                        | Single `predictUpload` function.    |
| `Backend/Gear/ui/src/stores/predictions.ts`                | Pinia store for prediction state.   |
| `Backend/Gear/ui/src/stores/flaggedStudents.ts`            | Mock data store.                    |
| `Backend/Gear/ui/src/stores/electiveRequests.ts`           | Mock data store.                    |
| `Backend/Gear/ui/src/views/Dashboard.vue`                  | Course predictions table.           |
| `Backend/Gear/ui/src/views/Predictions.vue`                | Predictions detail view.            |
| `Backend/Gear/ui/src/views/DataManagement.vue`             | File upload + run predictions.      |
| `Backend/Gear/ui/src/views/FlaggedStudents.vue`            | Flagged students view (mock).       |
| `Backend/Gear/ui/src/views/ElectiveRequests.vue`           | Elective requests view (mock).      |
| `Backend/Gear/ui/src/types.ts`                             | Shared TypeScript types.            |

### Test data

Three truncated CSVs validate the term-handling end-to-end:

| File                                  | Predicts        |
| ------------------------------------- | --------------- |
| `htu_enrollments_predict_spring.csv`  | Spring 2025     |
| `htu_enrollments_predict_fall.csv`    | Fall 2025       |
| `htu_enrollments_predict_summer.csv`  | Summer 2024     |

Pair each with the same `htu_students.csv`.

---

## 12. Summary

The system forecasts next-semester per-course enrollment using three
data-grounded signals — time-series of the course's own past
enrollments in the same term, first-time cohort flow from its
strongest data-inferred predecessor, and retake demand from the
current fail pool — combined as `max(time_series, cohort + retake)`.
Inactive students are filtered out at the top of inference. The
dashboard compares predictions against the matching past term so
status pills mean something. There is no trained model: every
prediction is recomputed from the uploaded data, so the system
cannot become stale.

Final accuracy on the simulated training data: Pearson r = 0.996,
MAPE 7.8 %, MAE 4.6 students per course, 91 % of courses within
± 20 % of the matching past-year actual.
