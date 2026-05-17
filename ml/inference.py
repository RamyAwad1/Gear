"""
Course-level enrollment forecasting — time-series + cohort-flow + retake.

For each course we compute three independent signals and combine them:

  1. Time-series:     the course's own per-term history projected forward
                      with a damped trend. Handles seasonality and slow
                      growth/decay; blind to current cohort composition.

  2. Cohort-flow:     count students who just passed the inferred
                      predecessor course and have never taken this one
                      ("first-time ready pool"), multiplied by the
                      historical first-time take-rate for that
                      (predecessor -> course, target term) pair. Handles
                      cohort dynamics for new takers.

  3. Retake demand:   active students whose most recent attempt at this
                      course was a failing grade, multiplied by the
                      historical retake rate for the target term. Captures
                      students who failed and will likely re-enroll.

Combined as: forecast = max(time_series, cohort_flow + retake_demand).
The cohort + retake sum is disjoint by construction (first-timers and
re-takers don't overlap), so they sum cleanly. The max with time-series
provides a safety net for courses where the cohort signal is weak or
predecessor inference failed.

No model artefact: everything is computed at request time from the
uploaded enrollment history. Exposes the same `predict_next_semester`
signature and four-tuple return as the previous versions, so
core/api_views.py doesn't change.
"""
import math

import numpy as np
import pandas as pd

TERM_RANK = {"Fall": 1, "Spring": 2, "Summer": 3}
PASS_GRADES = {"A", "A-", "B+", "B", "B-", "C+", "C", "C-",
               "D+", "D", "P", "M"}
INACTIVE_STATUSES = {"Graduated", "Dropped", "Withdrawn", "Suspended"}

# Predecessor inference: a course P is a predecessor of C if students
# who take C reliably took P one semester before. These thresholds tune
# how strict that requirement is.
PRED_MIN_OBSERVATIONS = 20      # ignore C if seen < N times
PRED_MIN_PRECEDENCE   = 0.40    # P must precede C in >= 40% of (P then C) pairs
PRED_MIN_TEMPORAL    = 0.85    # P must be taken before C in 85%+ of cases


def _next_semester_label(latest_year, latest_term):
    """HTU calendar: Fall(Y) -> Spring(Y) -> Summer(Y) -> Fall(Y+1)."""
    if latest_term == "Fall":
        return f"{latest_year}_Spring"
    if latest_term == "Spring":
        return f"{latest_year}_Summer"
    return f"{latest_year + 1}_Fall"


def _ts_forecast(history, target_term):
    """Pure per-course time-series forecast — same as the previous
    version. Used as the baseline that cohort-flow scales."""
    same_term = history[history["term"] == target_term].sort_values("sem_key")
    n = len(same_term)
    if n == 0:
        return 0.0
    if n == 1:
        return float(same_term["n_students"].iloc[0])
    recent = same_term.tail(3)["n_students"].astype(float).values
    weights = np.array([1.0, 2.0, 3.0])[-len(recent):]
    weights /= weights.sum()
    baseline = float(np.dot(recent, weights))
    diffs = np.diff(recent)
    trend = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    return max(0.0, baseline + trend * 0.5)


def _build_predecessor_map(enrolls_df):
    """
    For each course C, identify the single predecessor P that students
    most reliably complete one semester before taking C.

    Returns {course_code -> predecessor_code} for courses where a clear
    predecessor exists. Courses without a clear predecessor (year-1
    courses with no prereqs, fresh-elective entry points, etc.) are
    omitted, and those will fall back to pure time-series at forecast.

    Algorithm: for each student, walk their enrollment history in order.
    Every time they take course C at semester S, look at the courses
    they passed in semester S-1; each one casts a "vote" for being C's
    predecessor. Then for each C, the predecessor is the course with
    the highest vote count that also satisfies a minimum-support
    threshold.
    """
    passed = enrolls_df[enrolls_df["passed"]].copy()
    passed = passed.sort_values(["student_id", "sem_key"])

    # For each student, build the chronological list of (sem_key, passed_set).
    sem_passed = (
        passed.groupby(["student_id", "sem_key"])["course_code"]
        .apply(set).reset_index(name="passed_set")
    )

    # For each (student, sem_key), record what they took in the NEXT semester.
    taken = enrolls_df.sort_values(["student_id", "sem_key"]).copy()
    taken_by_sem = (
        taken.groupby(["student_id", "sem_key"])["course_code"]
        .apply(set).reset_index(name="taken_set")
    )

    # Build votes: for each student, for each (prev_sem, next_sem) pair,
    # every passed-course in prev_sem casts a vote for every course in next_sem.
    from collections import defaultdict
    votes = defaultdict(lambda: defaultdict(int))  # C -> {P -> count}
    c_total = defaultdict(int)                     # how many times C was taken

    by_student_passed = {
        sid: dict(zip(g["sem_key"], g["passed_set"]))
        for sid, g in sem_passed.groupby("student_id", sort=False)
    }
    by_student_taken = {
        sid: dict(zip(g["sem_key"], g["taken_set"]))
        for sid, g in taken_by_sem.groupby("student_id", sort=False)
    }

    for sid, taken_dict in by_student_taken.items():
        passed_dict = by_student_passed.get(sid, {})
        sorted_sems = sorted(taken_dict.keys())
        for i, sk in enumerate(sorted_sems):
            if i == 0:
                continue   # need a previous semester
            prev_sk = sorted_sems[i - 1]
            prev_passed = passed_dict.get(prev_sk, set())
            curr_taken = taken_dict[sk]
            for C in curr_taken:
                c_total[C] += 1
                for P in prev_passed:
                    if P == C:
                        continue
                    votes[C][P] += 1

    # Build the predecessor map under the support thresholds.
    pred_map = {}
    for C, total in c_total.items():
        if total < PRED_MIN_OBSERVATIONS:
            continue
        candidates = votes[C]
        if not candidates:
            continue
        best_P, best_count = max(candidates.items(), key=lambda kv: kv[1])
        precedence = best_count / total
        if precedence < PRED_MIN_PRECEDENCE:
            continue
        pred_map[C] = best_P

    return pred_map


def _first_time_takerate(enrolls_df, sem_counts, course, predecessor, target_term):
    """
    Like the previous historical-takerate function but counts ONLY
    first-time takers in the numerator. A first-time taker is a student
    who has never enrolled in `course` before this offering. This makes
    the cohort-flow signal disjoint from the retake-demand signal so the
    two can be summed without double-counting.

    Returns (avg_first_time_rate, n_observations). None if no usable
    historical observations of `course` in `target_term`.
    """
    past_offerings = sem_counts[
        (sem_counts["course_code"] == course)
        & (sem_counts["term"] == target_term)
    ]["sem_key"].tolist()

    course_enrolls = enrolls_df[enrolls_df["course_code"] == course]
    pred_enrolls  = enrolls_df[enrolls_df["course_code"] == predecessor]

    rates = []
    all_sem_keys = sorted(sem_counts["sem_key"].unique())
    for off_sk in past_offerings:
        prev_candidates = [s for s in all_sem_keys if s < off_sk]
        if not prev_candidates:
            continue
        prev_sk = max(prev_candidates)

        # Ready pool: passed predecessor recently AND never touched target
        ready = pred_enrolls[(pred_enrolls["sem_key"] == prev_sk)
                             & (pred_enrolls["passed"])]
        ready_ids = set(ready["student_id"].astype(str))
        already = set(
            course_enrolls[course_enrolls["sem_key"] < off_sk]
            ["student_id"].astype(str)
        )
        ready_pool = ready_ids - already
        if len(ready_pool) < 5:
            continue

        # First-time takers = this-offering takers minus prior-touchers
        takers = set(
            course_enrolls[course_enrolls["sem_key"] == off_sk]
            ["student_id"].astype(str)
        )
        first_time_from_ready = (takers - already) & ready_pool
        rates.append(len(first_time_from_ready) / len(ready_pool))

    if not rates:
        return None, 0
    return float(np.mean(rates)), len(rates)

def _compute_retake_pools(enrolls_df, active_ids, target_term):
    """
    For each course, estimate next-term retake demand AND identify the
    specific students in the current fail pool (for per-student listings).

    Returns {course_code: {"signal": float, "fail_pool": set[str],
                           "retake_rate": float}}.

    current_fail_pool[c]: active students whose LATEST attempt at c was
                          a failing grade (i.e. haven't subsequently passed).
    historical_retake_rate[c, term]: of the fail pool just before each
                          past offering of c in `target_term`, what
                          fraction re-enrolled in that offering.
    signal = len(current_fail_pool) * historical_retake_rate.
    """
    pools = {}
    for cc, ch in enrolls_df.groupby("course_code", sort=False):
        ch = ch.sort_values(["student_id", "sem_key"])
        latest_per_student = ch.groupby("student_id").tail(1)
        failers_latest = set(
            latest_per_student[~latest_per_student["passed"]]
            ["student_id"].astype(str)
        )
        current_fail_pool = failers_latest & active_ids
        if not current_fail_pool:
            pools[cc] = {"signal": 0.0, "fail_pool": set(), "retake_rate": 0.0}
            continue

        past_offerings_in_term = (
            ch[ch["term"] == target_term].groupby("sem_key")["student_id"]
            .apply(set).to_dict()
        )
        if not past_offerings_in_term:
            pools[cc] = {"signal": 0.0, "fail_pool": current_fail_pool, "retake_rate": 0.0}
            continue

        rates = []
        for off_sk, takers in past_offerings_in_term.items():
            prior = ch[ch["sem_key"] < off_sk]
            if len(prior) == 0:
                continue
            last_prior = prior.groupby("student_id").tail(1)
            fail_pool_then = set(
                last_prior[~last_prior["passed"]]["student_id"].astype(str)
            )
            if len(fail_pool_then) < 3:
                continue
            retook = fail_pool_then & set(map(str, takers))
            rates.append(len(retook) / len(fail_pool_then))

        if not rates:
            pools[cc] = {"signal": 0.0, "fail_pool": current_fail_pool, "retake_rate": 0.0}
            continue

        avg_rate = float(np.mean(rates))
        pools[cc] = {
            "signal": len(current_fail_pool) * avg_rate,
            "fail_pool": current_fail_pool,
            "retake_rate": avg_rate,
        }
    return pools


def predict_next_semester(
    enrolls_df, students_df,
    next_sem_label=None, section_cap=None, buffer_pct=None,
    count_mode="probability",
):
    """Returns (student_preds_df, demand_df, sections_df, latest_snap)."""

    if section_cap is None:
        section_cap = 30
    if buffer_pct is None:
        buffer_pct = 0.10

    # ── Normalize inputs ────────────────────────────────────────────────
    enrolls_df = enrolls_df.copy()
    students_df = students_df.copy()
    enrolls_df["course_code"] = enrolls_df["course_code"].astype(str).str.zfill(10)
    enrolls_df["sem_key"] = (
        enrolls_df["year"].astype(int) * 10
        + enrolls_df["term"].map(TERM_RANK).fillna(1).astype(int)
    )
    enrolls_df["passed"] = enrolls_df["grade"].astype(str).isin(PASS_GRADES)

    if "status" in students_df.columns:
        before = len(students_df)
        students_df = students_df[
            ~students_df["status"].astype(str).isin(INACTIVE_STATUSES)
        ].reset_index(drop=True)
        removed = before - len(students_df)
        if removed:
            print(f"[inference-ts] filtered {removed:,} inactive students; "
                  f"{len(students_df):,} active remain")

    if len(enrolls_df) == 0:
        empty = pd.DataFrame()
        return empty, empty, empty, pd.DataFrame()

    # ── Determine target semester ───────────────────────────────────────
    latest_sem_key = enrolls_df["sem_key"].max()
    latest_row = enrolls_df[enrolls_df["sem_key"] == latest_sem_key].iloc[0]
    latest_year = int(latest_row["year"])
    latest_term = str(latest_row["term"])
    if next_sem_label is None:
        next_sem_label = _next_semester_label(latest_year, latest_term)
    next_term = next_sem_label.split("_")[-1]

    # ── Per-course historical time series ───────────────────────────────
    sem_counts = (
        enrolls_df.groupby(
            ["course_code", "semester_label", "year", "term", "sem_key"]
        )["student_id"].nunique().reset_index(name="n_students")
    )

    # ── Infer predecessors from the data ────────────────────────────────
    pred_map = _build_predecessor_map(enrolls_df)
    print(f"[inference-ts] inferred predecessors for {len(pred_map)}/"
          f"{sem_counts['course_code'].nunique()} courses")

    # ── Active students' currently-passed sets (for ready-pool counting) ─
    active_ids = set(students_df["student_id"].astype(str))
    active_enrolls = enrolls_df[
        enrolls_df["student_id"].astype(str).isin(active_ids)
    ]
    # passed_in_latest[P] = set of active student_ids who passed P in latest sem
    passed_in_latest = (
        active_enrolls[
            (active_enrolls["sem_key"] == latest_sem_key)
            & (active_enrolls["passed"])
        ].groupby("course_code")["student_id"].apply(set).to_dict()
    )
    # already_took[C] = set of active student_ids who have ever taken C
    already_took = (
        active_enrolls.groupby("course_code")["student_id"].apply(set).to_dict()
    )

    # ── Per-course retake demand and fail pools ─────────────────────────
    # current_fail_pool * historical retake rate in target term.
    # Disjoint from the cohort-flow signal (which counts first-timers only),
    # so the two are summed cleanly. We also keep the fail_pool sets so we
    # can list specific predicted-retake students per course.
    active_ids_set = set(students_df["student_id"].astype(str))
    retake_pools = _compute_retake_pools(enrolls_df, active_ids_set, next_term)
    print(f"[inference-ts] retake demand computed for "
          f"{sum(1 for v in retake_pools.values() if v['signal'] > 0)} courses")

    # ── Forecast each course and track per-student candidates ───────────
    forecasts = []
    student_preds_rows = []
    for cc, ch in sem_counts.groupby("course_code"):
        ts_base = _ts_forecast(ch, next_term)
        pred = pred_map.get(cc)
        cohort_first = 0.0
        ready_pool = set()
        first_time_rate = 0.0

        if pred is not None:
            ft_rate, n_obs = _first_time_takerate(
                enrolls_df, sem_counts, cc, pred, next_term
            )
            if ft_rate is not None and n_obs >= 1:
                ready_pool = (
                    passed_in_latest.get(pred, set())
                    - already_took.get(cc, set())
                )
                first_time_rate = ft_rate
                cohort_first = len(ready_pool) * ft_rate

        retake_info = retake_pools.get(cc, {})
        retake_signal_val = retake_info.get("signal", 0.0)
        fail_pool = retake_info.get("fail_pool", set())
        retake_rate_val = retake_info.get("retake_rate", 0.0)

        # Combined cohort = first-time-from-predecessor + retake demand.
        # The two summands are disjoint by construction:
        #   - cohort_first counts students who passed P and never took C
        #   - retake counts students whose latest C grade was a fail
        cohort_total = cohort_first + retake_signal_val

        # max(ts, cohort): time-series captures stable seasonal patterns;
        # cohort captures "this cohort is unusually big/small" plus retake
        # demand. Whichever signal fires more strongly wins. Both are
        # bounded by real historical patterns so neither runs away.
        forecast = max(ts_base, cohort_total)

        forecasts.append({
            "course_code":     cc,
            "predicted_count": max(0, int(round(forecast))),
        })

        # Record per-student candidates for this course. Each row has the
        # student_id, the course, the reason they're a candidate, and a
        # rough confidence score (the historical conversion rate for that
        # signal). The API view enriches these with student metadata.
        for sid in ready_pool:
            student_preds_rows.append({
                "student_id":      str(sid),
                "course_code":     cc,
                "reason":          "passed_predecessor",
                "confidence":      round(float(first_time_rate), 3),
                "next_sem":        next_sem_label,
            })
        for sid in fail_pool:
            student_preds_rows.append({
                "student_id":      str(sid),
                "course_code":     cc,
                "reason":          "retake_candidate",
                "confidence":      round(float(retake_rate_val), 3),
                "next_sem":        next_sem_label,
            })

    demand_df = pd.DataFrame(forecasts)

    # ── Sections ────────────────────────────────────────────────────────
    def _sections(n):
        return math.ceil((n * (1.0 + buffer_pct)) / max(section_cap, 1))

    sections_df = demand_df.copy()
    sections_df["predicted_sections"] = (
        sections_df["predicted_count"].apply(_sections)
    )

    student_preds_df = pd.DataFrame(student_preds_rows) if student_preds_rows \
                       else pd.DataFrame(columns=["student_id", "course_code",
                                                  "reason", "confidence",
                                                  "next_sem"])
    if "next_sem" not in student_preds_df.columns or len(student_preds_df) == 0:
        # Edge case: no identified candidates anywhere. Keep the next_sem
        # field populated so api_views can still extract the predicted term.
        student_preds_df = pd.DataFrame({"next_sem": [next_sem_label]})

    latest_snap = pd.DataFrame()
    return student_preds_df, demand_df, sections_df, latest_snap