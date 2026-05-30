"""
generate_data.py — HTU Intelligent Registration System data simulator.

Built on the same catalogue, plan definitions, and prerequisite chains as
htu_full_generator_fixed.ipynb (CS / AI / Cybersecurity × Bachelor /
Technical). No invented courses — every entry comes from the actual study
plans.

Differences from the notebook:

  - Single .py file with a CLI (argparse) instead of a notebook
  - Minimum-section-size filter: any (course, semester) combo with
    fewer than 16 distinct enrolled students is dropped from the output
    (the section wouldn't have opened in reality)
  - Three truncated enrolment files instead of one, matching the
    test_data/ folder layout used by the predict_upload endpoint
  - SIM_END_TERM = "Summer" so all three prediction targets have data
    available to truncate against

Run from the project root:

    python generate_data.py --students 3000 --out-dir ./test_data --seed 24

Outputs into test_data/:

    htu_students.csv                       full students table
    htu_enrollments_predict_fall.csv       history truncated before the
                                           most recent Fall — predict Fall
    htu_enrollments_predict_spring.csv     same, truncated before Spring
    htu_enrollments_predict_summer.csv     same, truncated before Summer

Any one of the three enrollment CSVs plus htu_students.csv is a valid
input to /api/predict/upload/.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd


# ═════════════════════════════════════════════════════════════════════════
# A. CATALOGUE — exact copy from htu_full_generator_fixed.ipynb
#    Format: course_code → (name, credit_hours, lecture_hours, lab_hours)
# ═════════════════════════════════════════════════════════════════════════

CATALOG = {
    # ── Remedial (0 credit hours – do NOT count toward graduation) ──
    "0030301110": ("Remedial Arabic Language",                  0, 3, 0),
    "0030301120": ("Pre-Foundation English Intensive + Lab",    0, 4, 0),
    "0030303110": ("Remedial Math",                             0, 2, 0),

    # ── University Compulsory ──────────────────────────────────────
    "0030301111": ("Arabic Language & Communication Skills",    1, 3, 0),
    "0030301121": ("English Pre-Intermediate Intensive + Lab",  4, 4, 0),
    "0030301122": ("English Intermediate",                      3, 4, 0),
    "0030301123": ("English Upper-Intermediate",                3, 4, 0),
    "0030301124": ("English Advanced",                          3, 4, 0),  # BSc only
    "0040302111": ("Professional Skills",                       1, 1, 0),
    "0030302129": ("Military Sciences",                         1, 1, 0),
    "0040302211": ("Professional Practice",                     3, 4, 0),
    "0040302231": ("Entrepreneurship Bootcamp",                 4, 8, 0),
    "0030302232": ("Leadership Camp",                           1, 3, 0),  # BSc only

    # ── University Electives (1 credit each – pick from pool) ─────
    "0030301222": ("Research and Technical Writing",            1, 3, 0),
    "0030302121": ("Science and Society Seminar I",             1, 3, 0),
    "0030302122": ("Arab Contributions to Science and Art",     1, 3, 0),
    "0030302133": ("Jerusalem: History and Civilization",       1, 3, 0),
    "0030302134": ("Jordan: History and Civilization",          1, 3, 0),
    "0040302221": ("Speech and Debate",                         1, 2, 0),
    "0000302222": ("Intro to Philosophy and Critical Thinking", 1, 2, 0),
    "0030302237": ("Free Choice Elective I",                    1, 0, 0),

    # ── College Compulsory (shared by all 6 plans) ─────────────────
    "0040303130": ("Fundamentals of Computing",                 4, 6, 0),
    "0030303111": ("Functional Math",                           3, 3, 0),
    "0030303121": ("STEM Lab I",                                1, 0, 3),
    "0040303121": ("Maths for Computing",                       3, 4, 0),
    "0040201100": ("Programming",                               3, 4, 0),
    "0040201290": ("Computing Project Planning",                4, 4, 0),
    "0040303221": ("Discrete Maths",                            3, 4, 0),

    # ── CS Major Courses ───────────────────────────────────────────
    "0040201200": ("Advanced Programming",                      3, 4, 0),
    "0040201201": ("Data Structures and Algorithms",            3, 4, 0),
    "0040201220": ("Software Development Lifecycles",           3, 4, 0),
    "0040201260": ("Website Design and Development",            3, 4, 0),
    "0040201261": ("Prototyping",                               3, 4, 0),
    "0040201321": ("Systems Analysis and Design",               3, 4, 0),
    "0040201341": ("Operating Systems",                         3, 4, 0),
    "0040201360": ("Application Development",                   3, 4, 0),
    "0000201391": ("Computing Research Project",                6, 6, 0),
    "0010203180": ("Networking",                                3, 4, 0),
    "0000203280": ("Security",                                  3, 4, 0),
    "0010203380": ("Computer Organization and Design",          3, 4, 0),
    "0010204282": ("Database Design and Development",           3, 4, 0),
    "0000204313": ("Business Process Support",                  3, 4, 0),
    "0040201320": ("ERP Systems",                               3, 4, 0),
    "0040201362": ("Games Engine and Scripting",                3, 4, 0),
    "0040201430": ("Database Programming",                      3, 4, 0),
    "0040201440": ("Systems Programming",                       3, 4, 0),
    "0040201491": ("Capstone Project I",                        1, 0, 0),
    "0040201492": ("Capstone Project II",                       2, 0, 0),
    "0040201441": ("Internet of Things",                        3, 4, 0),
    "0040201442": ("Real Time Systems",                         3, 4, 0),
    "0040201450": ("Cloud Computing",                           3, 4, 0),
    "0040201460": ("E-Commerce",                                3, 4, 0),
    "0040201461": ("Mobile Application Development",            3, 4, 0),
    "0040201462": ("Virtual and Augmented Reality Development", 3, 4, 0),

    # ── AI Major Courses ───────────────────────────────────────────
    "0010204210": ("Data Analytics",                            3, 4, 0),
    "0010204250": ("AI and Intelligent Systems",                3, 4, 0),
    "0000204280": ("Principles of Data Science and Computing",  3, 4, 0),
    "0010204281": ("Data Science Programming",                  3, 4, 0),
    "0000204310": ("Advanced Programming for Data Analysis",    3, 4, 0),  # hidden prereq
    "0000204311": ("Big Data Analytics and Visualization",      3, 4, 0),
    "0010204330": ("Modeling and Simulation",                   3, 4, 0),
    "0010204350": ("Machine Learning",                          3, 4, 0),
    "0010204351": ("Natural Language Processing",               3, 4, 0),
    "0000204430": ("Data Mining",                               3, 4, 0),
    "0010204450": ("Deep Learning",                             3, 4, 0),
    "0010204491": ("Capstone Project I (AI)",                   1, 0, 0),
    "0010204492": ("Capstone Project II (AI)",                  2, 0, 0),
    "0000204410": ("Health Informatics",                        3, 4, 0),
    "0000204411": ("Time Series Analysis",                      3, 4, 0),
    "0000204412": ("Applied Analytical Models",                 3, 4, 0),
    "0010204431": ("Information Retrieval",                     3, 4, 0),
    "0010204451": ("Pattern Recognition",                       3, 4, 0),
    "0010204452": ("Computer Vision",                           3, 4, 0),

    # ── Cyber Major Courses ────────────────────────────────────────
    "0010203210": ("Network Security",                          3, 4, 0),
    "0000203360": ("Penetration Testing",                       3, 4, 0),
    "0010203361": ("Forensics",                                 3, 4, 0),
    "0010203362": ("Ethical Hacking",                           3, 4, 0),
    "0010203300": ("Information Security Management",           3, 4, 0),
    "0010203340": ("Cryptography",                              3, 4, 0),
    "0000203400": ("Risk Analysis and Systems Testing",         3, 4, 0),
    "0000203420": ("Secure Coding",                             3, 4, 0),
    "0010203491": ("Capstone Project I (Cyber)",                1, 0, 0),
    "0010203492": ("Capstone Project II (Cyber)",               2, 0, 0),
    "0010203401": ("Incident Response Management",              3, 4, 0),
    "0010203410": ("Mobile and Wireless Security",              3, 4, 0),
    "0010203411": ("Internet of Things Security",               3, 4, 0),
    "0010203420": ("Secure System Design and Development",      3, 4, 0),
    "0010203421": ("Web Security",                              3, 4, 0),

    # ── Apprenticeship Courses (Market Requirements, all plans) ───
    "0000201291": ("Apprenticeship for Computer Science 1",     6, 0, 6),
    "0000201292": ("Apprenticeship for Computer Science 2",     6, 0, 6),
    "0000201392": ("Apprenticeship for Computer Science 3",     6, 0, 6),
    "0000204290": ("Apprenticeship for DS and AI 1",            6, 0, 6),
    "0000204291": ("Apprenticeship for DS and AI 2",            6, 0, 6),
    "0000204391": ("Apprenticeship for DS and AI 3",            6, 0, 6),
    "0000203290": ("Apprenticeship for Cybersecurity 1",        6, 0, 6),
    "0000203291": ("Apprenticeship for Cybersecurity 2",        6, 0, 6),
    "0000203391": ("Apprenticeship for Cybersecurity 3",        6, 0, 6),
}

UNI_ELEC_POOL = [
    "0030301222", "0030302121", "0030302122",
    "0030302133", "0030302134", "0040302221",
    "0000302222", "0030302237",
]


# ═════════════════════════════════════════════════════════════════════════
# B. STUDY PLANS — exact copy from the notebook
# ═════════════════════════════════════════════════════════════════════════

_REMEDIALS     = ["0030301110", "0030301120", "0030303110"]
_UNI_COMP_BSC  = ["0030301111","0030301121","0030301122","0030301123","0030301124",
                  "0040302111","0030302129","0040302211","0040302231","0030302232"]
_UNI_COMP_TECH = ["0030301111","0030301121","0030301122","0030301123",
                  "0040302111","0030302129","0040302211","0040302231"]
_COLLEGE       = ["0040303130","0030303111","0030303121","0040303121",
                  "0040201100","0040201290","0040303221"]

_CS_MAJOR_SHARED = [
    "0040201200","0040201201","0040201220","0040201260","0040201261",
    "0040201321","0040201341","0040201360","0000201391",
    "0010203180","0000203280","0010203380","0010204282","0000204313",
]
_CS_MAJOR_BSC_ONLY = [
    "0040201320","0040201362","0040201430","0040201440",
    "0040201491","0040201492",
]
_CS_ELEC_POOL = ["0040201441","0040201442","0040201450",
                 "0040201460","0040201461","0040201462"]

_AI_MAJOR_SHARED = [
    "0040201201","0040201220","0040201341","0000201391",
    "0010203180","0000203280",
    "0010204210","0010204250","0000204280","0010204281",
    "0010204282","0000204311","0000204313","0010204350",
]
_AI_MAJOR_BSC_ONLY = [
    "0010204330","0010204351","0000204430","0010204450",
    "0010204491","0010204492",
]
_AI_HIDDEN    = ["0000204310"]
_AI_ELEC_POOL = ["0000204410","0000204411","0000204412",
                 "0010204431","0010204451","0010204452"]

_CY_MAJOR_SHARED = [
    "0040201201","0040201220","0040201260","0040201341","0000201391",
    "0010203180","0010203210","0000203280",
    "0000203360","0010203361","0010203362","0010203380",
    "0010204282","0000204313",
]
_CY_MAJOR_BSC_ONLY = [
    "0010203300","0010203340","0000203400","0000203420",
    "0010203491","0010203492",
]
_CY_ELEC_POOL = ["0010203401","0010203410","0010203411",
                 "0010203420","0010203421"]


def _make_plan(total_hrs, uni_comp, uni_elec_count, college,
               major_comp, major_hidden, elec_pool, elec_count,
               app1, app2, app3, cap1, cap2):
    return {
        "total_hrs":      total_hrs,
        "uni_comp":       uni_comp,
        "uni_elec_count": uni_elec_count,
        "remedials":      _REMEDIALS,
        "college":        college,
        "major_comp":     major_comp,
        "major_hidden":   major_hidden,
        "elec_pool":      elec_pool,
        "elec_count":     elec_count,
        "app1": app1, "app2": app2, "app3": app3,
        "cap1": cap1, "cap2": cap2,
    }


PLANS = {
    "CS_Bachelor": _make_plan(
        135, _UNI_COMP_BSC, 3, _COLLEGE,
        _CS_MAJOR_SHARED + _CS_MAJOR_BSC_ONLY, [],
        _CS_ELEC_POOL, 3,
        "0000201291","0000201292","0000201392",
        "0040201491","0040201492",
    ),
    "CS_Technical": _make_plan(
        105, _UNI_COMP_TECH, 1, _COLLEGE,
        _CS_MAJOR_SHARED, [],
        [], 0,
        "0000201291","0000201292","0000201392",
        None, None,
    ),
    "AI_Bachelor": _make_plan(
        135, _UNI_COMP_BSC, 3, _COLLEGE,
        _AI_MAJOR_SHARED + _AI_MAJOR_BSC_ONLY, _AI_HIDDEN,
        _AI_ELEC_POOL, 3,
        "0000204290","0000204291","0000204391",
        "0010204491","0010204492",
    ),
    "AI_Technical": _make_plan(
        105, _UNI_COMP_TECH, 1, _COLLEGE,
        _AI_MAJOR_SHARED, _AI_HIDDEN,
        [], 0,
        "0000204290","0000204291","0000204391",
        None, None,
    ),
    "Cyber_Bachelor": _make_plan(
        135, _UNI_COMP_BSC, 3, _COLLEGE,
        _CY_MAJOR_SHARED + _CY_MAJOR_BSC_ONLY, [],
        _CY_ELEC_POOL, 3,
        "0000203290","0000203291","0000203391",
        "0010203491","0010203492",
    ),
    "Cyber_Technical": _make_plan(
        105, _UNI_COMP_TECH, 1, _COLLEGE,
        _CY_MAJOR_SHARED, [],
        [], 0,
        "0000203290","0000203291","0000203391",
        None, None,
    ),
}


# ═════════════════════════════════════════════════════════════════════════
# C. PREREQUISITES — exact copy from the notebook
# ═════════════════════════════════════════════════════════════════════════

_UNI_PREREQS = [
    ("0030301111", "0030301110", False),
    ("0030301121", "0030301120", False),
    ("0030301122", "0030301121", False),
    ("0030301123", "0030301122", False),
    ("0030301124", "0030301123", False),
    ("0040302211", "0040302111", False),
    ("0040302211", "0030301122", False),
    ("0040302231", "0040302211", False),
    ("0040302231", "0030301123", False),
    ("0030302232", "0040302231", False),
]

_COLLEGE_PREREQS = [
    ("0040201100", "0040303130", False),
    ("0040201290", "0040302211", False),
    ("0030303111", "0030303110", False),
    ("0040303121", "0030303111", False),
    ("0040303221", "0040303121", False),
    ("0040303221", "0040201201", True),
]

_CS_MAJOR_PREREQS = [
    ("0040201200", "0040201100", False),
    ("0040201201", "0040201100", False),
    ("0040201220", "0040201100", False),
    ("0040201260", "0040201100", False),
    ("0040201261", "0040201220", False),
    ("0040201321", "0040201220", False),
    ("0040201341", "0040201201", False),
    ("0040201360", "0040201261", False),
    ("0000201391", "0040201290", False),
    ("0010203180", "0040303130", False),
    ("0000203280", "0040303130", False),
    ("0010203380", "0040201100", False),
    ("0010203380", "0040303221", True),
    ("0010204282", "0040201100", False),
    ("0000204313", "0010204282", False),
    ("0040201320", "0040201360", False),
    ("0040201362", "0040201100", False),
    ("0040201430", "0010204282", False),
    ("0040201430", "0040201201", False),
    ("0040201440", "0040201341", False),
    ("0040201491", "0000201391", False),
    ("0040201492", "0040201491", False),
    ("0000201292", "0000201291", False),
    ("0000201392", "0000201292", False),
]

_AI_MAJOR_PREREQS = [
    ("0040201201", "0040201100", False),
    ("0040201220", "0040201100", False),
    ("0040201341", "0040201201", False),
    ("0000201391", "0040201290", False),
    ("0010203180", "0040303130", False),
    ("0000203280", "0040303130", False),
    ("0010204210", "0040303130", False),
    ("0010204250", "0040303121", False),
    ("0000204280", "0040303121", False),
    ("0010204281", "0040201100", False),
    ("0010204282", "0040201100", False),
    ("0000204310", "0040201100", False),
    ("0000204311", "0000204310", False),
    ("0000204313", "0010204282", False),
    ("0010204350", "0010204210", False),
    ("0010204350", "0000204280", False),
    ("0010204330", "0010204250", False),
    ("0010204351", "0010204250", False),
    ("0000204430", "0010204210", False),
    ("0010204450", "0010204350", False),
    ("0010204491", "0000201391", False),
    ("0010204492", "0010204491", False),
    ("0000204291", "0000204290", False),
    ("0000204391", "0000204291", False),
]

_CY_MAJOR_PREREQS_BSC = [
    ("0040201201", "0040201100", False),
    ("0040201220", "0040201100", False),
    ("0040201260", "0040201100", False),
    ("0040201341", "0040201201", False),
    ("0000201391", "0040201290", False),
    ("0010203180", "0040303130", False),
    ("0010203210", "0000203280", False),
    ("0000203280", "0040303130", False),
    ("0010203362", "0000203280", False),
    ("0000203360", "0010203362", False),
    ("0010203361", "0010203380", False),
    ("0010203361", "0000203280", False),
    ("0010203380", "0040201100", False),
    ("0010203380", "0040303221", True),
    ("0010204282", "0040201100", False),
    ("0000204313", "0010204282", False),
    ("0010203300", "0010203210", False),
    ("0010203340", "0040303221", False),
    ("0000203400", "0000203280", False),
    ("0000203420", "0010204282", False),
    ("0010203491", "0000201391", False),
    ("0010203492", "0010203491", False),
    ("0000203291", "0000203290", False),
    ("0000203391", "0000203291", False),
]

_CY_MAJOR_PREREQS_TECH = [
    r for r in _CY_MAJOR_PREREQS_BSC
    if not (r[0] == "0010203362" and r[1] == "0000203280")
] + [("0010203362", "0010203210", False)]

PLAN_PREREQS = {
    "CS_Bachelor":     _UNI_PREREQS + _COLLEGE_PREREQS + _CS_MAJOR_PREREQS,
    "CS_Technical":    _UNI_PREREQS + _COLLEGE_PREREQS + _CS_MAJOR_PREREQS,
    "AI_Bachelor":     _UNI_PREREQS + _COLLEGE_PREREQS + _AI_MAJOR_PREREQS,
    "AI_Technical":    _UNI_PREREQS + _COLLEGE_PREREQS + _AI_MAJOR_PREREQS,
    "Cyber_Bachelor":  _UNI_PREREQS + _COLLEGE_PREREQS + _CY_MAJOR_PREREQS_BSC,
    "Cyber_Technical": _UNI_PREREQS + _COLLEGE_PREREQS + _CY_MAJOR_PREREQS_TECH,
}


# ═════════════════════════════════════════════════════════════════════════
# D. SIMULATION CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════

# Grade system — same as the notebook
GRADE_SYMBOLS = ["D",   "M",   "P",   "U",   "W"]
GRADE_POINTS  = {"D": 4.0, "M": 3.2, "P": 2.4, "U": 1.6, "W": 0.0}
IS_PASSING    = {"D": True, "M": True, "P": True, "U": False, "W": False}
GPA_INCLUDED  = {"D": True, "M": True, "P": True, "U": True,  "W": False}
GRADE_PROBS   = [0.17, 0.33, 0.43, 0.05, 0.02]

ADMISSION_YEARS  = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
ADM_YEAR_WEIGHTS = [0.10, 0.12, 0.15, 0.18, 0.18, 0.15, 0.12]

# Extended to Summer so all three prediction-target files have data
SIM_END_YEAR = 2025
SIM_END_TERM = "Summer"

PROB_REM_ARABIC  = 0.18
PROB_REM_ENGLISH = 0.25
PROB_REM_MATH    = 0.15

SUMMER_SKIP_CLEAN = 0.55
SUMMER_SKIP_FAIL  = 0.20

# Per-semester credit load distribution — mean ≈ 14.9, mode = 15
REG_CREDIT_CHOICES = list(range(12, 19))
REG_CREDIT_WEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.25, 0.07, 0.03]

# Section-minimum (realistic constraint: sections under this size don't open)
MIN_STUDENTS_PER_SECTION = 16


# ═════════════════════════════════════════════════════════════════════════
# E. SIMULATION CORE — copied from the notebook
# ═════════════════════════════════════════════════════════════════════════

def zpad(code):
    s = str(code).strip()
    return s.zfill(10) if s.isdigit() else s


def make_semester_sequence(start_year, end_year=SIM_END_YEAR, end_term=SIM_END_TERM):
    """Fall(Y) → Spring(Y) → Summer(Y) → Fall(Y+1) → … in HTU academic order."""
    ORDER = {"Fall": 1, "Spring": 2, "Summer": 3}
    stop_key = end_year * 10 + ORDER[end_term]
    y = start_year
    while True:
        for (cy, term) in [(y, "Fall"), (y, "Spring"), (y, "Summer")]:
            if cy * 10 + ORDER[term] > stop_key:
                return
            yield (cy, term)
        y += 1


def init_student_plan(plan_key, needs_rem_arabic, needs_rem_english, needs_rem_math):
    p = PLANS[plan_key]
    uni_elecs = random.sample(UNI_ELEC_POOL, p["uni_elec_count"])
    maj_elecs = (random.sample(p["elec_pool"], p["elec_count"])
                 if p["elec_count"] > 0 else [])

    personal_plan = (
        p["uni_comp"] + uni_elecs + _COLLEGE
        + [p["app1"], p["app2"], p["app3"]]
        + p["major_hidden"]
        + p["major_comp"]
        + maj_elecs
    )

    remedials_needed = []
    if needs_rem_arabic:  remedials_needed.append("0030301110")
    if needs_rem_english: remedials_needed.append("0030301120")
    if needs_rem_math:    remedials_needed.append("0030303110")

    grad_required = set(
        p["uni_comp"] + uni_elecs + _COLLEGE
        + [p["app1"], p["app2"], p["app3"]]
        + p["major_comp"] + maj_elecs
    )

    pre_passed = set()
    if not needs_rem_arabic:  pre_passed.add("0030301110")
    if not needs_rem_english: pre_passed.add("0030301120")
    if not needs_rem_math:    pre_passed.add("0030303110")

    full_codes = set(personal_plan) | set(remedials_needed) | pre_passed
    prereq_map = {}
    for (cc, pc, conc) in PLAN_PREREQS[plan_key]:
        if cc in full_codes:
            prereq_map.setdefault(cc, []).append((pc, conc))

    return {
        "full_plan":     remedials_needed + personal_plan,
        "grad_required": grad_required,
        "pre_passed":    pre_passed,
        "prereq_map":    prereq_map,
        "credit_map":    {c: CATALOG[c][1] for c in (remedials_needed + personal_plan) if c in CATALOG},
        "name_map":      {c: CATALOG[c][0] for c in (remedials_needed + personal_plan) if c in CATALOG},
        "app1": p["app1"], "app2": p["app2"], "app3": p["app3"],
        "cap1": p["cap1"], "cap2": p["cap2"],
        "total_hrs":     p["total_hrs"],
        "major_hidden":  set(p["major_hidden"]),
    }


def get_eligible(plan_state, passed):
    """Two-pass eligibility: hard prereqs, then co-reqs."""
    prereq_map = plan_state["prereq_map"]
    all_codes  = set(plan_state["full_plan"])

    pass1 = set()
    for c in all_codes:
        if c in passed: continue
        hard_reqs = [pc for (pc, conc) in prereq_map.get(c, []) if not conc]
        if all(r in passed for r in hard_reqs):
            pass1.add(c)

    eligible = set()
    for c in pass1:
        coreqs = [pc for (pc, conc) in prereq_map.get(c, []) if conc]
        if all(r in passed or r in pass1 for r in coreqs):
            eligible.add(c)
    return eligible


def select_courses(eligible, fail_retry, credit_limit, credit_map):
    """Greedy: retry-priority first, then remaining eligible. 0-cr always included."""
    ordered = (
        [c for c in fail_retry if c in eligible]
        + [c for c in eligible if c not in fail_retry]
    )
    selected, used = [], 0
    for c in ordered:
        ch = credit_map.get(c, 3)
        if ch == 0:
            selected.append(c)
        elif used + ch <= credit_limit:
            selected.append(c)
            used += ch
    return selected


def sem_gpa(records):
    pts = sum(GRADE_POINTS[r["grade"]] * r["credit_hours"]
              for r in records if GPA_INCLUDED[r["grade"]] and r["credit_hours"] > 0)
    hrs = sum(r["credit_hours"] for r in records
              if GPA_INCLUDED[r["grade"]] and r["credit_hours"] > 0)
    return round(pts / hrs, 2) if hrs > 0 else 0.0


def cum_gpa(all_records):
    """Last-attempt-wins for retakes; W excluded."""
    latest = {}
    for r in all_records:
        if r["grade"] == "W": continue
        cid = r["course_code"]
        if cid not in latest or not IS_PASSING[latest[cid]["grade"]]:
            latest[cid] = r
    pts = sum(GRADE_POINTS[v["grade"]] * v["credit_hours"]
              for v in latest.values() if v["credit_hours"] > 0)
    hrs = sum(v["credit_hours"] for v in latest.values() if v["credit_hours"] > 0)
    return round(pts / hrs, 2) if hrs > 0 else 0.0


def run_simulation(num_students):
    plan_keys        = list(PLANS.keys())
    students_data    = []
    enrollments_data = []

    for s_idx in range(1, num_students + 1):
        plan_key = random.choice(plan_keys)
        adm_year = random.choices(ADMISSION_YEARS, weights=ADM_YEAR_WEIGHTS)[0]
        student_id = f"20{adm_year % 100:02d}{s_idx:05d}"

        needs_rem_ar = random.random() < PROB_REM_ARABIC
        needs_rem_en = random.random() < PROB_REM_ENGLISH
        needs_rem_ma = random.random() < PROB_REM_MATH

        ps = init_student_plan(plan_key, needs_rem_ar, needs_rem_en, needs_rem_ma)

        passed         = set(ps["pre_passed"])
        failed_count   = {}
        all_history    = []
        cum_passed_hrs = 0
        status         = "Active"
        sem_number     = 0
        last_cum_gpa   = 0.0

        for (year, term) in make_semester_sequence(adm_year):
            if status == "Graduated":
                break
            is_summer = (term == "Summer")

            has_fails = any(v > 0 for v in failed_count.values())
            if is_summer:
                skip_p = SUMMER_SKIP_FAIL if has_fails else SUMMER_SKIP_CLEAN
                if random.random() < skip_p:
                    continue

            if ps["grad_required"].issubset(passed):
                status = "Graduated"
                break

            eligible = get_eligible(ps, passed)
            if not eligible:
                if ps["grad_required"].issubset(passed):
                    status = "Graduated"
                break

            remaining_hrs = sum(
                CATALOG[c][1] for c in ps["grad_required"]
                if c not in passed and c in CATALOG
            )
            if is_summer:
                if remaining_hrs <= 12:
                    credit_limit = min(remaining_hrs, 12)
                else:
                    credit_limit = random.randint(3, 9)
            else:
                if remaining_hrs <= 21:
                    credit_limit = min(remaining_hrs, 21)
                else:
                    credit_limit = random.choices(
                        REG_CREDIT_CHOICES, weights=REG_CREDIT_WEIGHTS)[0]

            fail_retry = [c for c in eligible if failed_count.get(c, 0) > 0]
            registered = select_courses(eligible, fail_retry, credit_limit, ps["credit_map"])
            if not registered:
                continue

            sem_number += 1
            sem_label   = f"{year}_{term}"
            sem_records = []

            for c in registered:
                ch    = ps["credit_map"].get(c, 0)
                cname = ps["name_map"].get(c, c)
                grade = random.choices(GRADE_SYMBOLS, weights=GRADE_PROBS)[0]
                rec = {
                    "student_id":            student_id,
                    "course_code":           c,
                    "course_name":           cname,
                    "credit_hours":          ch,
                    "semester_label":        sem_label,
                    "year":                  year,
                    "term":                  term,
                    "semester_number":       sem_number,
                    "grade":                 grade,
                    "is_retake":             failed_count.get(c, 0) > 0,
                    "cum_passed_hrs_before": cum_passed_hrs,
                }
                sem_records.append(rec)
                all_history.append(rec)
                if IS_PASSING[grade]:
                    passed.add(c)
                    failed_count.pop(c, None)
                elif grade in ("U", "W"):
                    failed_count[c] = failed_count.get(c, 0) + 1

            for rec in sem_records:
                if IS_PASSING[rec["grade"]]:
                    cum_passed_hrs += rec["credit_hours"]

            s_gpa = sem_gpa(sem_records)
            c_gpa = cum_gpa(all_history)
            last_cum_gpa = c_gpa
            for rec in sem_records:
                rec["sem_gpa"] = s_gpa
                rec["cum_gpa"] = c_gpa

            enrollments_data.extend(sem_records)

            if ps["grad_required"].issubset(passed):
                status = "Graduated"
                break

        students_data.append({
            "student_id":        student_id,
            "name":              f"Student {s_idx:05d}",
            "plan_key":          plan_key,
            "major":             plan_key.split("_")[0],
            "degree_type":       plan_key.split("_")[1],
            "admission_year":    adm_year,
            "cumulative_gpa":    last_cum_gpa,
            "earned_credit_hours": cum_passed_hrs,
            "is_graduating_flag": status == "Graduated",
            "needs_rem_arabic":  int(needs_rem_ar),
            "needs_rem_english": int(needs_rem_en),
            "needs_rem_math":    int(needs_rem_ma),
            "status":            status,
            "semesters_taken":   sem_number,
        })

    df_s = pd.DataFrame(students_data)
    df_e = pd.DataFrame(enrollments_data)
    df_e["course_code"] = df_e["course_code"].astype(str).apply(zpad)
    return df_s, df_e


# ═════════════════════════════════════════════════════════════════════════
# F. POST-PROCESSING — section filter + truncation for 3-file output
# ═════════════════════════════════════════════════════════════════════════

TERM_RANK = {"Fall": 0, "Spring": 1, "Summer": 2}


def _safe_csv(df, path):
    """Write CSV with code columns quoted so leading zeros are preserved
    when the file is read back without dtype hints."""
    import csv
    code_cols = [c for c in ["course_code", "student_id"] if c in df.columns]
    if code_cols:
        # Force code columns to string before write so pandas doesn't
        # re-infer them as numeric and drop the leading zero
        for col in code_cols:
            df[col] = df[col].astype(str)
        df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    else:
        df.to_csv(path, index=False)


def filter_min_section_size(df, min_students):
    """Drop every row belonging to a (course, semester) group with fewer than
    `min_students` distinct enrolled students."""
    sizes = (
        df.groupby(["course_code", "semester_label"])["student_id"]
        .nunique()
    )
    too_small = sizes[sizes < min_students].index
    if len(too_small) == 0:
        return df
    keep_mask = ~df.set_index(["course_code", "semester_label"]).index.isin(too_small)
    return df.loc[keep_mask].reset_index(drop=True)


def truncate_before_latest_term(df, target_term):
    """Drop all rows at or after the most recent occurrence of target_term
    (using HTU academic-year ordering: Fall < Spring < Summer within a year)."""
    matching = df.loc[df["term"] == target_term]
    if matching.empty:
        return df.copy()
    cutoff_year = int(matching["year"].max())
    cutoff_rank = TERM_RANK[target_term]
    rank = df["term"].map(TERM_RANK)
    keep = (df["year"] < cutoff_year) | (
        (df["year"] == cutoff_year) & (rank < cutoff_rank)
    )
    return df.loc[keep].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════
# G. DRIVER
# ═════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=3000)
    ap.add_argument("--out-dir",  default="./test_data")
    ap.add_argument("--seed",     type=int, default=24)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.students} students …")
    df_students, df_enrollments_raw = run_simulation(args.students)
    print(f"  Raw enrolment rows: {len(df_enrollments_raw):,}")

    # Apply section minimum
    df_enrollments = filter_min_section_size(df_enrollments_raw, MIN_STUDENTS_PER_SECTION)
    rows_dropped = len(df_enrollments_raw) - len(df_enrollments)

    # Write students CSV
    students_path = out_dir / "htu_students.csv"
    _safe_csv(df_students, students_path)

    # Write three truncated enrolment CSVs
    outputs = []
    for term, fname in [
        ("Fall",   "htu_enrollments_predict_fall.csv"),
        ("Spring", "htu_enrollments_predict_spring.csv"),
        ("Summer", "htu_enrollments_predict_summer.csv"),
    ]:
        truncated = truncate_before_latest_term(df_enrollments, term)
        path = out_dir / fname
        _safe_csv(truncated, path)

        latest_year = (
            int(truncated["year"].max()) if not truncated.empty else None
        )
        latest_term = "—"
        if latest_year is not None:
            terms_in_latest_year = truncated.loc[
                truncated["year"] == latest_year, "term"
            ].unique()
            latest_term = max(terms_in_latest_year, key=lambda t: TERM_RANK[t])
        outputs.append((term, path, len(truncated), latest_year, latest_term))

    # Summary
    sem_loads = (
        df_enrollments.loc[df_enrollments["credit_hours"] > 0]
        .groupby(["student_id", "semester_label"])["credit_hours"]
        .sum()
    )
    sem_terms_map = (
        df_enrollments
        .groupby(["student_id", "semester_label"])["term"]
        .first()
    )
    fs_loads = sem_loads[sem_terms_map.isin(["Fall", "Spring"])]
    sm_loads = sem_loads[sem_terms_map == "Summer"]

    print()
    print(f"Students:                  {len(df_students):,}")
    print(f"Enrolment rows (raw):      {len(df_enrollments_raw):,}")
    print(f"Dropped (section < {MIN_STUDENTS_PER_SECTION}):    {rows_dropped:,}")
    print(f"Enrolment rows (kept):     {len(df_enrollments):,}")
    print(f"Mean cr/sem (Fall+Spring): {fs_loads.mean():.2f}")
    print(f"Mean cr/sem (Summer):      {sm_loads.mean():.2f}" if len(sm_loads) else "Mean cr/sem (Summer):      —")
    print(f"Distinct courses kept:     {df_enrollments['course_code'].nunique()}")
    print()
    print(f"Wrote:")
    print(f"  {students_path}  ({len(df_students):,} students)")
    for term, path, n_rows, ly, lt in outputs:
        print(f"  {path}  ({n_rows:,} rows, ends {ly} {lt}, predicts next {term})")


if __name__ == "__main__":
    main()