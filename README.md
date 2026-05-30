# Gear: Intelligent Registration System

**Gear** is an intelligent course registration and resource allocation system developed for Al Hussein Technical University (HTU).

Designed to eliminate manual, guess-based scheduling, Gear forecasts per-course student demand for an upcoming semester directly from enrolment history, recommends the number of sections each course needs, and surfaces students at risk of delayed graduation. The forecasting engine is a custom three-signal statistical model — no trained ML artefact, no caching, fully reproducible from a single CSV upload.

**Team:** Saeed Iqtaish, Ramy Awad, Mahdi Alnobani

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, Django 6, Django REST Framework
- **Frontend:** Vue 3, Pinia, Vite, Tailwind CSS
- **Database:** PostgreSQL (schema only — the forecasting engine is stateless at runtime)
- **Forecasting:** Custom statistical engine (Pandas, NumPy) — no trained model

---

## ⚙️ Prerequisites

Before setting up the project, ensure you have the following installed on your machine:

- **Python (3.10+)**
- **Node.js (20+)** — for the Vue frontend
- **PostgreSQL (v17+)** — ensure the local database server is running
- **Git**

---

## 🚀 Local Setup Instructions (Windows / PowerShell)

Follow these exact steps to get the project running on your local machine.

### 1. Clone the Repository

```powershell
git clone <paste-your-repo-url-here>
cd htu_registration
```

### 2. Set Up the Virtual Environment

We use a virtual environment to prevent package version conflicts across different machines.

```powershell
python -m venv venv
.\venv\Scripts\activate
```

> ✅ Ensure you see `(venv)` at the start of your command prompt before proceeding.

### 3. Install Backend Dependencies

Install all required Python packages (Django, PostgreSQL drivers, Pandas, NumPy):

```powershell
pip install -r requirements.txt
```

### 4. Database Setup

You must create an empty PostgreSQL database for Django to connect to. Open your terminal and run:

```powershell
psql -U postgres
```

Then, inside the `psql` shell, run:

```sql
CREATE DATABASE htu_registration;
\q
```

> 💡 Enter your local postgres password when prompted (it will be invisible as you type).

### 5. Configure Security Variables (`.env`)

We use a `.env` file to keep database passwords secure and out of the source code.

Create a file named exactly `.env` in the root folder (next to `manage.py`) and add the following:

```env
DB_PASSWORD=your_local_password_here
```

### 6. Build the Database Tables

Run the Django migrations to translate the Python models into PostgreSQL tables:

```powershell
python manage.py migrate
```

> 💡 The forecasting pipeline does not read from or write to the database at runtime — every prediction is computed fresh from the uploaded CSVs. Migrations are still needed because the `AppUser` schema is defined for future authentication (see *Future Work* in the project report).

### 7. Create a Superuser (Admin)

Create an admin account to access the Django admin UI:

```powershell
python manage.py createsuperuser
```

> Follow the prompts to set a username, email, and password.

### 8. Generate Test Data

Generate a fresh realistic dataset (one students CSV + three term-truncated enrolment CSVs) into `test_data/`:

```powershell
python generate_data.py --students 3000 --out-dir .\test_data --seed 42
```

The simulator targets a realistic ~14 credit hours per student per Fall/Spring semester and filters out any (course, semester) section with fewer than 16 enrolled students.

### 9. Run the Backend Server

Launch the Django API:

```powershell
python manage.py runserver
```

This serves on `http://127.0.0.1:8000`. Navigate to [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin) to view the database via the Django admin.

### 10. Run the Frontend (in a separate terminal)

```powershell
cd ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser. From there:

1. Click **Data Management** and upload one of the three `htu_enrollments_predict_*.csv` files together with `htu_students.csv`
2. Click **Run Predictions**
3. The **Dashboard** lights up with KPIs, courses needing attention, the largest growth list, and top courses by demand
4. Use **Predictions** to drill into per-course detail and the specific students likely to enrol in each

---

## 📁 Project Layout

```
htu_registration/
├── core/                    Django app (API views, models, URLs)
├── htu_registration/        Django project (settings, root urls, wsgi)
├── ml/
│   ├── __init__.py          Package init
│   └── inference.py         The three-signal forecasting engine
├── ui/                      Vue 3 frontend (Pinia stores, views, components)
├── test_data/               Sample CSVs the system runs against
├── generate_data.py         Data simulator (run this to regenerate test_data/)
├── manage.py                Django entry point
├── requirements.txt
├── README.md                This file
└── SYSTEM_DOCUMENTATION.md  Architecture and internals
```

For the forecasting algorithm, API contracts, CSV schemas, and module-level details, see `SYSTEM_DOCUMENTATION.md`.

---

## 🔒 Version Control Note

This repository includes a strict `.gitignore` file. **Never commit** your local `.env` file, `venv/` folder, `__pycache__/`, or `node_modules/` to the repository.