# Gear: Intelligent Registration System

**Gear** is an intelligent, machine-learning-driven course registration and resource allocation system developed for Al Hussein Technical University (HTU).

Designed to eliminate manual, guess-based scheduling, Gear leverages Random Forest models to accurately predict course demand before registration opens. By integrating Explainable AI (SHAP/LIME), the system provides department heads with transparent, data-backed insights to optimize section distribution, allocate instructors efficiently, and proactively flag graduating students who need substitute courses—transforming registration from reactive crisis management to proactive planning.

**Team:** Saeed Iqtaish, Ramy Awad, Mahdi Alnobani

---

## 🛠️ Tech Stack

- **Backend:** Python, Django
- **Database:** PostgreSQL
- **Machine Learning:** Scikit-learn (Random Forest)
- **Explainable AI:** SHAP/LIME

---

## ⚙️ Prerequisites

Before setting up the project, ensure you have the following installed on your machine:

- **Python (3.10+)**
- **PostgreSQL (v17+)** — Ensure the local database server is running.
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

### 3. Install Dependencies

Install all required Python packages (Django, PostgreSQL drivers, Data Science libraries):

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

### 7. Create a Superuser (Admin)

Create an admin account to access the secure Django backend dashboard:

```powershell
python manage.py createsuperuser
```

> Follow the prompts to set a username, email, and password.

### 8. Run the Development Server

Launch the application locally:

```powershell
python manage.py runserver
```

Navigate to [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin) in your browser and log in with your superuser credentials to view the database.

---

## 🔒 Version Control Note

This repository includes a strict `.gitignore` file. **Never commit** your local `.env` file, `venv/` folder, or `__pycache__` to the repository.