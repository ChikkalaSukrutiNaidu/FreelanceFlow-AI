# 💼 FreelanceFlow AI

> **Hacktoberfest 2026 — Week 3: Side Hustle Challenge**  
> A simple, professional web application for freelancers and small-business owners to organize invoices and expenses using open-weight AI.

---

## 🌟 Overview

Freelancers, solopreneurs, and side-hustlers often struggle with messy administrative tasks: unorganized client notes, untracked receipts, awkward late-payment follow-ups, and unclear monthly cash flow.

**FreelanceFlow AI** solves these pain points through a lightweight, beginner-friendly Streamlit web application powered by the open-weight model **`openai/gpt-oss-20b`** via the ultra-fast **Groq API**.

---

## 🚀 Key Features

The application is structured into four core modules:

1. **📄 Invoice Organizer**
   - Ingests raw, unstructured invoice notes, email threads, or billing memos.
   - Extracts key parameters: Client info, Invoice #, Line items (hours/rates), Subtotal, Discounts, and Due Dates into clean, professional markdown tables.

2. **🏷️ Expense Categorizer**
   - Classifies business expenses and receipts into standardized accounting categories (Software & Subscriptions, Office Supplies, Meals, Travel, Infrastructure).
   - Identifies tax deductibility status to simplify freelance tax preparation.

3. **⏰ Payment Reminder Generator**
   - Solves the #1 freelance challenge: getting paid on time.
   - Generates customized, diplomatic reminder emails with adjustable tone (Friendly, Polite & Direct, Firm & Urgent).

4. **📊 Monthly Summary & AI Financial Insights**
   - Visualizes monthly revenue, pending invoices, operating expenses, and net profit with interactive KPI cards and Streamlit bar charts.
   - AI Financial Advisor provides risk assessment, cash-flow stability analysis, and tax reserve recommendations.

---

## 🧠 AI Architecture

- **Model**: `openai/gpt-oss-20b` (Open-weight LLM)
- **Inference Engine**: [Groq Cloud API](https://groq.com) (Low-latency LPU inference)
- **SDK**: Official `groq` Python SDK
- **Security**: API keys are securely managed via environment variables (`python-dotenv`) or runtime sidebar input. Keys are **never** hardcoded or committed to version control.

---

## 📂 Project Structure

```text
FreelanceFlow-AI/
├── .env.example         # Template for environment variables (GROQ_API_KEY)
├── .gitignore           # Git ignore rules protecting .env, .venv, and caches
├── requirements.txt     # Python package dependencies
├── app.py               # Main Streamlit web application & Groq AI logic
└── README.md            # Project documentation and demonstration guide
```

### File Breakdown:
- **`app.py`**: The core application entry point. Implements the Streamlit UI, session state management, pandas data visualizations, and Groq API calls using `openai/gpt-oss-20b`.
- **`requirements.txt`**: Lists all required Python packages with version constraints (`streamlit`, `groq`, `python-dotenv`, `pandas`).
- **`.env.example`**: Shows the expected format for local environment variables so new contributors can quickly configure their setup.
- **`.gitignore`**: Prevents sensitive credentials (`.env`), virtual environments (`.venv`), and Python bytecode from being tracked by Git.
- **`README.md`**: Complete documentation, installation instructions, and college demonstration guide.

---

## 🛠️ Installation & Setup Guide

### 1. Prerequisites
- Python 3.10+ (Python 3.10, 3.11, 3.12, 3.13, or 3.14)
- A free Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### 2. Clone or Open the Repository
Navigate to the project directory:
```bash
cd FreelanceFlow-AI
```

### 3. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Your API Key

Create a `.env` file from the example:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On macOS / Linux:**
```bash
cp .env.example .env
```

Edit the `.env` file and insert your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

*(Note: You can also enter or override your API key directly in the web app sidebar at runtime.)*

---

## 💻 Running the Application

Launch the Streamlit dashboard with:

```bash
streamlit run app.py
```

Once running, your default browser will open automatically at:
```text
http://localhost:8501
```

---

## 🎓 College Project Demonstration Guide

When explaining this project in an academic or college viva demonstration:

1. **Problem Statement**:
   - Freelancers lose 10–15 hours every month on manual bookkeeping, tracking receipts, and drafting client emails.
   - Existing accounting software (e.g., QuickBooks) is expensive and bloated for solo freelancers.

2. **Why Open-Weight AI (`openai/gpt-oss-20b`)?**:
   - Open-weight models democratize AI by avoiding proprietary vendor lock-in.
   - Hosted on Groq, it provides near-instant response times (<1 second inference), ideal for interactive web applications.

3. **Code Architecture (Zero Bloat)**:
   - **Streamlit**: Eliminates the need for complex frontend frameworks (React/Vue) by enabling pure Python stateful UI development.
   - **`dotenv` Integration**: Demonstrates production-grade security practices by isolating secrets from source code.
   - **Data Visualization**: Leverages `pandas` and native Streamlit charts for cash-flow metrics.
   - **Robust Error Handling**: Real API errors are caught and communicated transparently to the user without simulated/mocked fallbacks.

4. **Interactive Walkthrough**:
   - Click **"Load Sample Invoice"** in Tab 1 to demonstrate instant invoice parsing.
   - Click **"Load Sample Expenses"** in Tab 2 to demonstrate automatic expense categorization.
   - Generate a custom payment reminder in Tab 3 with different tone settings.
   - Inspect the financial metrics and click **"Generate Financial Insights"** in Tab 4.

---

## 📜 License & Compliance

Built for **Hacktoberfest 2026 Week 3 Side Hustle Challenge**.  
Licensed under the [MIT License](https://opensource.org/licenses/MIT).
