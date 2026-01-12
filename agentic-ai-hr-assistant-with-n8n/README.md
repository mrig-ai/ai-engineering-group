# HR Agentic AI with n8n

This project provides an **Agentic AI workflow for Human Resources** built on **n8n**, focused specifically on **evaluating whether a candidate CV matches a given job profile**.

The workflow compares a job profile against a candidate CV, analyzes the level of match using an LLM, and optionally sends the evaluation results to the candidate via email after explicit confirmation.

The core workflow is distributed as an n8n JSON template that can be imported and customized according to your environment and preferred AI provider.

---

## 📦 Project Contents

* **`HR_template.json`**
  The main n8n workflow template. This file contains the full schema required to create the CV-to-profile matching workflow inside n8n.

* **`data/`**
  Contains example job profiles and example CVs used for testing and validation.

* **`README.md`**
  This documentation file.

---

## 🧠 Architecture Overview

The HR Agentic AI workflow is designed for **CV-to-profile matching with conditional agent routing** and includes:

* A **job profile** provided as an attachment
* A **candidate CV** provided as an attachment
* A **CV Matching Agent** that evaluates alignment between the CV and the profile
* A **decision gate** to determine whether the candidate is a match

If the candidate **is a match**, the workflow continues with:

* An **Interview Agent** that evaluates interview readiness and key discussion points
* A **Salary Agent** that estimates an appropriate salary range based on the profile and CV
* A **Summary step** that consolidates all findings
* An **Email node** that sends a summarized report **as an attachment** to the HR team

If the candidate **is not a match**, the workflow can optionally stop or send a rejection/feedback email (configurable).

n8n acts as the orchestration layer, ensuring traceable, modular, and extensible agent behavior.

---

## ⚙️ Prerequisites

Before getting started, ensure you have:

* **Node.js** (required for local n8n installation)
* An **LLM provider account** (e.g., OpenAI, Google Gemini, or another supported model)
* An **Email service account** supported by n8n (SMTP, Gmail, Outlook, etc.)

---

## 🚀 Quick Start (Local Installation)

You can quickly run n8n locally using `npx`:

```bash
npx n8n
```

Once started:

* n8n will be available at: **[http://localhost:5678](http://localhost:5678)**
* Follow the setup wizard if this is your first run

---

## 📥 Importing the HR Workflow

1. Open the n8n Editor UI
2. Click **“Import”** or **“Import from File”**
3. Select the provided **`HR_template.json`** file
4. Confirm the import

The HR Agentic AI workflow will now appear in your workspace.

---

## 🔐 Required Credentials Setup

### 1. LLM / AI Model Credentials

You must create credentials for the AI model used in the workflow.

Supported examples:

* **OpenAI**
* **Google Gemini**
* **Other n8n-supported LLM providers**

Steps:

1. Go to **Credentials** in n8n
2. Click **Add Credential**
3. Select the appropriate AI provider
4. Enter your API key and configuration
5. Save and assign the credential to the AI nodes in the workflow

---

### 2. Email Tool Credentials

Email credentials are required for HR communications such as notifications and responses.

Steps:

1. Go to **Credentials** in n8n
2. Add an Email-related credential (e.g., SMTP, Gmail, Outlook)
3. Configure authentication details
4. Save and link the credential to the Email nodes in the workflow

---

## 🛠 Customization

You can customize the workflow to:

* Adjust **CV matching thresholds** and decision logic
* Modify prompts for the **Matching, Interview, and Salary agents**
* Change how interview readiness and salary ranges are calculated
* Customize the **summary report format** (PDF, text, or structured document)
* Configure which **HR recipients** receive the final attachment
* Add approval or human-in-the-loop review steps

All logic can be modified directly in the n8n visual editor.

---

## 📚 Learn More About n8n

For detailed documentation, advanced configuration, and deployment options, refer to the official n8n repository:

👉 [https://github.com/n8n-io/n8n](https://github.com/n8n-io/n8n)

---

## ⚠️ Notes

* This project evaluates **CV-to-profile alignment only** and does not make hiring decisions.
* Interview and salary evaluations are **AI-generated recommendations**, not guarantees.
* The AI must request confirmation before sending any email to a candidate.
* Salary estimates should be reviewed for legal and market compliance.
* Ensure compliance with data protection regulations (e.g. GDPR) when handling CV data.

