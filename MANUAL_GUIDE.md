# 🚀 HOW TO RUN PRESENTAGENT

### ✅ LATEST FIXES
1.  **Missing "Default Template"**: Fixed (Auto-repair).
2.  **Missing "Soffice"**: Fixed (Auto-detected).
3.  **Missing "Poppler" (PDF Error)**: Fixed (Replaced with Python library).

---

### STEP 1: Start the BRAIN (Backend)
1.  Open **Terminal 1**.
2.  Stop any running process (`Ctrl+C`).
3.  Run:
    ```powershell
    python presentagent/backend.py
    ```
4.  Wait for **`Application startup complete`**.

---

### STEP 2: Start the FACE (Frontend)
1.  Open **Terminal 2**.
2.  If running, stop it (`Ctrl+C`).
3.  Run:
    ```powershell
    cd presentagent
    npm run serve
    ```
4.  Wait for `http://localhost:8088`.

---

### STEP 3: USE IT
1.  Open **http://localhost:8088**.
2.  Click **Generate Slides**.
