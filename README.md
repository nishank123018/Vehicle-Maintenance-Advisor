# Vehicle Maintenance Advisor

An AI-powered automotive diagnostic advisor and maintenance scheduling assistant built with Flask, Google Gemini, LangChain, LangGraph, and domain-grounded knowledge retrieval.

---

## Features

- **Automotive Subsystem Diagnosis**: Analyzes user-reported symptoms, vehicle telemetry, and warning lights to identify affected subsystems (Battery, Brakes, Tires, Engine, Oil, Coolant, Filters, Electrical Indicators).
- **Safety Risk Assessment**: Immediate detection of critical driving hazards (such as brake failures, fluid leaks, extreme overheating, loss of steering) with emergency warnings.
- **LangGraph Workflow**: StateGraph architecture orchestrating:
  $$\text{START} \longrightarrow \text{Receive Vehicle Data} \longrightarrow \text{Analyze Symptoms} \longrightarrow \text{Evaluate Follow-up} \longrightarrow \text{Retrieve Knowledge} \longrightarrow \text{Generate Causes} \longrightarrow \text{Recommendations} \longrightarrow \text{END}$$
- **Maintenance Milestone Schedules**: Estimates vehicle service needs based on odometer reading, vehicle age, and powertrain type (Gasoline, Diesel, Hybrid, EV).
- **Interactive Conversational Troubleshooting**: In-browser chat with diagnostic follow-up prompts, powered by Google Gemini with graceful offline fallback.
- **Single-File Compact Architecture**: Self-contained web application and responsive dashboard within `app.py` under strict line limits without comments or docstrings.

---

## Requirements & Installation

1. Navigate to the agent directory:
   ```bash
   cd "vehicle maintenance advisor"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure your Google Gemini API key:
   ```bash
   # Windows PowerShell
   $env:GOOGLE_API_KEY="your-gemini-api-key"

   # Windows CMD
   set GOOGLE_API_KEY=your-gemini-api-key

   # Linux / macOS
   export GOOGLE_API_KEY="your-gemini-api-key"
   ```
   *Note: If no API key is provided, the advisor operates using comprehensive deterministic automotive knowledge and maintenance heuristics.*

---

## Running the Application

Launch the Flask server:
```bash
python app.py
```

Then open your browser at:
```
http://localhost:5000
```

---

## API Endpoints

### 1. Diagnostic Analysis (`POST /api/analyze`)
Performs automotive subsystem identification, severity check, milestone schedule lookup, and diagnostic cause generation.

**Sample Request:**
```json
{
  "vehicle_type": "Sedan",
  "make_model": "Honda Civic",
  "age": 4,
  "mileage": 45000,
  "fuel_type": "Gasoline",
  "warning_light": "None",
  "recent_maintenance": "Oil change 3 months ago",
  "symptoms": "High-pitched squealing noise when braking at low speeds"
}
```

**Sample Response:**
```json
{
  "severity": "Moderate / Advisory",
  "subsystems": "Brakes",
  "safety_warning": "Standard Advisory: Drive prudently and monitor symptom development. Schedule regular diagnostic inspection.",
  "possible_causes": [
    "Possible cause: Mechanical wear, contamination, or operational fatigue within Brakes.",
    "Likely issue: Sensor miscalibration, vacuum variance, or electrical signal deviation associated with Brakes monitoring sensor circuit.",
    "Consider checking: Fluid reservoir levels, structural fasteners, belt tension, and related wiring harnesses."
  ],
  "recommendations": [
    "Visual inspection: Examine relevant fluid reservoirs, hydraulic lines, serpentine belt, and terminal connectors.",
    "Maintenance milestone recommendation: 30,000-mile interval: Replace cabin and engine air filters, inspect brake pads, and test battery charge retention.",
    "Professional check: Arrange an OBD-II diagnostic trouble code scan and chassis evaluation with a certified technician."
  ],
  "schedule_info": "30,000-mile interval: Replace cabin and engine air filters, inspect brake pads, and test battery charge retention.",
  "followup_questions": [
    "Does braking pulsation or noise occur under light pressure or hard emergency stops?",
    "Do you feel lateral pulling in the steering wheel when pressing the brake pedal?"
  ]
}
```

### 2. Conversational Follow-up (`POST /api/chat`)
Answers vehicle maintenance questions and clarifies diagnostic context.

---

## Technical Constraints Compliance

- **File Count**: Self-contained within `app.py`, `requirements.txt`, and `README.md`.
- **app.py Size**: Strictly under 400 lines (393 lines).
- **Comments & Docstrings**: 0 Python `#` comments and 0 docstrings in `app.py`.
- **Zero External Vector DB / Auth / React**: Fully contained in-memory architecture.
