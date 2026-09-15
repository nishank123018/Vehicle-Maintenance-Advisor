import os, re, json
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

KNOWLEDGE_BASE = [
    {"topic": "Battery", "keywords": ["battery", "starting", "crank", "cranking", "clicking", "dead", "alternator", "voltage", "charge", "jump"], "info": "Slow engine cranking or rapid clicking typically indicates diminished battery charge, terminal oxidation, or charging system deficiency. Automotive lead-acid batteries generally maintain reliability for 3 to 5 years. Testing resting open-circuit voltage can identify battery health (12.6V represents standard nominal charge)."},
    {"topic": "Brakes", "keywords": ["brake", "braking", "squeal", "squealing", "squeak", "squeaking", "grind", "grinding", "spongy", "pedal", "stopping", "abs", "pads", "rotor", "rotors"], "info": "High-pitched squealing suggests brake pad wear indicators are making contact with rotors. Metallic grinding indicates severe friction material depletion requiring rotor replacement. Spongy pedal feel or sinking travel points to hydraulic fluid contamination, air pockets, or master cylinder internal bypass."},
    {"topic": "Tires", "keywords": ["tire", "tires", "tread", "vibration", "vibrating", "vibrate", "wobble", "shaking", "alignment", "pressure", "pulling", "tpms", "flat"], "info": "Highway velocity vibration frequently stems from wheel assembly imbalance or radial force variation. Lateral drift or localized shoulder wear indicates wheel alignment deviation. Standard rotation intervals are recommended every 5,000 to 7,500 miles to promote uniform tread life."},
    {"topic": "Engine", "keywords": ["engine", "misfire", "stumble", "rough", "idle", "idling", "stall", "stalling", "acceleration", "hesitation", "spark", "cylinder", "power loss"], "info": "Erratic idle rhythms and combustion hesitation commonly correlate with fouled spark plugs, failing ignition coils, mass airflow contamination, or unmetered vacuum intake leaks. A blinking diagnostic indicator signals active catalytic converter threat from unburned fuel."},
    {"topic": "Oil", "keywords": ["oil", "pressure", "level", "leak", "leaking", "burning", "viscosity", "valvetrain", "ticking", "lubricant"], "info": "Insufficient oil volume or degraded viscosity induces hydraulic lifter ticking and friction escalation. Typical full synthetic intervals span 5,000 to 10,000 miles depending on duty cycle. Immediate engine shutdown is critical if low oil pressure warning illuminates."},
    {"topic": "Coolant", "keywords": ["coolant", "overheat", "overheating", "radiator", "temperature", "steam", "thermostat", "antifreeze", "boiling", "water pump"], "info": "Rapid temperature elevation or vapor emission signals cooling circuit compromise such as thermostat seizure, radiator leakage, or head gasket breach. Continued thermal overload risks permanent cylinder head distortion."},
    {"topic": "Filters", "keywords": ["filter", "filters", "air", "cabin", "fuel", "intake", "odor", "airflow", "efficiency"], "info": "Restricted engine intake filters reduce throttle responsiveness and fuel economy. Blocked cabin filtration diminishes climate control velocity and introduces interior odors. Routine replacement occurs every 15,000 to 30,000 miles."},
    {"topic": "Lights", "keywords": ["light", "lights", "warning", "check engine", "abs", "cel", "indicator", "mil"], "info": "Steady diagnostic lamps indicate stored powertrain diagnostic trouble codes retrievable via OBD-II protocol. Active illumination warrants proactive scanning prior to component degradation."}
]

@tool(description="Analyzes symptoms and warning lights to identify affected vehicle subsystems")
def symptom_analyzer(symptoms: str, warning_light: str) -> str:
    combined = f"{symptoms} {warning_light}".lower()
    matched = [item["topic"] for item in KNOWLEDGE_BASE if any(k in combined for k in item["keywords"])]
    return ", ".join(matched) if matched else "General Powertrain and Mechanical"

@tool(description="Retrieves relevant maintenance knowledge based on query symptoms")
def maintenance_retriever(query: str) -> str:
    tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for item in KNOWLEDGE_BASE:
        overlap = sum(1 for k in item["keywords"] if k in tokens or any(t.startswith(k[:4]) for t in tokens if len(k) >= 4 and len(t) >= 4))
        if overlap > 0:
            scored.append((overlap, item["info"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s[1] for s in scored[:3]]
    return " ".join(selected) if selected else "Standard comprehensive multi-point vehicle inspection recommended."

@tool(description="Estimates upcoming vehicle maintenance schedule based on mileage and age")
def maintenance_schedule(mileage: int, age: int, fuel_type: str) -> str:
    milestones = []
    if mileage >= 100000 or age >= 10:
        milestones.append("100,000-mile milestone: Inspect timing belt/chain, water pump, exchange spark plugs, and flush coolant.")
    elif mileage >= 60000 or age >= 6:
        milestones.append("60,000-mile interval: Replace transmission lubricant, spark plugs, brake hydraulic fluid, and inspect bushings.")
    elif mileage >= 30000 or age >= 3:
        milestones.append("30,000-mile interval: Replace cabin and engine air filters, inspect brake pads, and test battery charge retention.")
    else:
        milestones.append("Recurring interval: Perform engine oil and filter service, rotate tires, and verify fluid reservoirs.")
    ft = fuel_type.lower()
    if ft == "electric":
        milestones.append("EV specific: Inspect traction battery coolant, regenerative friction pads, and high-voltage cabling.")
    elif ft == "diesel":
        milestones.append("Diesel specific: Service diesel particulate filter, empty fuel-water separator, and examine glow plugs.")
    return " ".join(milestones)

@tool(description="Checks safety risk level and detects critical driving hazards")
def severity_checker(symptoms: str, warning_light: str) -> Dict[str, str]:
    text = f"{symptoms} {warning_light}".lower()
    hazard_terms = [
        "cannot stop", "no brakes", "brake failure", "pedal to floor", "pedal to the floor",
        "fuel leak", "gas leak", "petrol leak", "severe overheating", "overheating",
        "steam", "smoke", "fire", "steering failure", "loss of steering", "flashing check engine",
        "blinking check engine", "oil pressure warning"
    ]
    if any(h in text for h in hazard_terms):
        return {
            "level": "Immediate Danger",
            "warning": "CRITICAL SAFETY WARNING: Severe condition detected. Please safely bring your vehicle to a stop as soon as possible, switch off the ignition, and arrange professional towing or roadside assistance. Do not attempt hazardous driving."
        }
    high_terms = ["grinding", "spongy brake", "spongy", "shaking", "misfire", "stalling", "low oil", "battery light", "oil light", "abs light", "overheat", "knocking"]
    if any(h in text for h in high_terms):
        return {"level": "High Priority", "warning": "Safety Advisory: Timely mechanical inspection is advised before taking extended highway journeys."}
    return {"level": "Moderate / Advisory", "warning": "Standard Advisory: Drive prudently and monitor symptom development. Schedule regular diagnostic inspection."}

def get_gemini_client():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key: return None
    models = [os.environ.get("GEMINI_MODEL", ""), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for m in models:
        if not m: continue
        try: return ChatGoogleGenerativeAI(model=m, google_api_key=key, temperature=0.2)
        except Exception: continue
    return None

class AgentState(TypedDict):
    vehicle_data: Dict[str, Any]
    symptoms: str
    subsystems: str
    need_followup: bool
    followup_questions: List[str]
    retrieved_knowledge: str
    schedule_info: str
    possible_causes: List[str]
    recommendations: List[str]
    severity: str
    safety_warning: str
    chat_history: List[Dict[str, str]]

def receive_vehicle_data(state: AgentState) -> Dict[str, Any]:
    v = state.get("vehicle_data", {})
    risk = severity_checker.invoke({"symptoms": state.get("symptoms", ""), "warning_light": v.get("warning_light", "")})
    return {"severity": risk["level"], "safety_warning": risk["warning"]}

def analyze_symptoms(state: AgentState) -> Dict[str, Any]:
    v = state.get("vehicle_data", {})
    subsystems = symptom_analyzer.invoke({"symptoms": state.get("symptoms", ""), "warning_light": v.get("warning_light", "")})
    return {"subsystems": subsystems}

def evaluate_followup(state: AgentState) -> Dict[str, Any]:
    sub = state.get("subsystems", "").lower()
    q_map = {
        "brake": ["Does braking pulsation or noise occur under light pressure or hard emergency stops?", "Do you feel lateral pulling in the steering wheel when pressing the brake pedal?"],
        "battery": ["Does the starter produce rapid rhythmic clicking or a slow drawn-out crank?", "Have dashboard accessories or interior lights appeared visibly dimmer prior to ignition?"],
        "tires": ["Does the vibration intensify at highway speeds or remain present at low speeds?", "Have you observed localized tire tread feathering or abnormal tire pressure drop?"],
        "engine": ["Does combustion hesitation occur when cold or after reaching operating temperature?", "Have you noticed rough idling, power dips under load, or abnormal exhaust odor?"],
        "oil": ["Have you noticed dark oil puddles under the vehicle or burning odor in the bay?", "When was the oil dipstick level and condition last physically inspected?"],
        "coolant": ["Does temperature rise rapidly in idling traffic or during uphill highway driving?", "Have you observed white vapor from exhaust or sweet-smelling fluid near the radiator?"],
        "filters": ["Has there been a noticeable drop in acceleration or climate control blower power?", "Does a musty odor or dusty atmosphere enter the cabin when the blower fan starts?"],
        "lights": ["Is the diagnostic indicator steadily illuminated or does it flash during acceleration?", "Did the warning light illuminate immediately upon starting or mid-journey?"]
    }
    matched = next((qs for k, qs in q_map.items() if k in sub), [
        "Does the symptom vary with vehicle road speed, engine RPM, or road surface quality?",
        "Did this symptom begin suddenly after a specific event or develop gradually over weeks?"
    ])
    return {"need_followup": True, "followup_questions": matched}

def retrieve_knowledge(state: AgentState) -> Dict[str, Any]:
    symptoms = state.get("symptoms", "")
    subsystems = state.get("subsystems", "")
    v = state.get("vehicle_data", {})
    rag_docs = maintenance_retriever.invoke({"query": f"{symptoms} {subsystems} {v.get('warning_light', '')}"})
    sched = maintenance_schedule.invoke({"mileage": int(v.get("mileage") or 0), "age": int(v.get("age") or 0), "fuel_type": str(v.get("fuel_type") or "gasoline")})
    return {"retrieved_knowledge": rag_docs, "schedule_info": sched}

def generate_causes(state: AgentState) -> Dict[str, Any]:
    v = state.get("vehicle_data", {})
    symptoms = state.get("symptoms", "")
    subsystems = state.get("subsystems", "")
    rag_docs = state.get("retrieved_knowledge", "")
    llm = get_gemini_client()
    if llm:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an automotive diagnostic advisor. Provide 3 possible causes for the reported symptoms. Use cautious advisory phrasing like 'Possible cause', 'Likely issue', or 'Consider checking'. Never state a confirmed diagnosis. Keep each point clear, educational, and concise. Format as a bulleted list."),
            ("human", "Vehicle: {v_type} {make_model}, Age: {age} yrs, Mileage: {mileage} miles, Fuel: {fuel}, Warning Light: {warning}, Service: {service}.\nSymptoms: {symptoms}\nSubsystems: {subsystems}\nContext: {rag}")
        ])
        chain = prompt | llm
        try:
            resp = chain.invoke({
                "v_type": v.get("vehicle_type", "Automobile"), "make_model": v.get("make_model", "Vehicle"),
                "age": v.get("age", "N/A"), "mileage": v.get("mileage", "N/A"), "fuel": v.get("fuel_type", "Gasoline"),
                "warning": v.get("warning_light", "None"), "service": v.get("recent_maintenance", "None"),
                "symptoms": symptoms, "subsystems": subsystems, "rag": rag_docs
            })
            causes = [line.strip("- *• ") for line in resp.content.split("\n") if line.strip("- *• ")]
            if causes: return {"possible_causes": causes[:4]}
        except Exception: pass
    wl = v.get("warning_light")
    wl_desc = f"{wl} warning activation" if wl and wl.lower() != "none" else f"{subsystems} monitoring sensor circuit"
    fallback = [
        f"Possible cause: Mechanical wear, contamination, or operational fatigue within {subsystems}.",
        f"Likely issue: Sensor miscalibration, vacuum variance, or electrical signal deviation associated with {wl_desc}.",
        f"Consider checking: Fluid reservoir levels, structural fasteners, belt tension, and related wiring harnesses."
    ]
    return {"possible_causes": fallback}

def generate_recommendations(state: AgentState) -> Dict[str, Any]:
    v = state.get("vehicle_data", {})
    causes = state.get("possible_causes", [])
    sched = state.get("schedule_info", "")
    rag = state.get("retrieved_knowledge", "")
    llm = get_gemini_client()
    if llm:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an automotive safety and maintenance advisor. Provide 3 practical basic maintenance suggestions and recommended checks. Never suggest hazardous DIY procedures. Enforce that drivers should consult certified mechanics. Format as bullet points."),
            ("human", "Vehicle: {make_model}, Mileage: {mileage}.\nCauses: {causes}\nSchedule notes: {sched}\nRAG notes: {rag}")
        ])
        chain = prompt | llm
        try:
            resp = chain.invoke({"make_model": v.get("make_model", "Vehicle"), "mileage": v.get("mileage", "N/A"), "causes": " | ".join(causes), "sched": sched, "rag": rag})
            recs = [line.strip("- *• ") for line in resp.content.split("\n") if line.strip("- *• ")]
            if recs: return {"recommendations": recs[:4]}
        except Exception: pass
    fallback = [
        "Visual inspection: Examine relevant fluid reservoirs, hydraulic lines, serpentine belt, and terminal connectors.",
        f"Maintenance milestone recommendation: {sched}",
        "Professional check: Arrange an OBD-II diagnostic trouble code scan and chassis evaluation with a certified technician."
    ]
    return {"recommendations": fallback}

workflow = StateGraph(AgentState)
for name, func in [("receive_vehicle_data", receive_vehicle_data), ("analyze_symptoms", analyze_symptoms), ("need_followup", evaluate_followup), ("retrieve_knowledge", retrieve_knowledge), ("generate_causes", generate_causes), ("recommendations", generate_recommendations)]:
    workflow.add_node(name, func)
workflow.add_edge(START, "receive_vehicle_data")
workflow.add_edge("receive_vehicle_data", "analyze_symptoms")
workflow.add_edge("analyze_symptoms", "need_followup")
workflow.add_edge("need_followup", "retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", "generate_causes")
workflow.add_edge("generate_causes", "recommendations")
workflow.add_edge("recommendations", END)
advisor_graph = workflow.compile()

app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vehicle Maintenance Advisor</title>
<style>
:root{--bg:rgb(15,23,42);--surface:rgb(30,41,59);--border:rgb(51,65,85);--text:rgb(241,245,249);--muted:rgb(148,163,184);--accent:rgb(59,130,246);--danger-bg:rgb(69,10,10);--danger-border:rgb(239,68,68);--danger-text:rgb(254,202,202);--warn-bg:rgb(69,39,10);--warn-border:rgb(245,158,11);--card:rgb(24,32,47)}
*{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
body{background:var(--bg);color:var(--text);padding:24px 16px;line-height:1.5}
.container{max-width:1100px;margin:0 auto}
header{margin-bottom:24px;border-bottom:1px solid var(--border);padding-bottom:16px}
h1{font-size:1.75rem;font-weight:700;color:rgb(255,255,255);display:flex;align-items:center;gap:8px}
.subtitle{color:var(--muted);font-size:0.95rem;margin-top:4px}
.grid{display:grid;grid-template-columns:1fr 1.2fr;gap:24px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.form-group{display:flex;flex-direction:column;margin-bottom:12px}
label{font-size:0.82rem;font-weight:600;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:0.04em}
input,select,textarea{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px;color:var(--text);font-size:0.95rem}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--accent)}
textarea{resize:vertical;min-height:75px}
.btn{background:var(--accent);color:rgb(255,255,255);border:none;border-radius:8px;padding:12px 20px;font-weight:600;font-size:0.95rem;cursor:pointer;width:100%;transition:opacity 0.2s}
.btn:hover{opacity:0.9}.btn:disabled{opacity:0.5;cursor:not-allowed}
.alert-box{padding:14px;border-radius:8px;margin-bottom:16px;display:none;font-weight:500;font-size:0.9rem}
.alert-danger{background:var(--danger-bg);border:1px solid var(--danger-border);color:var(--danger-text)}
.alert-warning{background:var(--warn-bg);border:1px solid var(--warn-border);color:rgb(254,243,199)}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:14px}
.card h3{font-size:0.95rem;font-weight:700;color:rgb(147,197,253);margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
ul{padding-left:20px;font-size:0.9rem;color:rgb(226,232,240)}li{margin-bottom:6px}
.badge{padding:4px 10px;border-radius:999px;font-size:0.75rem;font-weight:700;letter-spacing:0.04em}
.badge-danger{background:rgb(220,38,38);color:white}.badge-warning{background:rgb(217,119,6);color:white}.badge-info{background:rgb(37,99,235);color:white}
.chat-window{height:200px;overflow-y:auto;background:rgb(15,23,42);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:10px;display:flex;flex-direction:column;gap:8px}
.chat-bubble{padding:8px 12px;border-radius:8px;font-size:0.88rem;max-width:85%}
.chat-agent{background:rgb(30,41,59);border:1px solid var(--border);align-self:flex-start}
.chat-user{background:var(--accent);color:white;align-self:flex-end}
.chat-input-row{display:flex;gap:8px}
.chat-btn{background:var(--accent);color:white;border:none;border-radius:8px;padding:0 16px;cursor:pointer;font-weight:600}
.question-pill{display:inline-block;background:rgb(30,41,59);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:0.8rem;margin:4px 4px 4px 0;cursor:pointer}
.question-pill:hover{border-color:var(--accent);color:rgb(147,197,253)}
.empty-state{text-align:center;padding:40px 20px;color:var(--muted);font-size:0.95rem}
</style>
</head>
<body>
<div class="container">
<header><h1>Vehicle Maintenance Advisor</h1><p class="subtitle">AI-guided diagnostics, safety assessment, and proactive maintenance scheduling</p></header>
<div class="grid">
<div class="panel">
<form id="vehicleForm" onsubmit="submitDiagnosis(event)">
<div class="form-row">
<div class="form-group"><label>Vehicle Type</label><select id="vehicle_type" required><option value="Sedan">Sedan</option><option value="SUV">SUV</option><option value="Truck">Truck</option><option value="Hatchback">Hatchback</option><option value="Coupe">Coupe</option><option value="Electric Vehicle">Electric Vehicle</option></select></div>
<div class="form-group"><label>Make / Model</label><input type="text" id="make_model" placeholder="e.g. Toyota Camry" required></div>
</div>
<div class="form-row">
<div class="form-group"><label>Vehicle Age (Years)</label><input type="number" id="age" min="0" max="60" value="4" required></div>
<div class="form-group"><label>Current Mileage</label><input type="number" id="mileage" min="0" max="999999" value="45000" required></div>
</div>
<div class="form-row">
<div class="form-group"><label>Fuel Type</label><select id="fuel_type"><option value="Gasoline">Gasoline</option><option value="Diesel">Diesel</option><option value="Hybrid">Hybrid</option><option value="Electric">Electric</option></select></div>
<div class="form-group"><label>Warning Light</label><select id="warning_light"><option value="None">None</option><option value="Check Engine">Check Engine</option><option value="Brake Light">Brake Light</option><option value="Battery Light">Battery Light</option><option value="Oil Pressure Light">Oil Pressure Light</option><option value="ABS Light">ABS Light</option><option value="Tire Pressure TPMS">Tire Pressure TPMS</option></select></div>
</div>
<div class="form-group"><label>Recent Maintenance</label><input type="text" id="recent_maintenance" placeholder="e.g. Oil change 3 months ago"></div>
<div class="form-group"><label>Observed Problem / Symptoms</label><textarea id="symptoms" placeholder="Describe sounds, vibrations, handling anomalies, or leaks..." required></textarea></div>
<button type="submit" id="analyzeBtn" class="btn">Analyze Vehicle Condition</button>
</form>
</div>
<div class="panel">
<div id="alertBox" class="alert-box"></div>
<div id="resultsContent" style="display:none">
<div class="card"><h3><span>Severity Level</span><span id="severityBadge" class="badge"></span></h3><p id="subsystemsText" style="font-size:0.88rem;color:var(--muted)"></p></div>
<div class="card"><h3>Possible Causes</h3><ul id="causesList"></ul></div>
<div class="card"><h3>Recommended Checks & Suggestions</h3><ul id="recsList"></ul></div>
<div class="card"><h3>Estimated Maintenance Schedule</h3><p id="scheduleText" style="font-size:0.88rem;color:rgb(226,232,240)"></p></div>
<div class="card"><h3>Follow-up Diagnostic Questions</h3><div id="followupContainer"></div></div>
<div class="card"><h3>Conversational Troubleshooting</h3><div id="chatWindow" class="chat-window"></div>
<div class="chat-input-row"><input type="text" id="chatInput" placeholder="Answer questions or ask follow-ups..." onkeydown="if(event.key==='Enter')sendChat()"><button type="button" class="chat-btn" onclick="sendChat()">Send</button></div>
</div>
</div>
<div id="emptyState" class="empty-state">Fill out vehicle information and symptoms on the left to initiate comprehensive diagnostics.</div>
</div>
</div>
</div>
<script>
let activeState = null;
async function submitDiagnosis(e){
e.preventDefault();
const btn = document.getElementById("analyzeBtn");
btn.disabled = true; btn.innerText = "Analyzing Symptoms...";
const payload = {
vehicle_type: document.getElementById("vehicle_type").value, make_model: document.getElementById("make_model").value,
age: parseInt(document.getElementById("age").value)||0, mileage: parseInt(document.getElementById("mileage").value)||0,
fuel_type: document.getElementById("fuel_type").value, warning_light: document.getElementById("warning_light").value,
recent_maintenance: document.getElementById("recent_maintenance").value, symptoms: document.getElementById("symptoms").value
};
try{
const res = await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
const data = await res.json();
activeState = data; renderResults(data);
}catch(err){ alert("Diagnostic request failed. Please verify connection."); }
finally{ btn.disabled = false; btn.innerText = "Analyze Vehicle Condition"; }
}
function renderResults(data){
document.getElementById("emptyState").style.display = "none";
document.getElementById("resultsContent").style.display = "block";
const alertBox = document.getElementById("alertBox");
if(data.safety_warning && data.severity === "Immediate Danger"){
alertBox.className = "alert-box alert-danger"; alertBox.innerText = data.safety_warning; alertBox.style.display = "block";
}else if(data.safety_warning && data.severity === "High Priority"){
alertBox.className = "alert-box alert-warning"; alertBox.innerText = data.safety_warning; alertBox.style.display = "block";
}else{ alertBox.style.display = "none"; }
const badge = document.getElementById("severityBadge");
badge.innerText = data.severity;
badge.className = "badge " + (data.severity === "Immediate Danger" ? "badge-danger" : data.severity === "High Priority" ? "badge-warning" : "badge-info");
document.getElementById("subsystemsText").innerText = data.subsystems;
document.getElementById("causesList").innerHTML = data.possible_causes.map(c => `<li>${c}</li>`).join("");
document.getElementById("recsList").innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join("");
document.getElementById("scheduleText").innerText = data.schedule_info;
document.getElementById("followupContainer").innerHTML = (data.followup_questions || []).map(q => `<div class="question-pill" onclick="selectQuestion('${q.replace(/'/g, "\\\\'")}')">${q}</div>`).join("");
document.getElementById("chatWindow").innerHTML = `<div class="chat-bubble chat-agent">Vehicle diagnostics initialized for ${data.vehicle_data.make_model}. Feel free to respond to the diagnostic questions above or ask for clarification.</div>`;
}
function selectQuestion(q){
const inp = document.getElementById("chatInput");
inp.value = "Regarding: " + q + " -> "; inp.focus();
}
async function sendChat(){
const inp = document.getElementById("chatInput");
const msg = inp.value.trim();
if(!msg || !activeState) return;
const chatWindow = document.getElementById("chatWindow");
chatWindow.innerHTML += `<div class="chat-bubble chat-user">${msg}</div>`;
inp.value = ""; chatWindow.scrollTop = chatWindow.scrollHeight;
try{
const res = await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg,state:activeState})});
const reply = await res.json();
chatWindow.innerHTML += `<div class="chat-bubble chat-agent">${reply.response}</div>`;
chatWindow.scrollTop = chatWindow.scrollHeight;
}catch(e){ chatWindow.innerHTML += `<div class="chat-bubble chat-agent">Error receiving response.</div>`; }
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/analyze", methods=["POST"])
def analyze_endpoint():
    data = request.get_json() or {}
    initial_state = {
        "vehicle_data": data, "symptoms": data.get("symptoms", ""), "subsystems": "",
        "need_followup": False, "followup_questions": [], "retrieved_knowledge": "",
        "schedule_info": "", "possible_causes": [], "recommendations": [],
        "severity": "", "safety_warning": "", "chat_history": []
    }
    return jsonify(advisor_graph.invoke(initial_state))

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    body = request.get_json() or {}
    user_msg = body.get("message", "")
    prev_state = body.get("state", {})
    llm = get_gemini_client()
    hazard_terms = ["cannot stop", "no brakes", "brake failure", "pedal to floor", "fuel leak", "gas leak", "steam", "smoke", "fire", "steering failure", "oil pressure"]
    if any(h in user_msg.lower() for h in hazard_terms):
        return jsonify({"response": "CRITICAL ADVISORY: The symptoms you described point to severe operating risk. Please stop driving safely, turn off the engine, and contact a certified mechanic or towing service immediately."})
    if llm:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an automotive diagnostic advisor assisting a vehicle owner. Use cautious advisory wording ('Possible cause', 'Consider checking'). Never make confirmed mechanical guarantees. Answer the user's question concisely, referencing their vehicle and previous diagnostic context."),
            ("human", "Vehicle: {v_data}\nPrevious causes: {causes}\nRecommendations: {recs}\nUser follow-up: {msg}")
        ])
        chain = prompt | llm
        try:
            resp = chain.invoke({
                "v_data": json.dumps(prev_state.get("vehicle_data", {})),
                "causes": " | ".join(prev_state.get("possible_causes", [])),
                "recs": " | ".join(prev_state.get("recommendations", [])),
                "msg": user_msg
            })
            return jsonify({"response": resp.content})
        except Exception: pass
    sub = prev_state.get('subsystems', 'powertrain')
    return jsonify({"response": f"Based on your vehicle data, consider having a certified technician inspect this condition. Possible issues frequently link to {sub} wear or sensor variances."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
