"""
NOTEBOOK 05 — VETDX DASHBOARD (v4 — Final)
Run with: streamlit run 05_streamlit_dashboard.py

NEW IN v4:
  1. Zoonotic disease alerts — based on actual dataset (25 diseases)
  2. Contagion / isolation alerts — based on actual dataset (54 diseases)
  3. Clickable differential diagnoses — each shows description + treatment
  4. Downloadable case report (.txt)
  5. Disease knowledge panel — typical danger, case count
  6. All 12 test scenarios handled correctly
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, re, datetime, warnings, hashlib
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VetDx — Veterinary Disease Predictor",
    page_icon="🐾", layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
:root {
    --bg:#0c0f18; --surface:#131720; --surface2:#1c2131;
    --accent:#5b9cf6; --accent2:#8b5cf6;
    --success:#34d399; --warning:#fbbf24; --danger:#f87171;
    --text:#f0f4ff; --muted:#a0aab8; --border:#2e3850;
}
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:var(--bg)!important;color:var(--text);font-size:15px;}
h1,h2,h3,h4{font-family:'Playfair Display',serif;}
p,li,span,div,label{color:var(--text)!important;}
.vetdx-header{border-bottom:1px solid var(--border);padding:1.6rem 0 1.2rem;margin-bottom:1.8rem;}
.vetdx-title{font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:600;color:var(--text);margin:0;}
.vetdx-title span{color:var(--accent);font-style:italic;}
.vetdx-sub{color:var(--muted);font-size:0.9rem;margin:0.3rem 0 0;font-weight:300;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.3rem;margin-bottom:0.9rem;}
.card-title{font-family:'Playfair Display',serif;font-size:1rem;font-weight:600;color:var(--accent);margin:0 0 0.9rem;padding-bottom:0.45rem;border-bottom:1px solid var(--border);}
.result-card{background:var(--surface);border:1px solid var(--accent);border-radius:12px;padding:1.4rem;margin-bottom:0.8rem;position:relative;}
.result-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:12px 12px 0 0;}
.disease-name{font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:600;color:var(--text);margin:0.6rem 0 0.35rem;line-height:1.2;}
.badge{display:inline-block;padding:0.25rem 0.85rem;border-radius:20px;font-size:0.82rem;font-weight:500;}
.badge-critical{background:rgba(239,68,68,.18);color:#ffb3b3;border:1px solid rgba(239,68,68,.45);}
.badge-high{background:rgba(251,191,36,.15);color:#ffe99a;border:1px solid rgba(251,191,36,.4);}
.badge-moderate{background:rgba(59,130,246,.15);color:#b8d4ff;border:1px solid rgba(59,130,246,.4);}
.badge-low{background:rgba(52,211,153,.15);color:#9af0d0;border:1px solid rgba(52,211,153,.4);}
/* Alerts */
.alert-zoonotic{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.5);border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.6rem;font-size:0.88rem;color:#ffb3b3;line-height:1.6;}
.alert-contagion{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.4);border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.6rem;font-size:0.88rem;color:#ffe99a;line-height:1.6;}
/* Urgency banners */
.urgency-Critical{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.4);border-radius:8px;padding:0.7rem 1rem;font-size:0.88rem;color:#ffb3b3;margin-bottom:0.8rem;}
.urgency-High{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.35);border-radius:8px;padding:0.7rem 1rem;font-size:0.88rem;color:#ffe99a;margin-bottom:0.8rem;}
.urgency-Moderate{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);border-radius:8px;padding:0.7rem 1rem;font-size:0.88rem;color:#b8d4ff;margin-bottom:0.8rem;}
.urgency-Low{background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);border-radius:8px;padding:0.7rem 1rem;font-size:0.88rem;color:#9af0d0;margin-bottom:0.8rem;}
/* Treatment cards */
.treat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:7px;margin-top:0.5rem;}
.treat-card{background:var(--surface2);border:1px solid var(--border);border-radius:7px;padding:0.6rem 0.75rem;display:flex;align-items:flex-start;gap:0.45rem;}
.treat-card.antibiotic{border-left:3px solid #f87171;}
.treat-card.fluid{border-left:3px solid #5b9cf6;}
.treat-card.supportive{border-left:3px solid #fbbf24;}
.treat-card.surgical{border-left:3px solid #c084fc;}
.treat-card.nutritional{border-left:3px solid #34d399;}
.treat-card.topical{border-left:3px solid #fb923c;}
.treat-card.other{border-left:3px solid var(--border);}
.treat-icon{font-size:0.98rem;flex-shrink:0;margin-top:1px;}
.treat-text{font-size:0.85rem;color:var(--text);line-height:1.5;}
.treat-type{font-size:0.72rem;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:0.04em;}
/* Prob bars */
.prob-bar-bg{background:var(--border);border-radius:3px;height:5px;margin-bottom:0.6rem;}
.prob-bar-fill{height:5px;border-radius:3px;background:linear-gradient(90deg,var(--accent),var(--accent2));}
/* Metric row */
.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:0.8rem 0;}
.metric-box{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.7rem;text-align:center;}
.metric-val{font-family:'Playfair Display',serif;font-size:1.4rem;color:var(--accent);}
.metric-lbl{font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;}
/* Disease info panel */
.disease-panel{background:var(--surface2);border:1px solid var(--border);border-radius:9px;padding:0.9rem 1rem;margin-top:0.8rem;}
.disease-panel-title{font-size:0.8rem;font-weight:500;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);margin-bottom:0.4rem;}
.disease-panel-row{display:flex;gap:1.5rem;flex-wrap:wrap;font-size:0.86rem;color:var(--muted);}
.disease-panel-row strong{color:var(--text);}
/* Validation */
.val-error{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.4);border-radius:8px;padding:0.75rem 1rem;font-size:0.88rem;color:#ffb3b3;margin-bottom:0.8rem;}
.val-warn{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.3);border-radius:8px;padding:0.65rem 1rem;font-size:0.86rem;color:#ffe99a;margin-bottom:0.6rem;}
/* Preview pending banner */
.preview-banner{background:rgba(139,92,246,.12);border:1px solid rgba(139,92,246,.4);border-radius:8px;padding:0.75rem 1rem;font-size:0.88rem;color:#c4b5fd;margin-bottom:0.8rem;line-height:1.6;}
/* Misc */
.overlap-notice{background:var(--surface2);border-left:2px solid var(--accent2);padding:0.6rem 0.9rem;font-size:0.84rem;color:var(--muted);border-radius:0 6px 6px 0;margin:0.5rem 0;line-height:1.6;}
.age-unknown{display:inline-block;background:rgba(139,92,246,.15);color:#c4b5fd;border:1px solid rgba(139,92,246,.3);padding:0.25rem 0.8rem;border-radius:20px;font-size:0.82rem;margin-top:4px;}
.disclaimer{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.22);border-radius:8px;padding:0.75rem 1rem;font-size:0.83rem;color:#ffe99a;margin-top:1rem;line-height:1.6;}
.stButton>button{background:linear-gradient(135deg,#5b9cf6,#8b5cf6)!important;color:white!important;border:none!important;border-radius:8px!important;padding:0.65rem 1.5rem!important;font-family:'IBM Plex Sans',sans-serif!important;font-weight:500!important;font-size:0.95rem!important;width:100%!important;}
.save-btn>button{background:linear-gradient(135deg,#34d399,#059669)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.stSelectbox label,.stNumberInput label,.stTextInput label,.stTextArea label,.stCheckbox label{color:var(--text)!important;font-size:0.92rem!important;}
.stRadio label{color:var(--text)!important;font-size:0.92rem!important;}
.stCaption,.stMarkdown p{font-size:0.88rem!important;color:var(--muted)!important;}
section[data-testid="stSidebar"] .stMarkdown p{color:var(--muted)!important;}
.stExpander summary{font-size:0.92rem!important;color:var(--text)!important;}
div[data-testid="stMetricValue"]{font-size:1.4rem!important;color:var(--accent)!important;}
div[data-testid="stMetricLabel"]{font-size:0.78rem!important;color:var(--muted)!important;}
</style>
""", unsafe_allow_html=True)

# ── Disease lists (from actual dataset) ───────────────────────────────────────
ZOONOTIC_DISEASES = {
    'Avian Influenza Infection', 'Bluetongue', 'Bluetongue Virus',
    'Bovine Influenza', 'Bovine Parainfluenza', 'Bovine Tuberculosis',
    'Canine Influenza', 'Canine Leptospirosis', 'Chlamydia in Sheep',
    'Dermatophytosis', 'Equine Influenza', 'Equine Influenza Virus Infection',
    'Equine Leptospirosis', 'Equine Lyme Disease', 'Equine West Nile Virus',
    'Feline Chlamydia', 'Feline Chlamydiosis', 'Feline Dermatophytosis',
    'Foot and Mouth Disease', 'Foot and Mouth Disease ', 'Leptospirosis',
    'Lyme Disease', 'Psittacine Beak and Feather Disease',
    'Swine Flu', 'Swine Influenza',
}

CONTAGIOUS_DISEASES = {
    'African Swine Fever', 'Avian Influenza Infection', 'Blue Tongue Virus',
    'Bluetongue', 'Bluetongue Virus', 'Bovine Influenza', 'Bovine Leukemia Virus',
    'Bovine Parainfluenza', 'Bovine Respiratory Syncytial Virus',
    'Bovine Viral Diarrhea', 'Canine Distemper', 'Canine Distemper Virus Infection',
    'Canine Infectious Hepatitis', 'Canine Infectious Tracheobronchitis',
    'Canine Influenza', 'Canine Parvoviral Enteritis', 'Canine Parvovirus',
    'Caprine Arthritis Encephalitis Virus', 'Caprine Viral Arthritis',
    'Equine Herpesvirus', 'Equine Infectious Anemia', 'Equine Influenza',
    'Equine Influenza Virus Infection', 'Equine Viral Arteritis',
    'Equine West Nile Virus', 'Feline Calicivirus', 'Feline Coronavirus',
    'Feline Herpesvirus', 'Feline Immunodeficiency Virus',
    'Feline Infectious Peritonitis', 'Feline Infectious Peritonitis ',
    'Feline Leukemia Virus', 'Feline Panleukopenia', 'Feline Rhinotracheitis',
    'Feline Viral Rhinotracheitis', 'Foot and Mouth Disease',
    'Foot and Mouth Disease ', 'Maedi-Visna', 'Myxomatosis',
    'Porcine Circovirus Disease', 'Porcine Epidemic Diarrhea',
    'Porcine Epidemic Diarrhea Virus',
    'Porcine Reproductive and Respiratory Syndrome',
    'Rabbit Calicivirus', 'Rabbit Hemorrhagic Disease',
    'Rabbit Hemorrhagic Disease ', 'Rabbit Viral Hemorrhagic Disease',
    'Scrapie', 'Scrapie Disease', 'Swine Fever', 'Swine Influenza',
    'Viral Hemorrhagic Disease', 'Viral Infection', 'Viral Respiratory Disease',
}

URGENCY_CONFIG = {
    "Critical": ("urgency-Critical", "🚨 CRITICAL — Immediate emergency veterinary care required. Do not delay."),
    "High":     ("urgency-High",     "⚠️ HIGH PRIORITY — Urgent veterinary assessment needed within hours."),
    "Moderate": ("urgency-Moderate", "🔵 MODERATE — Veterinary consultation recommended within 24 hours."),
    "Low":      ("urgency-Low",      "🟢 LOW — Monitor closely. Schedule routine veterinary appointment."),
}

# ── Clinically accurate treatment overrides for high-frequency diseases ────────
# These override sparse or generic CSV treatment entries with specific protocols.
# For any disease not listed here, the dashboard falls back to the CSV value.
TREATMENT_OVERRIDE = {
    "Parasitic Infestation":
        "Antiparasitic medication (species-appropriate anthelmintic or antiprotozoal) · "
        "supportive care · environmental decontamination · "
        "follow-up faecal examination in 2–4 weeks",
    "Bacterial Pneumonia":
        "Broad-spectrum antibiotics (amoxicillin-clavulanate or enrofloxacin pending culture) · "
        "nebulisation therapy · anti-inflammatory support · rest · recheck in 5–7 days",
    "Leptospirosis":
        "Penicillin G or doxycycline (antibiotic phase) · IV fluid support · renal monitoring · "
        "strict biosafety protocols — zoonotic risk · notify public health if required",
    "Canine Leptospirosis":
        "Doxycycline (drug of choice) · IV fluid support · renal function monitoring · "
        "strict biosafety — zoonotic · isolate from other animals · notify public health authorities",
    "Canine Parvovirus":
        "Intensive IV fluid therapy · antiemetics (maropitant) · broad-spectrum antibiotics "
        "(secondary infection prevention) · strict isolation essential · "
        "no oral intake until vomiting controlled · monitor PCV and electrolytes",
    "Canine Parvoviral Enteritis":
        "Intensive IV fluid therapy · antiemetics · broad-spectrum antibiotics · "
        "strict isolation · nutritional support once stable · electrolyte monitoring",
    "Canine Distemper":
        "Supportive care — no specific antiviral available · anticonvulsants if neurological signs present · "
        "antibiotics for secondary bacterial infection · strict isolation · vaccination of in-contact animals",
    "Canine Distemper Virus Infection":
        "Supportive care · anticonvulsants for neurological signs · antibiotics for secondary infection · "
        "isolation · vaccination of exposed animals",
    "Feline Herpesvirus":
        "Antiviral therapy (famciclovir) · L-lysine supplementation · topical ocular treatment "
        "if corneal involvement · supportive nutrition · stress reduction · isolate from other cats",
    "Feline Rhinotracheitis":
        "Famciclovir antiviral · L-lysine · topical ocular and nasal treatment · "
        "nutritional support · isolate from other cats",
    "Feline Viral Rhinotracheitis":
        "Famciclovir antiviral · L-lysine supplementation · topical eye drops if ocular involvement · "
        "supportive nutrition and hydration · isolation",
    "Feline Upper Respiratory Infection":
        "Broad-spectrum antibiotics if secondary bacterial involvement · antiviral if herpesvirus confirmed · "
        "nutritional support · nasal decongestant · isolate from other cats · steam inhalation",
    "Feline Calicivirus":
        "Supportive care · NSAIDs for oral pain · antibiotics for secondary infection · "
        "nutritional support (assist feeding if anorexic) · isolate from other cats",
    "Feline Panleukopenia":
        "Intensive supportive care — IV fluids · antiemetics · broad-spectrum antibiotics · "
        "nutritional support · strict isolation · no specific antiviral available · blood transfusion if severe",
    "Feline Leukemia Virus":
        "No curative treatment — supportive and palliative management · immunosuppressive management · "
        "treat secondary infections aggressively · nutritional support · "
        "regular monitoring · palliative care planning",
    "Feline Immunodeficiency Virus":
        "No cure — long-term management of secondary infections · antiviral support (zidovudine in some cases) · "
        "nutritional support · indoor-only lifestyle · regular bloodwork monitoring",
    "Feline Infectious Peritonitis":
        "GS-441524 antiviral (gold standard where available) · supportive care · "
        "drainage of effusion if respiratory compromise · corticosteroids as adjunct · "
        "close monitoring of clinical response",
    "Feline Coronavirus":
        "Supportive care · manage secondary complications · monitor for FIP development · "
        "nutritional support · reduce stress · isolate if concurrent shedding",
    "Dermatophytosis":
        "Topical antifungal (miconazole/chlorhexidine shampoo twice weekly) · "
        "systemic itraconazole or terbinafine for severe or generalised cases · "
        "environmental decontamination · biosafety precautions — zoonotic risk",
    "Feline Dermatophytosis":
        "Topical miconazole/chlorhexidine shampoo · systemic itraconazole · "
        "environmental decontamination · gloves when handling — zoonotic",
    "Gastroenteritis":
        "Fluid replacement (oral or IV depending on severity) · bland diet reintroduction after 12–24 hour fast · "
        "antiemetics (maropitant) · probiotics · withhold food if vomiting present",
    "Septicemia":
        "Emergency IV broad-spectrum antibiotics · aggressive fluid resuscitation · "
        "vasopressor support if shock present · blood culture before antibiotics if possible · "
        "ICU monitoring · source control",
    "Foot and Mouth Disease":
        "Supportive wound care for oral and pedal lesions · soft feed and clean water · "
        "NSAIDs for pain · strict quarantine · mandatory reporting to veterinary authorities — notifiable disease",
    "Canine Infectious Hepatitis":
        "Supportive IV fluids · liver protectants (SAMe, milk thistle) · "
        "antibiotics for secondary infection · vitamin K if coagulopathy · "
        "monitor liver enzymes · avoid hepatotoxic drugs",
    "Canine Infectious Tracheobronchitis":
        "Mild cases — rest and supportive care · antibiotics (doxycycline) if bacterial component · "
        "antitussives for dry non-productive cough · isolate from other dogs · "
        "avoid kennels until resolved",
    "Equine Colic":
        "Pain management (flunixin meglumine) · nasogastric intubation to relieve gastric pressure · "
        "IV fluids · laxatives for impaction colic · surgical referral if medical management fails · "
        "walk in hand · monitor gut sounds",
    "Bovine Respiratory Disease":
        "Florfenicol or tulathromycin (broad-spectrum) · NSAIDs for fever and inflammation · "
        "supportive nutrition · isolate affected animals · "
        "review housing ventilation and stocking density",
    "Bovine Viral Diarrhea":
        "Supportive care — IV fluids and electrolytes · treatment of secondary infections · "
        "nutritional support · biosecurity measures · vaccination of herd · "
        "test and remove persistently infected animals",
    "Avian Influenza":
        "Supportive care only · strict biosafety and full PPE · mandatory reporting to authorities · "
        "quarantine of affected flock · antiviral (oseltamivir) under veterinary guidance only · "
        "depopulation may be required",
    "Rabbit GI Stasis":
        "Critical — immediate gut motility support (metoclopramide or cisapride) · "
        "aggressive oral and IV hydration · pain management (meloxicam) · "
        "syringe-feed critical care formula · warmth · avoid long fasting periods · "
        "high-fibre diet on recovery",
    "Viral Hemorrhagic Disease":
        "No specific treatment — supportive and palliative care · haemostatic support · "
        "fluid therapy · nutritional support · strict isolation · "
        "vaccination of in-contact animals where available",
    "Feline Chlamydia":
        "Doxycycline (first-line) for minimum 4 weeks · topical tetracycline eye ointment · "
        "treat all in-contact cats simultaneously · biosafety precautions",
    "Feline Chlamydiosis":
        "Doxycycline orally for 4–6 weeks · topical ocular antibiotics · "
        "treat all in-contact cats · monitoring for recurrence",
    "Scrapie Disease":
        "No treatment available — fatal prion disease · mandatory reporting in most jurisdictions · "
        "humane euthanasia · flock quarantine · carcass disposal per regulatory requirements",
    "Viral Infection":
        "Supportive care tailored to presenting signs · antipyretics (meloxicam) · "
        "fluid support · nutritional supplementation · "
        "monitor for secondary bacterial infection · isolate if contagious",
    "Bacterial Infection":
        "Bacterial culture and sensitivity where possible · appropriate antibiotic course · "
        "supportive care · anti-inflammatory if indicated · recheck at 5–7 days",
    "Skin Infection":
        "Topical antibacterial or antifungal depending on organism · "
        "systemic antibiotics for deep or widespread infection · "
        "medicated shampoo · address underlying cause (allergy, parasite, immunosuppression)",
    "Eye Infection":
        "Topical antibiotic eye drops or ointment (chloramphenicol or fusidic acid) · "
        "topical antiviral if herpesvirus suspected · gentle eye cleaning · "
        "e-collar to prevent self-trauma · systemic antibiotics if severe",
    "Ear Infection":
        "Clean ear canal thoroughly · topical antibiotic/antifungal/steroid combination drops · "
        "systemic antibiotics for deep or resistant infections · "
        "address underlying cause (allergy, anatomy) · recheck in 2 weeks",
    "Urinary Tract Infection":
        "Appropriate antibiotic based on culture/sensitivity (trimethoprim-sulfa or amoxicillin) · "
        "increased water intake · urinary acidifier if recurrent · "
        "monitor urine output · recheck urine culture after treatment",
    "Respiratory Infection":
        "Broad-spectrum antibiotics if bacterial · antivirals if viral confirmed · "
        "nebulisation therapy · rest · fluid support · "
        "monitor oxygen saturation · recheck in 3–5 days",
    "Gastrointestinal Infection":
        "Fluid and electrolyte replacement · antiemetics · bland diet · "
        "probiotics · antibiotics only if systemic signs present · "
        "faecal culture to identify pathogen",
    "Nutritional Deficiency":
        "Identify and correct specific deficiency · dietary supplementation · "
        "balanced species-appropriate diet · recheck bloodwork in 4–6 weeks · "
        "rule out malabsorption as underlying cause",
    "Trauma":
        "Stabilise patient · pain management (opioids or NSAIDs) · wound management · "
        "radiographs to assess skeletal injury · surgical intervention if indicated · "
        "cage rest and restricted activity during recovery",
    "Poisoning":
        "Remove from source · induce emesis if recent ingestion and safe to do so · "
        "activated charcoal · IV fluid diuresis · specific antidote if available · "
        "contact poison control for guidance · monitor organ function",
}

TREATMENT_RULES = [
    (["antibiotic","antimicrobial","iv antibiotic","penicillin","amoxicillin","tetracycline","enrofloxacin"],
     "antibiotic","💊","Antimicrobial"),
    (["antiviral","interferon","oseltamivir"],
     "antibiotic","🧪","Antiviral"),
    (["antiparasit","antihelmintic","antifungal","antifung","ivermectin","fenbendazole"],
     "antibiotic","🔬","Antiparasitic / Antifungal"),
    (["oxygen","ventilat","respiratory support","bronchodilat","nebulisation","nebulization"],
     "fluid","🫁","Respiratory Support"),
    (["fluid","iv fluid","hydration","saline","electrolyte","rehydrat","intravenous"],
     "fluid","💧","Fluid Therapy"),
    (["surgery","surgical","operation","procedure","excis","debridement"],
     "surgical","🔪","Surgical Intervention"),
    (["probiotic","nutritional","dietary","diet","supplement","vitamin","mineral","feeding"],
     "nutritional","🥗","Nutritional Support"),
    (["topical","ointment","cream","lotion","shampoo","spray","wound care","dressing","bandage"],
     "topical","🧴","Topical Treatment"),
    (["pain","analgesic","nsaid","anti-inflamm","anti inflamm","steroid","corticosteroid","meloxicam"],
     "topical","💉","Pain / Anti-inflammatory"),
    (["isolat","quarantine","biosecurity","disinfect","decontaminat","separation"],
     "supportive","🚧","Isolation / Biosecurity"),
    (["supportive","rest","monitor","follow-up","follow up","nursing","palliative","observation"],
     "supportive","🏥","Supportive Care"),
    (["environmental","management","husbandry","housing","temperature control","habitat"],
     "supportive","🏠","Environmental Management"),
    (["vaccination","vaccine","immunis","immuniz","booster"],
     "nutritional","💉","Vaccination"),
]

BREEDS_BY_SPECIES = {
    "Bird":   ["Unknown / Mixed","Eagle","Parrot","Canary","Cockatiel","Macaw"],
    "Cat":    ["Unknown / Mixed","Abyssinian","Bengal","Bombay","British Shorthair",
               "Burmese","Maine Coon","Manx","Persian","Ragdoll","Russian Blue",
               "Scottish Fold","Siamese","Siberian","Sphynx"],
    "Cow":    ["Unknown / Mixed","Aberdeen Angus","Angus","Ayrshire","Belted Galloway",
               "Brahman","Brown Swiss","Charolais","Dexter","Guernsey","Hereford",
               "Holstein","Jersey","Limousin","Red Angus","Simmental"],
    "Dog":    ["Unknown / Mixed","Akita","Beagle","Boxer","Bulldog","Chihuahua",
               "Cocker Spaniel","Dachshund","Doberman Pinscher","German Shepherd",
               "Golden Retriever","Husky","Labrador","Pit Bull","Poodle",
               "Rottweiler","Shih Tzu","Yorkshire Terrier"],
    "Dolphin":["Unknown / Mixed","Bottlenose"],
    "Fish":   ["Unknown / Mixed","Clownfish","Goldfish","Betta","Koi"],
    "Frog":   ["Unknown / Mixed","Tree Frog","Bullfrog","Dart Frog"],
    "Goat":   ["Unknown / Mixed","Alpine","Angora","Boer","Nigerian Dwarf",
               "Nubian","Saanen","Toggenburg"],
    "Horse":  ["Unknown / Mixed","American Quarter","Andalusian","Appaloosa","Arabian",
               "Belgian","Clydesdale","Morgan","Mustang","Paint","Percheron",
               "Shetland Pony","Shire","Thoroughbred","Welsh Pony"],
    "Lizard": ["Unknown / Mixed","Bearded Dragon","Gecko","Green Iguana","Leopard Gecko"],
    "Pig":    ["Unknown / Mixed","Berkshire","Chester White","Duroc","Hampshire",
               "Landrace","Large White","Yorkshire"],
    "Rabbit": ["Unknown / Mixed","Dutch","English Angora","Flemish Giant","Holland Lop",
               "Mini Lop","Mini Rex","Netherland Dwarf"],
    "Shark":  ["Unknown / Mixed","Great White","Hammerhead"],
    "Sheep":  ["Unknown / Mixed","Blackface","Cheviot","Dorset","Merino","Suffolk","Texel"],
    "Snake":  ["Unknown / Mixed","Boa Constrictor","Cobra","Corn Snake","Python"],
    "Turtle": ["Unknown / Mixed","Box Turtle","Red-eared Slider","Sea Turtle"],
}

# ── Breed-specific adult weight ranges (kg) ──────────────────────────────────
# Used to warn the practitioner when entered weight is implausible for the breed.
# These are approximate biological ranges for healthy adults.
# Age scaling is applied separately for young animals.
BREED_WEIGHT_RANGES = {
    # Dogs
    "Chihuahua":           (1.0,  3.5),
    "Yorkshire Terrier":   (1.5,  3.5),
    "Shih Tzu":            (4.0,  7.5),
    "Dachshund":           (3.5,  9.0),
    "Beagle":              (9.0, 14.0),
    "Cocker Spaniel":      (9.0, 14.5),
    "Poodle":              (4.0, 32.0),   # Toy to Standard range
    "Bulldog":             (18.0,25.0),
    "Pit Bull":            (14.0,35.0),
    "Boxer":               (22.0,36.0),
    "Akita":               (27.0,45.0),
    "Husky":               (16.0,27.0),
    "Labrador":            (25.0,36.0),
    "Golden Retriever":    (25.0,36.0),
    "German Shepherd":     (22.0,40.0),
    "Doberman Pinscher":   (27.0,45.0),
    "Rottweiler":          (36.0,60.0),
    # Cats
    "Persian":             (3.0,  6.0),
    "Siamese":             (3.0,  5.0),
    "Maine Coon":          (4.0, 11.0),
    "Bengal":              (3.5,  7.0),
    "British Shorthair":   (4.0,  8.0),
    "Ragdoll":             (5.5, 10.0),
    "Sphynx":              (3.5,  7.0),
    "Scottish Fold":       (3.0,  6.0),
    # Rabbits
    "Netherland Dwarf":    (0.8,  1.3),
    "Holland Lop":         (1.5,  2.3),
    "Mini Lop":            (1.8,  3.2),
    "Mini Rex":            (1.6,  2.3),
    "Dutch":               (1.8,  2.5),
    "Flemish Giant":       (5.0,  9.0),
    "English Angora":      (2.5,  4.5),
    # Horses
    "Shetland Pony":       (100, 200),
    "Welsh Pony":          (150, 300),
    "Thoroughbred":        (450, 600),
    "Shire":               (700, 1100),
    "Clydesdale":          (600, 1000),
    "Belgian":             (700, 1000),
    "Arabian":             (360, 500),
    # Goats
    "Nigerian Dwarf":      (9.0, 18.0),
    "Boer":                (55.0,135.0),
    "Angora":              (38.0, 55.0),
}

# Age-based weight scaling multipliers
# For very young animals, adult weight is scaled down
def get_weight_range(species, breed, age):
    """Return (min_kg, max_kg, source) biologically plausible for this animal.
    Source indicates whether breed-level or species-level range was used."""
    breed_range = BREED_WEIGHT_RANGES.get(breed)
    species_range = SPECIES_RANGES.get(species, {}).get("weight", (0.1, 2000))

    base_min, base_max = breed_range if breed_range else species_range
    source = "breed" if breed_range else "species"

    # Age scaling: very young animals are a fraction of adult weight
    if age is not None and isinstance(age, (int, float)) and age > 0:
        if age < 0.25:      scale = 0.05   # <3 months: ~5% adult weight
        elif age < 0.5:     scale = 0.15   # 3-6 months
        elif age < 1.0:     scale = 0.40   # 6-12 months
        elif age < 2.0:     scale = 0.75   # 1-2 years
        else:               scale = 1.0    # adult
        base_min = max(0.01, base_min * scale)
        base_max = base_max * scale if scale < 1.0 else base_max

    return base_min, base_max, source


SPECIES_RANGES = {
    "Dog":    {"temp":(37.5,39.2),"hr":(60,140), "weight":(2,90)},
    "Cat":    {"temp":(38.0,39.2),"hr":(140,220),"weight":(2,10)},
    "Horse":  {"temp":(37.2,38.6),"hr":(28,44),  "weight":(400,600)},
    "Cow":    {"temp":(38.0,39.5),"hr":(48,84),  "weight":(300,700)},
    "Rabbit": {"temp":(38.5,40.0),"hr":(130,325),"weight":(1,8)},
    "Bird":   {"temp":(40.0,42.0),"hr":(200,400),"weight":(0.02,2)},
    "Goat":   {"temp":(38.5,40.0),"hr":(70,90),  "weight":(20,90)},
    "Sheep":  {"temp":(38.5,40.0),"hr":(70,90),  "weight":(40,100)},
    "Pig":    {"temp":(38.0,40.0),"hr":(55,86),  "weight":(50,300)},
    "Dolphin":{"temp":(36.0,37.5),"hr":(60,130), "weight":(100,600)},
    "Fish":   {"temp":(20.0,28.0),"hr":(40,80),  "weight":(0.01,20)},
    "Frog":   {"temp":(18.0,28.0),"hr":(20,60),  "weight":(0.01,1)},
    "Lizard": {"temp":(30.0,38.0),"hr":(20,60),  "weight":(0.05,5)},
    "Snake":  {"temp":(26.0,32.0),"hr":(20,50),  "weight":(0.1,10)},
    "Turtle": {"temp":(24.0,30.0),"hr":(20,40),  "weight":(0.1,50)},
    "Shark":  {"temp":(15.0,25.0),"hr":(20,60),  "weight":(500,2000)},
}

CLINICAL_KEYWORDS = {
    "vomit","vomiting","diarrhea","diarrhoea","lethargy","lethargic","appetite",
    "eating","drinking","weight","cough","coughing","sneeze","discharge","nasal",
    "eye","ear","limp","lameness","swelling","swollen","pain","painful","fever",
    "temperature","shiver","seizure","collapse","weakness","breathing","breath",
    "respiratory","wheeze","skin","lesion","wound","itching","rash","bleed",
    "urine","urination","feces","stool","blood","pale","gum","dehydration",
    "not eating","not drinking","not moving","stopped","sudden","gradual",
    "acute","chronic","since","days","weeks","hours","yesterday","coat","fur",
    "thin","obese","wasting","incoordination","stumbling","blindness","deaf",
    "swallowing","regurgitation","posture","hunched","sore","bite","trauma",
    "injury","foreign","ingested","abdomen","belly","chest","back","tail",
    "nose","mouth","teeth","paw","hoof","scratching","inflamed","inflammation",
    "infection","abscess","mass","lump","drool","mucus","pus","constipation",
    "bloat","gas","seizure","shaking","tremor","discharge","spotted","rash",
    "limping","hobbling","dragging","paralysis","unsteady","falling","circling",
    "hiding","withdrawn","aggressive","restless","whimpering","crying","quiet",
    "weak","tired","slow","lethargic","inactive","collapsed","unconscious",
    "bloody","watery","smelly","foul","odour","odor","smell","green","yellow",
    "found","stray","rescue","hit","accident","attack","bite","sting",
    "swallowed","ingested","ate","chewed",
}

HISTORY_FILE = "prediction_history.json"
STRUCT_COLS  = ['Age','Weight','Appetite_Loss','Vomiting','Diarrhea',
                'Coughing','Labored_Breathing','Lameness','Skin_Lesions',
                'Nasal_Discharge','Eye_Discharge','Body_Temperature',
                'Heart_Rate','Symptom_Count','Temp_Risk','Symptom_Risk',
                'Critical_Risk','Dangerous_Level']
SPECIES_BREED_COLS = ['Species','Breed','Gender']

# ── Engineered features — MUST exactly match Notebook 03 / 04 ──────────────────
SPECIES_VITAL_RANGES = {
    'Dog':     {'temp': (37.5, 39.2), 'hr': (60, 140)},
    'Cat':     {'temp': (38.0, 39.2), 'hr': (140, 220)},
    'Cow':     {'temp': (38.0, 39.3), 'hr': (48, 84)},
    'Horse':   {'temp': (37.2, 38.3), 'hr': (28, 44)},
    'Bird':    {'temp': (39.0, 42.2), 'hr': (200, 400)},
    'Rabbit':  {'temp': (38.3, 39.7), 'hr': (130, 325)},
    'Goat':    {'temp': (38.5, 39.7), 'hr': (70, 135)},
    'Sheep':   {'temp': (38.5, 39.9), 'hr': (70, 90)},
    'Pig':     {'temp': (38.7, 39.8), 'hr': (60, 90)},
    'Snake':   {'temp': (25.0, 32.0), 'hr': (20, 70)},
    'Lizard':  {'temp': (26.0, 35.0), 'hr': (30, 80)},
    'Turtle':  {'temp': (22.0, 28.0), 'hr': (20, 40)},
    'Fish':    {'temp': (18.0, 26.0), 'hr': (60, 100)},
    'Frog':    {'temp': (20.0, 28.0), 'hr': (30, 90)},
    'Shark':   {'temp': (18.0, 24.0), 'hr': (20, 60)},
    'Dolphin': {'temp': (36.0, 37.5), 'hr': (90, 120)},
}
DEFAULT_VITAL_RANGE = {'temp': (37.5, 40.0), 'hr': (60, 160)}

def compute_vital_deviation(row_species, body_temp, heart_rate):
    r = SPECIES_VITAL_RANGES.get(row_species, DEFAULT_VITAL_RANGE)
    t_mid = (r['temp'][0] + r['temp'][1]) / 2
    t_rng = (r['temp'][1] - r['temp'][0]) / 2
    h_mid = (r['hr'][0]   + r['hr'][1])   / 2
    h_rng = (r['hr'][1]   - r['hr'][0])   / 2
    temp_dev = (body_temp  - t_mid) / t_rng if t_rng > 0 else 0.0
    hr_dev   = (heart_rate - h_mid) / h_rng if h_rng > 0 else 0.0
    return temp_dev, hr_dev

STRUCT_COLS_EXTENDED = STRUCT_COLS + [
    'Temp_Dev_Species', 'HR_Dev_Species',
    'GI_Combo', 'Resp_Combo', 'Severity_x_Age',
    'Temp_x_Symptoms', 'Multi_System', 'HR_x_Risk'
]


# ── Treatment parsing ──────────────────────────────────────────────────────────
def parse_treatments(treatment_str):
    if not treatment_str or str(treatment_str) in ('Not_Recorded','Not recorded','None','nan'):
        return []
    raw   = [t.strip().strip('.') for t in str(treatment_str).split(',') if t.strip()]
    cards = []
    seen_categories = set()
    for item in raw:
        if len(item) < 3: continue
        if item.lower().startswith(('and ','or ','with ','including ')): continue
        item_lower = item.lower()
        matched = False
        for keywords, css, icon, category in TREATMENT_RULES:
            if any(kw in item_lower for kw in keywords):
                if category not in seen_categories:
                    cards.append({'text':item,'css':css,'icon':icon,'category':category})
                    seen_categories.add(category)
                matched = True
                break
        if not matched and item not in seen_categories:
            cards.append({'text':item,'css':'other','icon':'🩺','category':'Veterinary Treatment'})
            seen_categories.add(item)
    return cards


def render_treatment_cards(treatment_str, danger_category):
    """Return HTML for urgency banner + treatment cards."""
    urg_cls, urg_msg = URGENCY_CONFIG.get(danger_category, URGENCY_CONFIG["Moderate"])
    html = f'<div class="{urg_cls}">{urg_msg}</div>'
    cards = parse_treatments(treatment_str)
    if not cards:
        html += ('<div style="color:var(--muted);font-size:0.83rem;padding:0.5rem 0">'
                 '📋 Specific treatment protocol not available. Consult a veterinary specialist.</div>')
        return html
    html += '<div class="treat-grid">'
    for c in cards:
        html += (f'<div class="treat-card {c["css"]}">'
                 f'<span class="treat-icon">{c["icon"]}</span>'
                 f'<div><div class="treat-text">{c["text"]}</div>'
                 f'<div class="treat-type">{c["category"]}</div></div></div>')
    html += '</div>'
    return html


# ── Input validation ───────────────────────────────────────────────────────────
def validate_symptom(text):
    text = text.strip()
    if not text:
        return False, "Please enter a primary presenting complaint."
    if len(text) < 5:
        return False, "Please describe the symptom in more detail (at least a few words)."
    if len(text) > 500:
        return False, "Description too long. Please keep it under 500 characters."
    if re.match(r'^[\d\s\W]+$', text):
        return False, "Please use words to describe the symptoms, not only numbers or symbols."
    if len(set(text.lower().replace(' ',''))) < 4:
        return False, "This does not look like a clinical description. Please describe what you observed."
    for row in ['qwertyuiop','asdfghjkl','zxcvbnm']:
        for length in range(5, min(len(text)+1,10)):
            for i in range(len(row)-length+1):
                if row[i:i+length] in text.lower().replace(' ',''):
                    return False, "This looks like random keyboard input. Please describe the animal's actual symptoms."
    if sum(1 for c in text if c.isalpha()) < 4:
        return False, "Please use words to describe the symptoms."
    words = [w.lower().strip('.,!?;:') for w in text.split() if len(w) > 1]
    if len(words) < 2:
        return False, "Please provide at least a 2-word description (e.g. 'severe vomiting' or 'not eating')."
    # Use word-boundary matching, not plain substring matching — otherwise
    # short keywords like "thin" or "ear" false-positive inside unrelated
    # words (e.g. "something" contains "thin", "wear" contains "ear").
    text_lower = text.lower()
    has_clinical = any(
        re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        for kw in CLINICAL_KEYWORDS
    )
    if not has_clinical:
        common = {'limp','lame','weak','sick','ill','pale','thin','hot','cold',
                  'wet','dry','swollen','hurt','bad','found','stray'}
        if not any(w in common for w in words):
            return True, "⚠️ We could not recognise specific clinical terms — prediction may be less accurate. Please check your description."
    return True, ""


def overlap_warnings(primary, additional, checkboxes):
    combined = (primary + ' ' + ' '.join(additional)).lower()
    rules = {
        "Appetite_Loss":     (["appetite","not eating","anorexia","off food","inappetence"],"Appetite Loss"),
        "Vomiting":          (["vomit","vomiting","threw up","nausea"],"Vomiting"),
        "Diarrhea":          (["diarrhea","diarrhoea","loose stool","watery stool"],"Diarrhea"),
        "Coughing":          (["cough","coughing","hacking"],"Coughing"),
        "Labored_Breathing": (["breathing","respiratory","wheeze","wheezing","breathless"],"Labored Breathing"),
        "Lameness":          (["limp","lame","limping","not walking","hobbling"],"Lameness"),
        "Skin_Lesions":      (["skin","lesion","rash","wound","sore","ulcer"],"Skin Lesions"),
        "Nasal_Discharge":   (["nasal","runny nose","nose discharge","snot"],"Nasal Discharge"),
        "Eye_Discharge":     (["eye discharge","weeping eye","ocular discharge"],"Eye Discharge"),
    }
    warns = []
    for flag, (terms, label) in rules.items():
        if checkboxes.get(flag):
            for t in terms:
                if t in combined:
                    warns.append(f'"{label}" already ticked — no need to mention "{t}" in text too.')
                    break
    return warns


def is_disease_zoonotic(name):
    n = name.strip()
    return n in ZOONOTIC_DISEASES or any(
        z.strip().lower() == n.lower() for z in ZOONOTIC_DISEASES)


def is_disease_contagious(name):
    n = name.strip()
    return n in CONTAGIOUS_DISEASES or any(
        c.strip().lower() == n.lower() for c in CONTAGIOUS_DISEASES)


# ── Model loaders ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        le            = joblib.load('../models/saved_models/label_encoder.pkl')
        struct_scaler = joblib.load('../models/saved_models/struct_scaler.pkl')
        ord_enc       = joblib.load('../models/saved_models/species_breed_encoder.pkl')
        with open('../results/best_classifier.txt') as f:
            cfg = dict(l.strip().split('=') for l in f.readlines())
        if 'XGBoost' in cfg['best_classifier']:
            import xgboost as xgb
            clf = xgb.XGBClassifier()
            clf.load_model('../models/saved_models/xgboost.json')
        else:
            clf = joblib.load('../models/saved_models/random_forest.pkl')
        return le, struct_scaler, ord_enc, clf, cfg
    except Exception as e:
        return None, None, None, None, {"error": str(e)}


@st.cache_resource
def load_shap_explainer():
    """Load SHAP TreeExplainer for XGBoost — fast exact Shapley values."""
    try:
        import shap, xgboost as xgb
        clf = xgb.XGBClassifier()
        clf.load_model('../models/saved_models/xgboost.json')
        return shap.TreeExplainer(clf)
    except Exception:
        return None


@st.cache_resource
def load_transformer():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        with open('../results/best_transformer.txt') as f:
            cfg = dict(l.strip().split('=') for l in f.readlines())
        name = cfg['best_transformer'].replace('_finetuned','')
        path = f'../models/saved_models/{name}_finetuned'
        tok  = AutoTokenizer.from_pretrained(path)
        mdl  = AutoModel.from_pretrained(path)
        dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mdl.eval(); mdl.to(dev)
        return tok, mdl, dev
    except Exception:
        return None, None, None


@st.cache_data
def load_disease_kb():
    try:
        df  = pd.read_csv('../data/animal_disease_merged_cleaned.csv')
        db  = {}
        for disease, group in df.groupby('Disease_Prediction'):
            tr = group[group['Treatment']   != 'Not_Recorded']
            ds = group[group['Description'] != 'Not_Recorded']
            dc = group['Danger_Level_Category'].value_counts()
            disease_clean = disease.strip()

            # Per-species case counts so dashboard shows species-relevant number
            species_counts = {}
            if 'Species' in df.columns:
                species_counts = group['Species'].value_counts().to_dict()

            db[disease_clean] = {
                'treatment':      TREATMENT_OVERRIDE.get(disease_clean) or
                                  (tr['Treatment'].iloc[0] if len(tr) > 0 else None),
                'description':    ds['Description'].iloc[0] if len(ds) > 0 else None,
                'typical_danger': dc.index[0] if len(dc) > 0 else 'Moderate',
                'n_cases':        len(group),          # total across all species
                'species_counts': species_counts,       # {species: count} breakdown
            }
        for disease_name, treatment_text in TREATMENT_OVERRIDE.items():
            if disease_name not in db:
                db[disease_name] = {
                    'treatment':      treatment_text,
                    'description':    None,
                    'typical_danger': 'Moderate',
                    'n_cases':        0,
                    'species_counts': {},
                }
        return db
    except Exception:
        return {
            name: {
                'treatment':      text,
                'description':    None,
                'typical_danger': 'Moderate',
                'n_cases':        0,
                'species_counts': {},
            }
            for name, text in TREATMENT_OVERRIDE.items()
        }


def extract_embedding(text, tokenizer, model, device):
    import torch
    enc = tokenizer(text, padding=True, truncation=True,
                    max_length=128, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        cls = model(**enc).last_hidden_state[:, 0, :]
    return cls.cpu().numpy()


def compute_risk(temperature, heart_rate, symptom_count, species):
    r = SPECIES_RANGES.get(species, {"temp":(37.5,40.0),"hr":(60,160)})
    t_min, t_max = r["temp"]
    h_min, h_max = r["hr"]
    t_range = t_max - t_min
    h_range = h_max - h_min
    temp_risk    = max(0.0, min(1.0,(temperature-t_min)/t_range if t_range>0 else 0.5))
    # Heart-rate deviation from midpoint, normalised to [0,1]
    hr_mid       = (h_min + h_max) / 2
    hr_risk      = max(0.0, min(1.0, abs(heart_rate - hr_mid) / (h_range/2) if h_range>0 else 0.0))
    symptom_risk = min(1.0, symptom_count/8.0)
    critical_risk = (1.0 if (temperature>t_max+1.0 or heart_rate>h_max*1.2 or symptom_count>=6)
                     else 0.5 if (temperature>t_max or heart_rate>h_max or symptom_count>=4) else 0.0)
    dangerous_level = (temp_risk*35) + (hr_risk*20) + (symptom_risk*25) + (critical_risk*20)
    return temp_risk, symptom_risk, critical_risk, dangerous_level


def get_danger(dl):
    return "Critical" if dl>=75 else "High" if dl>=50 else "Moderate" if dl>=25 else "Low"


def danger_badge(level):
    icons = {"Critical":"🔴","High":"🟠","Moderate":"🟡","Low":"🟢"}
    return f'<span class="badge badge-{level.lower()}">{icons.get(level,"⚪")} {level} Risk</span>'


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Optional: validate that data is a list
            if isinstance(data, list):
                return data
            else:
                return []
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # Log the error (visible in Streamlit terminal)
        st.warning(f"History file corrupted, starting fresh. Error: {e}")
        # Optionally rename the corrupted file for later inspection
        if os.path.exists(HISTORY_FILE):
            backup = HISTORY_FILE + ".corrupt"
            os.rename(HISTORY_FILE, backup)
        return []


def save_history(history):
    # Write to a temporary file first, then replace the original
    temp_file = HISTORY_FILE + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        os.replace(temp_file, HISTORY_FILE)   # atomic on most OS
    except Exception as e:
        st.error(f"Failed to save history: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)


def make_patient_id(patient_name, species, breed=""):
    """Stable ID from name + species + breed.
    'Max' the German Shepherd and 'Max' the Labrador → different IDs.
    'Max' the Cat and 'Max' the Dog → different IDs.
    The same Max (German Shepherd, Dog) returning → same ID every time.
    Limitation: two unrelated animals with identical name + species + breed
    cannot be distinguished without a clinic registration number."""
    key = (f"{str(patient_name).strip().lower()}::"
           f"{str(species).strip().lower()}::"
           f"{str(breed).strip().lower()}")
    return hashlib.md5(key.encode('utf-8')).hexdigest()[:10]


def find_patient_history(history, patient_name, species, breed=""):
    """Return all prior visits for this patient, oldest first."""
    if not patient_name or not str(patient_name).strip():
        return []
    pid = make_patient_id(patient_name, species, breed)
    return [r for r in history if r.get('patient_id') == pid]


def generate_report(patient_name, species, breed, age, gender, weight,
                     age_unknown, temperature, heart_rate, symptom_count,
                     primary_symptom, predicted, conf_pct, danger_category,
                     dangerous_level, top10_names, top10_probs,
                     treatment_str, vet_notes, is_zoonotic, is_contagious,
                     shap_drivers=None):
    lines = [
        "="*60,
        "VETDX — VETERINARY DISEASE PREDICTION REPORT",
        f"Asia Pacific University of Technology and Innovation (APU)",
        "="*60,
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}",
        f"Patient   : {patient_name}",
        f"Species   : {species}",
        f"Breed     : {breed}",
        f"Age       : {'Unknown (stray/rescue)' if age_unknown else str(age)+' years'}",
        f"Gender    : {gender}",
        f"Weight    : {weight} kg",
        "",
        "VITAL SIGNS",
        "-"*40,
        f"Body Temperature : {temperature} °C",
        f"Heart Rate       : {heart_rate} bpm",
        f"Symptoms Flagged : {symptom_count}",
        f"Risk Score       : {dangerous_level:.1f}%",
        f"Risk Category    : {danger_category}",
        "",
        "PRESENTING COMPLAINT",
        "-"*40,
        primary_symptom,
        "",
        "PRIMARY PREDICTION",
        "-"*40,
        f"Disease    : {predicted}",
        f"Confidence : {conf_pct:.1f}%",
        f"Risk Level : {danger_category}",
    ]
    if is_zoonotic:
        lines += ["",
                  "*** ZOONOTIC DISEASE ALERT ***",
                  "This condition can potentially transmit to humans.",
                  "Advise strict hygiene and consult public health authorities if required."]
    if is_contagious:
        lines += ["",
                  "*** CONTAGION RISK ***",
                  "This condition may be transmissible to other animals.",
                  "Consider isolating the patient until diagnosis is confirmed."]
    # SHAP drivers — key clinical decision transparency
    if shap_drivers:
        lines += ["", "KEY PREDICTION DRIVERS (SHAP ANALYSIS)", "-"*40,
                  "The following clinical features most influenced this prediction:"]
        for i, (feat, val) in enumerate(shap_drivers, 1):
            direction = "↑ positive influence" if val > 0 else "↓ negative influence"
            lines.append(f"  {i}. {feat:<35} {direction}  (Shapley value: {val:+.4f})")
        lines.append("Note: Positive = increases probability of this diagnosis.")
        lines.append("      Negative = decreases probability of this diagnosis.")
    lines += ["","TOP 10 DIFFERENTIAL DIAGNOSES","-"*40]
    for i,(name,prob) in enumerate(zip(top10_names,top10_probs),1):
        zoo_tag  = " [ZOONOTIC]"   if is_disease_zoonotic(name)   else ""
        cont_tag = " [CONTAGIOUS]" if is_disease_contagious(name) else ""
        lines.append(f"  #{i:2d}  {name.strip():<45}{prob*100:.1f}%{zoo_tag}{cont_tag}")
    if treatment_str and str(treatment_str) not in ('Not_Recorded','Not recorded','None','nan'):
        cards = parse_treatments(treatment_str)
        if cards:
            lines += ["","TREATMENT RECOMMENDATIONS","-"*40]
            for c in cards:
                lines.append(f"  {c['icon']}  {c['text']}  ({c['category']})")
    if vet_notes:
        lines += ["","VETERINARIAN NOTES","-"*40, vet_notes]
    lines += [
        "","="*60,
        "DISCLAIMER: This report is generated by a clinical decision",
        "support system (VetDx). It is NOT a substitute for professional",
        "veterinary diagnosis or treatment. Always confirm with physical",
        "examination, laboratory tests, and qualified clinical expertise.",
        "="*60,
    ]
    return "\n".join(lines)


# ── Load resources ─────────────────────────────────────────────────────────────
le, struct_scaler, ord_enc, classifier, cfg = load_models()
tokenizer, bert_model, bert_device          = load_transformer()
disease_kb                                  = load_disease_kb()
models_loaded = le is not None and classifier is not None
bert_loaded   = tokenizer is not None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vetdx-header">
<p class="vetdx-title">Vet<span>Dx</span></p>
<p class="vetdx-sub">AI-Powered Veterinary Disease Prediction &nbsp;·&nbsp;
16 species &nbsp;·&nbsp; 162 diseases &nbsp;·&nbsp; Clinical decision support</p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    # Handle navigation override from drill-down buttons
    _override = st.session_state.pop('_page_override', None)
    page = st.radio("", ["🔍 New Prediction","📋 History","📊 Analytics","ℹ️ About"],
                    label_visibility="collapsed",
                    index=["🔍 New Prediction","📋 History","📊 Analytics","ℹ️ About"].index(_override)
                          if _override else 0)
    st.markdown("---")
    st.markdown("**🩺 Quick Reference**")
    _sp_ref = st.selectbox(
        "Species vitals", list(BREEDS_BY_SPECIES.keys()),
        label_visibility="collapsed", key="sidebar_vitals_ref"
    )
    _r = SPECIES_RANGES.get(_sp_ref, {"temp":(37.5,40.0),"hr":(60,140),"weight":(2,90)})
    st.markdown(f"""
    <div style='font-size:0.74rem;color:var(--muted);line-height:1.85;background:var(--surface2);
         border:1px solid var(--border);border-radius:7px;padding:0.55rem 0.8rem;margin-bottom:0.4rem'>
    🌡️ Temp &nbsp;<strong style='color:var(--text)'>{_r['temp'][0]}–{_r['temp'][1]} °C</strong><br>
    💓 HR &nbsp;&nbsp;&nbsp;<strong style='color:var(--text)'>{_r['hr'][0]}–{_r['hr'][1]} bpm</strong><br>
    ⚖️ Wt &nbsp;&nbsp;&nbsp;<strong style='color:var(--text)'>{_r['weight'][0]}–{_r['weight'][1]} kg</strong>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    if models_loaded:
        st.success("✅ Models loaded")
        st.caption(f"Classifier: {cfg.get('best_classifier','—')[:28]}")
        st.caption(f"Transformer: {cfg.get('best_transformer','—')}")
    else:
        st.error("❌ Models not found — run Notebooks 01–03")
    if bert_loaded:
        st.success("✅ BERT loaded")
    else:
        st.warning("⚠️ BERT not loaded")
    st.markdown("---")
    # Live session stats
    _sidebar_hist = load_history()
    _n_total   = len(_sidebar_hist)
    _n_zoo     = sum(1 for r in _sidebar_hist if r.get('is_zoonotic'))
    _n_high    = sum(1 for r in _sidebar_hist if r.get('danger_category') in ('High','Critical'))
    _n_species = len(set(r.get('species','') for r in _sidebar_hist))
    st.markdown("**📋 Session Stats**")
    st.markdown(f"""
    <div style='font-size:0.74rem;color:var(--muted);line-height:1.9;background:var(--surface2);
         border:1px solid var(--border);border-radius:7px;padding:0.55rem 0.8rem;margin-bottom:0.4rem'>
    🐾 Cases saved &nbsp;<strong style='color:var(--text)'>{_n_total}</strong><br>
    🧬 Species seen &nbsp;<strong style='color:var(--text)'>{_n_species}</strong><br>
    ⚠️ High/Critical &nbsp;<strong style='color:#fbbf24'>{_n_high}</strong><br>
    🦠 Zoonotic &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong style='color:#f87171'>{_n_zoo}</strong>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📊 Model Performance**")
    st.markdown("""
    <div style='font-size:0.75rem;color:var(--muted);line-height:2;background:var(--surface2);
         border:1px solid var(--border);border-radius:7px;padding:0.6rem 0.8rem'>
    AUC-ROC &nbsp;&nbsp;<strong style='color:#34d399'>0.9843</strong><br>
    Top-5 Acc &nbsp;<strong style='color:#34d399'>99.00%</strong><br>
    Top-3 Acc &nbsp;<strong style='color:#fbbf24'>73.66%</strong><br>
    F1 Macro &nbsp;&nbsp;<strong style='color:#fbbf24'>0.1868</strong><br>
    Classes &nbsp;&nbsp;&nbsp;<strong style='color:var(--text)'>162</strong><br>
    Species &nbsp;&nbsp;&nbsp;<strong style='color:var(--text)'>16</strong>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1 — NEW PREDICTION
# ==============================================================================
if page == "🔍 New Prediction":

    # ── Form reset counter ────────────────────────────────────────────────────
    # Incrementing this suffix changes all widget keys so Streamlit treats them
    # as brand-new widgets and renders them empty — the only reliable clear method.
    if 'form_counter' not in st.session_state:
        st.session_state['form_counter'] = 0
    _fc = st.session_state['form_counter']

    col_form, col_res = st.columns([1,1], gap="large")

    with col_form:
        # Patient info
        st.markdown('<div class="card"><p class="card-title">🐾 Patient Information</p>',
                    unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1: species = st.selectbox("Species *", list(BREEDS_BY_SPECIES.keys()))
        with c2: breed   = st.selectbox("Breed", BREEDS_BY_SPECIES[species],
                                         help="Select 'Unknown / Mixed' if breed is not known")
        c3,c4 = st.columns(2)
        with c3: gender = st.selectbox("Gender", ["Unknown","Male","Female"])
        with c4:
            wr = SPECIES_RANGES.get(species,{}).get("weight",(0.1,1000))
            weight = st.number_input("Weight (kg) *", min_value=0.01, max_value=2000.0,
                                      value=float(round((wr[0]+wr[1])/2,1)),
                                      help=f"Typical {species}: {wr[0]}–{wr[1]} kg")
        st.markdown("**Age**")
        age_unknown = st.checkbox("Age unknown — stray / rescue / uncertain")
        if age_unknown:
            age = 5
            st.markdown('<span class="age-unknown">📋 Using estimated median age internally</span>',
                        unsafe_allow_html=True)
        else:
            age = st.number_input("Age (years)", min_value=0, max_value=60, value=3)

        # ── Breed + age aware weight plausibility check ───────────────
        # Only warn when a breed-specific range is known (not Unknown/Mixed)
        _w_age = None if age_unknown else age
        _w_min, _w_max, _w_src = get_weight_range(species, breed, _w_age)
        _default_weight = float(round((SPECIES_RANGES.get(species,{}).get("weight",(0.1,1000))[0]+
                                       SPECIES_RANGES.get(species,{}).get("weight",(0.1,1000))[1])/2,1))
        _weight_changed = abs(weight - _default_weight) > 0.05

        if _w_src == "breed" and _weight_changed:
            if weight < _w_min * 0.5:
                st.warning(
                    f"⚠️ **Weight may be too low** — {weight:.1f} kg is below the expected minimum "
                    f"for a {breed} "
                    f"({'age-adjusted ' if (_w_age and _w_age < 2) else ''}"
                    f"range: {_w_min:.1f}–{_w_max:.1f} kg). "
                    f"Please verify the entry."
                )
            elif weight > _w_max * 1.5:
                st.warning(
                    f"⚠️ **Weight may be too high** — {weight:.1f} kg exceeds the expected maximum "
                    f"for a {breed} "
                    f"({'age-adjusted ' if (_w_age and _w_age < 2) else ''}"
                    f"range: {_w_min:.1f}–{_w_max:.1f} kg). "
                    f"Please verify the entry."
                )
            else:
                st.caption(
                    f"✅ Weight {weight:.1f} kg is within the expected range for a "
                    f"{breed}"
                    f"{f' at {_w_age} yr' if _w_age and _w_age < 2 else ''} "
                    f"({_w_min:.1f}–{_w_max:.1f} kg)."
                )
        elif _w_src == "species" and _weight_changed:
            _sp_min, _sp_max = SPECIES_RANGES.get(species,{}).get("weight",(0.1,1000))
            if weight < _sp_min * 0.3 or weight > _sp_max * 2.0:
                st.warning(
                    f"⚠️ **Weight may be implausible** — {weight:.1f} kg is outside the typical "
                    f"range for {species} ({_sp_min}–{_sp_max} kg). Please verify."
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # Vital signs
        st.markdown('<div class="card"><p class="card-title">💓 Vital Signs</p>',
                    unsafe_allow_html=True)
        normal = SPECIES_RANGES.get(species,{"temp":(37.5,40.0),"hr":(60,160)})
        c5,c6 = st.columns(2)
        with c5:
            temperature = st.number_input("Body Temperature (°C) *",
                min_value=20.0, max_value=45.0, value=38.5, step=0.1,
                help=f"Normal {species}: {normal['temp'][0]}–{normal['temp'][1]} °C")
            if temperature > normal["temp"][1]: st.warning(f"⚠️ Above normal ({normal['temp'][1]} °C)")
            elif temperature < normal["temp"][0]: st.info(f"ℹ️ Below normal ({normal['temp'][0]} °C)")
        with c6:
            hr_def = int((normal["hr"][0]+normal["hr"][1])//2)
            heart_rate = st.number_input("Heart Rate (bpm) *",
                min_value=10, max_value=500, value=hr_def,
                help=f"Normal {species}: {normal['hr'][0]}–{normal['hr'][1]} bpm")
            if heart_rate > normal["hr"][1]: st.warning(f"⚠️ Above normal ({normal['hr'][1]} bpm)")
            elif heart_rate < normal["hr"][0]: st.info(f"ℹ️ Below normal ({normal['hr'][0]} bpm)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Symptoms
        st.markdown('<div class="card"><p class="card-title">🩺 Clinical Symptoms</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="overlap-notice">
        <strong>Two-part symptom input:</strong><br>
        • <strong>Text description</strong> — describe what you observed (feeds the clinical decision model)<br>
        • <strong>Indicator flags</strong> — tick confirmed symptoms (feeds the clinical prediction model)<br>
        <em>Avoid repeating in the text what you have already ticked below.</em>
        </div>""", unsafe_allow_html=True)

        primary_symptom = st.text_input("Primary presenting complaint *",
            placeholder="e.g. Has not eaten since Tuesday, seems lethargic and hunched over",
            help="Describe what brought the animal in. Do not repeat symptoms you will tick below.",
            key=f"symptom_text_primary_{_fc}")
        additional_text = st.text_area("Additional observations (one per line, optional)",
            placeholder="unusual posture\npale gums\nswollen abdomen",
            height=70, key=f"symptom_text_additional_{_fc}")
        additional_symptoms = ([s.strip() for s in additional_text.strip().split('\n') if s.strip()]
                               if additional_text.strip() else [])
        duration = st.selectbox("Duration of illness *",
            ["Less than 24 hours","1 day","2 days","3 days","4 days","5 days",
             "6 days","7 days","10 days","14 days","3 weeks",
             "More than 3 weeks","Unknown / Gradual onset"])

        st.markdown("**Tick all symptoms confirmed present:**")
        scols = st.columns(3)
        symptom_flags = {}
        symptom_labels = {
            "Appetite_Loss":"Appetite Loss","Vomiting":"Vomiting","Diarrhea":"Diarrhea",
            "Coughing":"Coughing","Labored_Breathing":"Labored Breathing","Lameness":"Lameness",
            "Skin_Lesions":"Skin Lesions","Nasal_Discharge":"Nasal Discharge",
            "Eye_Discharge":"Eye Discharge",
        }
        for i,(key,label) in enumerate(symptom_labels.items()):
            with scols[i%3]:
                symptom_flags[key] = st.checkbox(label, key=f"symptom_cb_{key}_{_fc}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Notes
        st.markdown('<div class="card"><p class="card-title">📝 Case Notes</p>',
                    unsafe_allow_html=True)
        patient_name = st.text_input("Patient name / ID (optional)",
                                      placeholder="e.g. Max, Stray #007",
                                      key=f"patient_name_input_{_fc}")

        # ── Returning patient lookup ────────────────────────────────────────────
        if patient_name and patient_name.strip():
            _hist_for_lookup = load_history()
            _prior_visits = find_patient_history(_hist_for_lookup, patient_name, species, breed)
            if _prior_visits:
                _last = _prior_visits[-1]
                _last_date = _last['timestamp'][:10]
                st.markdown(f"""
                <div style='background:var(--surface2);border:1px solid var(--accent);
                     border-radius:7px;padding:0.55rem 0.8rem;margin:0.3rem 0 0.6rem;
                     font-size:0.78rem;color:var(--text)'>
                    🔁 <strong>Returning patient</strong> — {len(_prior_visits)} prior
                    visit{'s' if len(_prior_visits)!=1 else ''} on record.<br>
                    <span style='color:var(--muted)'>Last visit: {_last_date} ·
                    {_last['predicted_disease']} ({_last['confidence']}%)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption(f"🆕 New patient — no prior visits found for "
                           f"'{patient_name.strip()}' ({species}).")

        vet_notes    = st.text_area("Veterinarian notes (optional)",
                                     placeholder="Prior conditions, medications, relevant history...",
                                     height=60, key=f"vet_notes_input_{_fc}")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Symptom completeness nudge ────────────────────────────────────────
        _ticked = sum(1 for k in symptom_labels if st.session_state.get(f"symptom_cb_{k}_{_fc}", False))
        _text_len = len(primary_symptom.strip()) if primary_symptom else 0
        if _text_len > 0 and _ticked == 0:
            st.info("💡 **Tip:** You have entered a symptom description but no indicator "
                    "flags are ticked. Ticking confirmed symptoms (e.g. Vomiting, Appetite "
                    "Loss) gives the model structured clinical signals alongside your text "
                    "and can improve prediction quality.")

        predict_btn = st.button("🔍 Run Disease Prediction", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    with col_res:
        if predict_btn:
            if not patient_name or not patient_name.strip():
                st.warning("⚠️ No patient name entered — the prediction will run but the record "
                           "will be saved without a name. Consider entering a name first.")
            valid, val_msg = validate_symptom(primary_symptom)
            vague_warning = (valid and "could not recognise specific clinical terms" in val_msg)

            if not primary_symptom.strip():
                st.markdown('<div class="val-error">⚠️ Please enter a primary presenting complaint.</div>',
                            unsafe_allow_html=True)
            elif not valid:
                st.markdown(f'<div class="val-error">⚠️ {val_msg}</div>', unsafe_allow_html=True)
                st.markdown("""
                **Examples of valid descriptions:**
                - *"Has not eaten for 2 days, very lethargic and weak"*
                - *"Persistent cough with nasal discharge since Monday"*
                - *"Sudden lameness on left front leg after exercise"*
                - *"Swollen abdomen and laboured breathing"*
                - *"Found on roadside, limping, very thin"*
                """)
            elif not models_loaded:
                st.error("Models not loaded. Run Notebooks 01–03 first.")
            else:
                # Overlap warnings (non-blocking)
                overlaps = overlap_warnings(primary_symptom, additional_symptoms, symptom_flags)
                if overlaps:
                    with st.expander("ℹ️ Input note — click to see", expanded=False):
                        st.caption("These overlaps do not break the prediction but are noted:")
                        for o in overlaps[:3]: st.caption(f"• {o}")

                # Show the vague‑language warning if applicable
                if vague_warning:
                    st.markdown(f'<div class="val-warn">⚠️ {val_msg}</div>', unsafe_allow_html=True)

                with st.spinner("Analysing clinical data..."):
                    present       = [lbl for key,lbl in symptom_labels.items() if symptom_flags[key]]
                    absent        = [lbl for key,lbl in symptom_labels.items() if not symptom_flags[key]]
                    symptom_count = sum(symptom_flags.values())

                    temp_risk, symptom_risk, critical_risk, dangerous_level = \
                        compute_risk(temperature, heart_rate, symptom_count, species)
                    danger_category = get_danger(dangerous_level)

                    age_str      = "unknown" if age_unknown else f"{age} years"
                    symptom_text = (
                        f"Species: {species}. Breed: {breed}. Age: {age_str}. "
                        f"Gender: {gender}. Weight: {weight} kg. "
                        f"Primary symptom: {primary_symptom}. "
                        f"Additional: {', '.join(additional_symptoms) if additional_symptoms else 'none'}. "
                        f"Duration: {duration}. "
                        f"Symptoms present: {', '.join(present) if present else 'none'}. "
                        f"Body temperature: {temperature} C. Heart rate: {heart_rate} bpm. "
                        f"Danger level: {danger_category}."
                    )

                    temp_dev_species, hr_dev_species = compute_vital_deviation(
                        species, temperature, heart_rate
                    )
                    gi_combo      = int(symptom_flags["Vomiting"]) * int(symptom_flags["Diarrhea"])
                    resp_combo    = int(symptom_flags["Coughing"]) * int(symptom_flags["Labored_Breathing"])
                    severity_x_age = dangerous_level * age

                    # ── 3 additional engineered features (added in NB03 update) ──────
                    # Temp_x_Symptoms: temperature deviation × symptom count
                    temp_x_symptoms = temp_dev_species * symptom_count
                    # Multi_System: 1 if both respiratory AND GI symptoms present
                    multi_system    = int((resp_combo > 0) and (gi_combo > 0))
                    # HR_x_Risk: heart rate deviation × dangerous level
                    hr_x_risk       = hr_dev_species * dangerous_level

                    struct_row = np.array([[
                        age, weight,
                        int(symptom_flags["Appetite_Loss"]),
                        int(symptom_flags["Vomiting"]),
                        int(symptom_flags["Diarrhea"]),
                        int(symptom_flags["Coughing"]),
                        int(symptom_flags["Labored_Breathing"]),
                        int(symptom_flags["Lameness"]),
                        int(symptom_flags["Skin_Lesions"]),
                        int(symptom_flags["Nasal_Discharge"]),
                        int(symptom_flags["Eye_Discharge"]),
                        temperature, heart_rate, symptom_count,
                        temp_risk, symptom_risk, critical_risk, dangerous_level,
                        temp_dev_species, hr_dev_species,
                        gi_combo, resp_combo, severity_x_age,
                        temp_x_symptoms, multi_system, hr_x_risk
                    ]])
                    struct_scaled = struct_scaler.transform(struct_row)
                    bert_emb = (extract_embedding(symptom_text, tokenizer, bert_model, bert_device)
                                if bert_loaded else np.zeros((1,768)))
                    sbg_enc  = (ord_enc.transform(pd.DataFrame([[species,breed,gender]],
                                                               columns=SPECIES_BREED_COLS))
                                if ord_enc is not None else np.zeros((1,3)))
                    features = np.hstack([bert_emb, struct_scaled, sbg_enc])

                    probs        = classifier.predict_proba(features)[0]
                    top10_idx    = np.argsort(probs)[-10:][::-1]
                    top10_names  = le.inverse_transform(top10_idx)
                    top10_probs  = probs[top10_idx]
                    predicted    = top10_names[0]
                    confidence   = top10_probs[0]
                    conf_pct     = confidence * 100

                # Calibrated confidence label
                if conf_pct >= 70:
                    conf_label, conf_label_color = "High confidence", "#34d399"
                elif conf_pct >= 30:
                    conf_label, conf_label_color = "Moderate — review differentials", "#fbbf24"
                else:
                    conf_label, conf_label_color = "Low — consider full differential list", "#f87171"

                # SHAP — per-prediction feature explanation
                shap_bars_html = ""
                try:
                    shap_explainer = load_shap_explainer()
                    if shap_explainer is not None:
                        import shap
                        shap_vals = shap_explainer.shap_values(features)
                        # XGBoost returns (samples, features, classes)
                        pred_class_idx = np.argmax(probs)
                        if shap_vals.ndim == 3:
                            sv_pred = shap_vals[0, :, pred_class_idx]
                        else:
                            sv_pred = shap_vals[0]
                        # Named structured + species features only (skip 768 BERT dims)
                        named_features = STRUCT_COLS_EXTENDED + SPECIES_BREED_COLS
                        n_bert = features.shape[1] - len(named_features)
                        named_shap = list(zip(named_features, sv_pred[n_bert:]))
                        top5_shap  = sorted(named_shap, key=lambda x: abs(x[1]), reverse=True)[:5]
                        max_abs    = max(abs(v) for _, v in top5_shap) or 1e-9
                        shap_bars_html = "<div style='margin:0.6rem 0'><p style='font-size:0.75rem;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin:0 0 0.4rem'>Key prediction drivers (SHAP)</p>"
                        for feat, val in top5_shap:
                            bar_w   = int(abs(val) / max_abs * 100)
                            bar_col = "#34d399" if val > 0 else "#f87171"
                            arrow   = "↑" if val > 0 else "↓"
                            shap_bars_html += (
                                f"<div style='display:flex;align-items:center;gap:6px;margin:3px 0'>"
                                f"<span style='font-size:0.73rem;color:var(--muted);width:140px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{feat}</span>"
                                f"<div style='flex:1;background:var(--border);border-radius:3px;height:6px'>"
                                f"<div style='width:{bar_w}%;height:6px;border-radius:3px;background:{bar_col}'></div></div>"
                                f"<span style='font-size:0.72rem;color:{bar_col};width:20px;text-align:right'>{arrow}</span>"
                                f"</div>"
                            )
                        shap_bars_html += "</div>"
                except Exception:
                    shap_bars_html = ""

                # Disease checks
                pred_zoonotic   = is_disease_zoonotic(predicted)
                pred_contagious = is_disease_contagious(predicted)
                kb_entry        = disease_kb.get(predicted.strip(), {})

                # ── Primary result card ───────────────────────────────────────
                conf_color = ("#34d399" if conf_pct>=60 else
                              "#fbbf24" if conf_pct>=30 else "#f87171")

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                if patient_name:
                    age_disp = "Age unknown" if age_unknown else f"{age} yrs"
                    st.markdown(f"<p style='color:var(--muted);font-size:0.78rem;margin:0'>"
                                f"Patient: <strong>{patient_name}</strong> · "
                                f"{species}{' · '+breed if breed!='Unknown / Mixed' else ''} · "
                                f"{age_disp}</p>", unsafe_allow_html=True)

                st.markdown(f'<p class="disease-name">{predicted.strip()}</p>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display:flex;gap:0.8rem;align-items:center;
                     margin-bottom:0.6rem;flex-wrap:wrap">
                    <span style="font-size:0.95rem;font-weight:500;color:{conf_color}">
                        {conf_pct:.1f}% confidence
                    </span>
                    <span style="font-size:0.78rem;color:{conf_label_color};
                          background:rgba(0,0,0,0.2);padding:2px 8px;border-radius:4px">
                        {conf_label}
                    </span>
                    {danger_badge(danger_category)}
                    {"<span class='badge' style='background:rgba(239,68,68,.12);color:#fca5a5;border:1px solid rgba(239,68,68,.3)'>🦠 Zoonotic</span>" if pred_zoonotic else ""}
                    {"<span class='badge' style='background:rgba(251,191,36,.1);color:#fde68a;border:1px solid rgba(251,191,36,.25)'>🚧 Contagious</span>" if pred_contagious else ""}
                </div>""", unsafe_allow_html=True)

                desc = kb_entry.get('description')
                if desc and desc != 'Not_Recorded':
                    st.markdown(f"<p style='color:var(--muted);font-size:0.82rem;"
                                f"line-height:1.5;margin:0 0 0.5rem'>{desc}</p>",
                                unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Zoonotic / contagion alert banners ────────────────────────
                if pred_zoonotic:
                    st.markdown("""
                    <div class="alert-zoonotic">
                    🦠 <strong>ZOONOTIC DISEASE ALERT</strong> — This condition can potentially
                    transmit to humans. Advise the owner to practise strict hygiene, wear gloves
                    when handling the animal, and consult a physician if they develop symptoms.
                    Report to relevant public health authorities if required by local regulations.
                    </div>""", unsafe_allow_html=True)

                if pred_contagious:
                    st.markdown("""
                    <div class="alert-contagion">
                    🚧 <strong>CONTAGION RISK</strong> — This condition may be transmissible to
                    other animals. Consider isolating the patient from other animals in the
                    household or practice until diagnosis is confirmed.
                    </div>""", unsafe_allow_html=True)

                # ── Vital sign metrics ────────────────────────────────────────
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-box"><div class="metric-val">{temperature}°C</div>
                        <div class="metric-lbl">Temperature</div></div>
                    <div class="metric-box"><div class="metric-val">{heart_rate}</div>
                        <div class="metric-lbl">Heart Rate</div></div>
                    <div class="metric-box"><div class="metric-val">{symptom_count}</div>
                        <div class="metric-lbl">Flags Ticked</div></div>
                    <div class="metric-box"><div class="metric-val">{dangerous_level:.0f}%</div>
                        <div class="metric-lbl">Risk Score</div></div>
                </div>""", unsafe_allow_html=True)

                # ── Species vital sign interpretation ─────────────────────────
                _vr = SPECIES_RANGES.get(species, {"temp":(37.5,40.0),"hr":(60,160)})
                _tlo, _thi = _vr.get("temp",(37.5,40.0))
                _hlo, _hhi = _vr.get("hr",(60,160))
                _temp_status = (
                    f"🔴 above normal ({_tlo}–{_thi}°C for {species})"
                    if temperature > _thi else
                    f"🔵 below normal ({_tlo}–{_thi}°C for {species})"
                    if temperature < _tlo else
                    f"🟢 within normal range ({_tlo}–{_thi}°C for {species})"
                )
                _hr_status = (
                    f"🔴 above normal ({_hlo}–{_hhi} bpm for {species})"
                    if heart_rate > _hhi else
                    f"🔵 below normal ({_hlo}–{_hhi} bpm for {species})"
                    if heart_rate < _hlo else
                    f"🟢 within normal range ({_hlo}–{_hhi} bpm for {species})"
                )
                st.markdown(f"""
                <div style='background:var(--surface2);border:1px solid var(--border);
                     border-radius:7px;padding:0.55rem 0.9rem;font-size:0.77rem;
                     color:var(--muted);margin-bottom:0.7rem;line-height:1.8'>
                    <strong style='color:var(--text)'>Species-normalised vitals ({species})</strong><br>
                    🌡️ Temperature {temperature}°C — {_temp_status}<br>
                    💓 Heart rate {heart_rate} bpm — {_hr_status}
                </div>""", unsafe_allow_html=True)

                # ── Abnormal vitals cross-reference ───────────────────────────
                # When a vital sign is flagged abnormal, show which top-5 candidates
                # are specifically associated with that presentation — helps the
                # practitioner prioritise the differential list clinically.
                _temp_abnormal = temperature > _thi or temperature < _tlo
                _hr_abnormal   = heart_rate > _hhi or heart_rate < _hlo
                if _temp_abnormal or _hr_abnormal:
                    _fever_assoc    = ["Bacterial Pneumonia","Leptospirosis","Septicemia",
                                       "Canine Parvovirus","Feline Panleukopenia",
                                       "Feline Upper Respiratory Infection",
                                       "Bovine Respiratory Disease","Foot and Mouth Disease",
                                       "Canine Distemper","Avian Influenza"]
                    _hypotherm_assoc= ["Shock","Hypothermia","Poisoning","Trauma",
                                       "Neonatal Illness"]
                    _tachy_assoc    = ["Septicemia","Canine Parvovirus","Bacterial Pneumonia",
                                       "Feline Herpesvirus","Gastroenteritis","Trauma","Pain"]
                    _brady_assoc    = ["Poisoning","Hypothermia","Cardiac Disease",
                                       "Neurological Disease","Hypothyroidism"]

                    _xref_lines = []
                    if temperature > _thi:
                        matched = [n for n in top10_names[:5]
                                   if any(a.lower() in n.lower() or n.lower() in a.lower()
                                          for a in _fever_assoc)]
                        if matched:
                            _xref_lines.append(
                                f"🌡️ <strong>Elevated temperature</strong> consistent "
                                f"with: {', '.join(matched[:3])}")
                    elif temperature < _tlo:
                        matched = [n for n in top10_names[:5]
                                   if any(a.lower() in n.lower() or n.lower() in a.lower()
                                          for a in _hypotherm_assoc)]
                        if matched:
                            _xref_lines.append(
                                f"🌡️ <strong>Low temperature</strong> may suggest: "
                                f"{', '.join(matched[:3])}")
                    if heart_rate > _hhi:
                        matched = [n for n in top10_names[:5]
                                   if any(a.lower() in n.lower() or n.lower() in a.lower()
                                          for a in _tachy_assoc)]
                        if matched:
                            _xref_lines.append(
                                f"💓 <strong>Tachycardia</strong> consistent with: "
                                f"{', '.join(matched[:3])}")
                    elif heart_rate < _hlo:
                        matched = [n for n in top10_names[:5]
                                   if any(a.lower() in n.lower() or n.lower() in a.lower()
                                          for a in _brady_assoc)]
                        if matched:
                            _xref_lines.append(
                                f"💓 <strong>Bradycardia</strong> may suggest: "
                                f"{', '.join(matched[:3])}")

                    if _xref_lines:
                        _xref_body = "<br>".join(_xref_lines)
                        st.markdown(f"""
                        <div style='background:rgba(91,156,246,0.08);
                             border:1px solid rgba(91,156,246,0.3);
                             border-radius:7px;padding:8px 12px;margin:4px 0;
                             font-size:0.8rem;color:var(--muted)'>
                        📌 <strong style='color:#5b9cf6'>Vital sign correlation:</strong><br>
                        {_xref_body}
                        </div>""", unsafe_allow_html=True)

                # ── Clickable differential diagnoses ──────────────────────────
                st.markdown("**Top 10 Differential Diagnoses**")
                proba_sorted = np.sort(probs)[::-1]
                top3_sum  = float(proba_sorted[:3].sum())  * 100
                top5_sum  = float(proba_sorted[:5].sum())  * 100
                top10_sum = float(proba_sorted[:10].sum()) * 100
                # Probability mass concentration bar
                top1_sum  = float(proba_sorted[:1].sum())  * 100
                _bar_color = "#34d399" if top5_sum >= 90 else "#fbbf24" if top5_sum >= 70 else "#f87171"
                st.markdown(f"""
                <div style='margin:0.3rem 0 0.5rem'>
                  <div style='display:flex;justify-content:space-between;
                              font-size:0.72rem;color:var(--muted);margin-bottom:3px'>
                    <span>Probability mass concentration</span>
                    <span style='color:{_bar_color}'>{top5_sum:.0f}% in Top-5</span>
                  </div>
                  <div style='background:var(--surface2);border-radius:4px;height:8px;
                              overflow:hidden;border:1px solid var(--border)'>
                    <div style='height:100%;width:{top10_sum:.0f}%;background:#2e3850;border-radius:4px;
                                position:relative'>
                      <div style='height:100%;width:{(top5_sum/top10_sum*100) if top10_sum>0 else 0:.0f}%;
                                  background:{_bar_color};opacity:0.7;border-radius:4px'></div>
                    </div>
                  </div>
                  <div style='display:flex;gap:8px;margin-top:5px;flex-wrap:wrap'>
                    <span style='background:var(--surface2);border:1px solid var(--border);
                          border-radius:6px;padding:3px 10px;font-size:0.76rem;color:var(--muted)'>
                        Top-1 &nbsp;<strong style='color:var(--text)'>{top1_sum:.0f}%</strong>
                    </span>
                    <span style='background:var(--surface2);border:1px solid var(--border);
                          border-radius:6px;padding:3px 10px;font-size:0.76rem;color:var(--muted)'>
                        Top-3 &nbsp;<strong style='color:var(--text)'>{top3_sum:.0f}%</strong>
                    </span>
                    <span style='background:var(--surface2);border:1px solid {_bar_color};
                          border-radius:6px;padding:3px 10px;font-size:0.76rem;color:var(--muted)'>
                        Top-5 &nbsp;<strong style='color:{_bar_color}'>{top5_sum:.0f}%</strong>
                    </span>
                    <span style='background:var(--surface2);border:1px solid var(--border);
                          border-radius:6px;padding:3px 10px;font-size:0.76rem;color:var(--muted)'>
                        Top-10 &nbsp;<strong style='color:var(--text)'>{top10_sum:.0f}%</strong>
                    </span>
                    <span style='font-size:0.74rem;color:var(--muted);padding:3px 0;
                          align-self:center'>of probability mass</span>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.caption("Click any disease to see description and treatment. "
                           "High Top-5 concentration = model is diagnostically decisive. "
                           "Low concentration = high ambiguity, expand workup.")

                # ── Clinical uncertainty banner ───────────────────────────────
                # Case A: Low Top-5 concentration — genuinely ambiguous presentation
                if top5_sum < 50:
                    st.markdown(f"""
                    <div style='background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.5);
                         border-radius:8px;padding:10px 14px;margin:6px 0;font-size:0.83rem'>
                    ⚠️ <strong style='color:#fbbf24'>High diagnostic ambiguity</strong> —
                    the top-5 candidates account for only <strong style='color:#fbbf24'>
                    {top5_sum:.0f}%</strong> of probability mass.
                    The presenting signs are consistent with many conditions.
                    <strong>Broaden the clinical workup</strong> — consider haematology,
                    biochemistry, or imaging before committing to a diagnosis.
                    </div>""", unsafe_allow_html=True)

                # Case B: Critical risk but low Top-1 confidence — do not be reassured
                elif danger_category in ("Critical", "High") and confidence < 0.25:
                    st.markdown(f"""
                    <div style='background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.5);
                         border-radius:8px;padding:10px 14px;margin:6px 0;font-size:0.83rem'>
                    🚨 <strong style='color:#ef4444'>Urgent attention required despite diagnostic uncertainty</strong> —
                    risk score is <strong style='color:#ef4444'>{danger_category}</strong> but top-1 confidence
                    is only <strong style='color:#ef4444'>{confidence*100:.0f}%</strong>.
                    Do not delay clinical intervention while awaiting diagnostic clarity.
                    Stabilise the patient and run differential workup in parallel.
                    </div>""", unsafe_allow_html=True)

                for rank,(name,prob) in enumerate(zip(top10_names, top10_probs), 1):
                    pct      = prob * 100
                    bar_w    = max(1, min(100, pct))
                    kb_d     = disease_kb.get(name.strip(), {})
                    d_zoo    = is_disease_zoonotic(name)
                    d_cont   = is_disease_contagious(name)
                    zoo_tag  = " 🦠" if d_zoo   else ""
                    cont_tag = " 🚧" if d_cont  else ""
                    expanded = (rank == 1)

                    with st.expander(
                        f"#{rank}  {name.strip()}{zoo_tag}{cont_tag}  —  {pct:.1f}%",
                        expanded=expanded
                    ):
                        # Probability bar
                        st.markdown(f"""
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{bar_w}%"></div>
                        </div>""", unsafe_allow_html=True)

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption(f"Confidence: **{pct:.1f}%**")
                            st.caption(f"Typical severity: **{kb_d.get('typical_danger','—')}**")
                            st.caption(f"Training cases: **{kb_d.get('n_cases',0):,}**")
                        with col_b:
                            if d_zoo:
                                st.caption("🦠 **Zoonotic** — can spread to humans")
                            if d_cont:
                                st.caption("🚧 **Contagious** — can spread to other animals")

                        # Disease description
                        d_desc = kb_d.get('description')
                        if d_desc and d_desc != 'Not_Recorded':
                            st.markdown(f"<p style='font-size:0.81rem;color:var(--muted);"
                                        f"line-height:1.5;margin:0.4rem 0'>{d_desc}</p>",
                                        unsafe_allow_html=True)

                        # Treatment for this specific disease
                        d_treat = kb_d.get('treatment')
                        if d_treat and str(d_treat) not in ('Not_Recorded','None','nan'):
                            cards = parse_treatments(d_treat)
                            if cards:
                                st.markdown("<p style='font-size:0.78rem;font-weight:500;"
                                           "color:var(--muted);margin:0.5rem 0 0.3rem;"
                                           "text-transform:uppercase;letter-spacing:0.05em'>"
                                           "Treatment approach:</p>",
                                            unsafe_allow_html=True)
                                t_html = '<div style="display:flex;flex-wrap:wrap;gap:5px">'
                                for c in cards[:6]:
                                    t_html += (f'<span style="background:var(--surface2);'
                                               f'border:1px solid var(--border);border-radius:5px;'
                                               f'padding:2px 9px;font-size:0.75rem;color:var(--text)">'
                                               f'{c["icon"]} {c["text"]}</span>')
                                t_html += '</div>'
                                st.markdown(t_html, unsafe_allow_html=True)
                        else:
                            st.caption("Treatment data not available for this condition.")

                # ── Primary treatment section ─────────────────────────────────
                st.markdown("<br>**Treatment for Primary Prediction**",
                            unsafe_allow_html=True)
                treat_html = render_treatment_cards(
                    kb_entry.get('treatment'), danger_category)
                st.markdown(treat_html, unsafe_allow_html=True)

                # ── Disease knowledge panel ───────────────────────────────────
                n_cases_total   = kb_entry.get('n_cases', 0)
                species_counts  = kb_entry.get('species_counts', {})\
if hasattr(kb_entry, 'get') else {}
                n_cases_species = species_counts.get(species, 0)

                if n_cases_total > 0:
                    if n_cases_species > 0:
                        cases_label = f"{n_cases_species:,} ({species})"
                        cases_sub   = f"{n_cases_total:,} total across all species"
                    else:
                        cases_label = f"{n_cases_total:,} (all species)"
                        cases_sub   = f"No {species}-specific records in dataset"
                    st.markdown(f"""
                    <div class="disease-panel">
                        <div class="disease-panel-title">Disease Profile</div>
                        <div class="disease-panel-row">
                            <span><strong>Typical severity:</strong> {kb_entry.get('typical_danger','—')}</span>
                            <span><strong>Dataset cases:</strong> {cases_label}</span>
                            <span style="font-size:0.78rem;color:var(--muted)">{cases_sub}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Data sparsity warning — clinically important for rare species
                    if n_cases_species == 0:
                        st.markdown(f"""
                        <div style='background:rgba(91,156,246,0.10);border:1px solid rgba(91,156,246,0.35);
                             border-radius:7px;padding:8px 12px;margin:4px 0;font-size:0.8rem'>
                        📊 <strong style='color:#5b9cf6'>Species data note:</strong>
                        No <strong>{species}</strong> records for <em>{predicted.strip()}</em>
                        were found in the training dataset. This prediction is extrapolated
                        from patterns learned across other species — treat with increased
                        clinical caution and validate through physical examination.
                        </div>""", unsafe_allow_html=True)
                    elif n_cases_species < 20:
                        st.markdown(f"""
                        <div style='background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
                             border-radius:7px;padding:8px 12px;margin:4px 0;font-size:0.8rem'>
                        📊 <strong style='color:#fbbf24'>Limited species data:</strong>
                        Only <strong>{n_cases_species}</strong> {species} records for this disease
                        exist in the training data. Prediction confidence may be lower than
                        indicated — consider specialist referral.
                        </div>""", unsafe_allow_html=True)

                # ── SHAP bar chart ───────────────────────────────────────────
                if shap_bars_html:
                    st.markdown(shap_bars_html, unsafe_allow_html=True)

                # ── Disclaimer ────────────────────────────────────────────────
                st.markdown("""
                <div class="disclaimer">
                ⚕️ <strong>Clinical Decision Support Only.</strong>
                Probabilistic estimates to assist, not replace, professional veterinary
                judgment. Always confirm with physical examination and laboratory tests.
                </div>""", unsafe_allow_html=True)

                # ── Download report ───────────────────────────────────────────
                report_txt = generate_report(
                    patient_name or f"Patient #{len(load_history())+1}",
                    species, breed, age, gender, weight,
                    age_unknown, temperature, heart_rate, symptom_count,
                    primary_symptom, predicted, conf_pct, danger_category,
                    dangerous_level, top10_names, top10_probs,
                    kb_entry.get('treatment'), vet_notes,
                    pred_zoonotic, pred_contagious,
                    shap_drivers=top5_shap if 'top5_shap' in dir() else None
                )
                fname = (f"vetdx_{(patient_name or 'report').replace(' ','_')}"
                         f"_{datetime.date.today()}.txt")
                st.download_button(
                    "📄 Download Case Report",
                    data=report_txt.encode('utf-8'),
                    file_name=fname,
                    mime="text/plain",
                    use_container_width=True
                )

                # ── Save to history — PREVIEW THEN CONFIRM ────────────────────
                history = load_history()
                _resolved_name = str(patient_name or f"Patient #{len(history)+1}")
                _pid = make_patient_id(patient_name, species, breed) if (patient_name and patient_name.strip()) else None
                _visit_num = (len(find_patient_history(history, patient_name, species, breed)) + 1
                              if _pid else 1)
                record = {
                    "id": int(len(history) + 1),
                    "patient_id": _pid,
                    "visit_number": int(_visit_num),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "patient_name": _resolved_name,
                    "species": str(species),
                    "breed": str(breed),
                    "age": "Unknown" if age_unknown else int(age),
                    "gender": str(gender),
                    "weight": float(weight),
                    "primary_symptom": str(primary_symptom),
                    "temperature": float(temperature),
                    "heart_rate": int(heart_rate),
                    "symptom_count": int(symptom_count),
                    "predicted_disease": str(predicted),
                    "confidence": float(round(float(conf_pct), 1)),
                    "danger_category": str(danger_category),
                    "dangerous_level": float(round(float(dangerous_level), 1)),
                    "is_zoonotic": bool(pred_zoonotic),
                    "is_contagious": bool(pred_contagious),
                    "top10": [
                        {
                            "disease": str(d.strip()),
                            "probability": float(round(float(p) * 100, 1))
                        }
                        for d, p in zip(top10_names, top10_probs)
                    ],
                    "treatment": str(kb_entry.get('treatment') or "Not recorded"),
                    "vet_notes": str(vet_notes),
                }
                # Store in session state — don't write to disk yet
                st.session_state['pending_record'] = record

                st.markdown("""
                <div class="preview-banner">
                📋 <strong>Review before saving</strong> — the prediction above has
                <em>not yet been saved</em> to the patient history. Check the results,
                adjust any inputs if needed and re-run, then click
                <strong>Save to History</strong> when you are satisfied.
                </div>""", unsafe_allow_html=True)

        # ── Save button — outside the predict_btn block so it persists ────────
        if st.session_state.get('pending_record'):
            pr = st.session_state['pending_record']
            st.markdown(f"""
            <div style='background:var(--surface2);border:1px solid var(--border);
                 border-radius:8px;padding:0.7rem 1rem;font-size:0.86rem;
                 color:var(--muted);margin:0.5rem 0 0.3rem;line-height:1.9'>
            <strong style='color:var(--text)'>Ready to save:</strong><br>
            🐾 <strong style='color:var(--text)'>{pr["patient_name"]}</strong>
            &nbsp;·&nbsp; {pr["species"]} &nbsp;·&nbsp; {pr["breed"]}<br>
            🔬 Prediction: <strong style='color:var(--accent)'>{pr["predicted_disease"]}</strong>
            &nbsp;·&nbsp; Confidence: {pr["confidence"]}%<br>
            🕐 {pr["timestamp"][:16].replace("T"," ")}
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="save-btn">', unsafe_allow_html=True)
            save_clicked = st.button("💾 Save to History", use_container_width=True,
                                     key="save_to_history_btn")
            st.markdown('</div>', unsafe_allow_html=True)

            if save_clicked:
                # Confirmation dialog via session state
                st.session_state['confirm_save'] = True

            if st.session_state.get('confirm_save'):
                st.warning(
                    f"⚠️ **Confirm save** — This will permanently add the prediction for "
                    f"**{pr['patient_name']}** ({pr['predicted_disease']}) "
                    f"to the history file. Are you sure?"
                )
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Yes, save it", use_container_width=True,
                                 key="confirm_yes"):
                        history = load_history()
                        pr['id'] = len(history) + 1
                        history.append(pr)
                        save_history(history)
                        st.session_state.pop('pending_record', None)
                        st.session_state.pop('confirm_save', None)
                        # ── Increment form counter → all widget keys change →
                        # Streamlit sees brand-new widgets and renders them empty.
                        # This is the only reliable way to clear text inputs,
                        # text areas, and checkboxes simultaneously in Streamlit.
                        st.session_state['form_counter'] = st.session_state.get('form_counter', 0) + 1
                        st.session_state['_form_cleared'] = True
                        if pr.get('visit_number', 1) > 1:
                            st.success(
                                f"✅ Saved — Record #{pr['id']} "
                                f"(Visit #{pr['visit_number']} for {pr['patient_name']}). "
                                f"Form has been reset for the next patient."
                            )
                        else:
                            st.success(f"✅ Saved — Record #{pr['id']}. Form has been reset for the next patient.")
                        st.rerun()
                with col_no:
                    if st.button("❌ No, go back", use_container_width=True,
                                 key="confirm_no"):
                        st.session_state.pop('confirm_save', None)
                        st.rerun()

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 2rem;margin-top:1rem">
                <p style="font-size:2.4rem;margin:0">🔬</p>
                <p style="font-family:'Playfair Display',serif;font-size:1.1rem;
                   margin:0.7rem 0 0.3rem">Ready to Analyse</p>
                <p style="color:var(--muted);font-size:0.83rem">
                Complete the patient form and click<br>
                <strong>Run Disease Prediction</strong></p>
            </div>
            <div class="card" style="margin-top:0.8rem">
                <p class="card-title">💡 How It Works</p>
                <div style="font-size:0.8rem;color:var(--muted);line-height:2">
                <span style="color:var(--accent)">①</span> Enter species, vitals &amp; symptoms<br>
                <span style="color:var(--accent)">②</span> BERT encodes symptom text → 768 features<br>
                <span style="color:var(--accent)">③</span> XGBoost fuses text + structured data<br>
                <span style="color:var(--accent)">④</span> Top-10 differentials ranked by probability<br>
                <span style="color:var(--accent)">⑤</span> SHAP explains key prediction drivers
                </div>
            </div>""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 2 — HISTORY
# ==============================================================================
elif page == "📋 History":
    st.markdown("## Prediction History")
    history = load_history()

    if not history:
        st.markdown('<div class="card" style="text-align:center;padding:2rem">'
                    '<p style="color:var(--muted)">No predictions yet.</p></div>',
                    unsafe_allow_html=True)
    else:
        view_mode = st.radio(
            "View", ["All Records", "Patient Case History"],
            horizontal=True, label_visibility="collapsed"
        )

        if view_mode == "Patient Case History":
            # ── Build list of distinct returning patients (visit_number >= 2 group) ──
            patients_by_id = {}
            for r in history:
                pid = r.get('patient_id')
                if not pid:
                    continue
                patients_by_id.setdefault(pid, []).append(r)

            returning_patients = {
                pid: sorted(recs, key=lambda x: x['timestamp'])
                for pid, recs in patients_by_id.items() if len(recs) >= 1
            }

            if not returning_patients:
                st.info("No named patients on record yet. Add a patient name when "
                        "running a prediction to start building case histories.")
            else:
                options = {
                    f"{recs[0].get('patient_name','—')} ({recs[0]['species']}) "
                    f"— {len(recs)} visit{'s' if len(recs)!=1 else ''}": pid
                    for pid, recs in sorted(
                        returning_patients.items(),
                        key=lambda kv: kv[1][-1]['timestamp'], reverse=True
                    )
                }
                chosen_label = st.selectbox("Select patient", list(options.keys()))
                chosen_pid   = options[chosen_label]
                visits       = returning_patients[chosen_pid]

                st.markdown(f"### 🐾 {visits[0].get('patient_name','—')} — "
                            f"{visits[0]['species']} ({visits[0].get('breed','—')})")
                st.caption(f"{len(visits)} visit{'s' if len(visits)!=1 else ''} on record · "
                           f"first seen {visits[0]['timestamp'][:10]} · "
                           f"last seen {visits[-1]['timestamp'][:10]}")

                # ── Visit-to-visit clinical trend (only when ≥2 visits) ───────
                if len(visits) >= 2:
                    _v_first = visits[0]
                    _v_last  = visits[-1]
                    _risk_order = {"Low":0,"Moderate":1,"High":2,"Critical":3}
                    _r0 = _risk_order.get(_v_first.get('danger_category','Moderate'), 1)
                    _r1 = _risk_order.get(_v_last.get('danger_category','Moderate'), 1)
                    _t0 = float(_v_first.get('temperature', 38.5))
                    _t1 = float(_v_last.get('temperature', 38.5))
                    _c0 = float(_v_first.get('confidence', 0))
                    _c1 = float(_v_last.get('confidence', 0))
                    _s0 = int(_v_first.get('symptom_count', 0))
                    _s1 = int(_v_last.get('symptom_count', 0))

                    def _trend(a, b, higher_is_worse=True):
                        if a == b: return "→ Unchanged", "#a0aab8"
                        improving = (b < a) if higher_is_worse else (b > a)
                        if improving: return f"↓ Improving", "#34d399"
                        return f"↑ Worsening", "#f87171"

                    _risk_trend, _risk_col   = _trend(_r0, _r1)
                    _temp_trend, _temp_col   = _trend(_t0, _t1)
                    _sympt_trend, _sympt_col = _trend(_s0, _s1)
                    _conf_trend, _conf_col   = _trend(_c0, _c1, higher_is_worse=False)

                    st.markdown(f"""
                    <div style='background:var(--surface2);border:1px solid var(--border);
                         border-radius:8px;padding:12px 16px;margin:8px 0'>
                      <div style='font-size:0.8rem;color:var(--muted);margin-bottom:8px;
                                  font-weight:600;text-transform:uppercase;letter-spacing:0.05em'>
                        📈 Clinical trend — Visit 1 → Visit {len(visits)}
                      </div>
                      <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px'>
                        <div style='text-align:center'>
                          <div style='font-size:0.75rem;color:var(--muted)'>Risk level</div>
                          <div style='font-size:1rem;font-weight:700;color:{_risk_col}'>{_risk_trend}</div>
                          <div style='font-size:0.72rem;color:var(--muted)'>{_v_first.get('danger_category','')} → {_v_last.get('danger_category','')}</div>
                        </div>
                        <div style='text-align:center'>
                          <div style='font-size:0.75rem;color:var(--muted)'>Temperature</div>
                          <div style='font-size:1rem;font-weight:700;color:{_temp_col}'>{_temp_trend}</div>
                          <div style='font-size:0.72rem;color:var(--muted)'>{_t0:.1f}°C → {_t1:.1f}°C</div>
                        </div>
                        <div style='text-align:center'>
                          <div style='font-size:0.75rem;color:var(--muted)'>Symptom flags</div>
                          <div style='font-size:1rem;font-weight:700;color:{_sympt_col}'>{_sympt_trend}</div>
                          <div style='font-size:0.72rem;color:var(--muted)'>{_s0} → {_s1} flags</div>
                        </div>
                        <div style='text-align:center'>
                          <div style='font-size:0.75rem;color:var(--muted)'>Diagnosis confidence</div>
                          <div style='font-size:1rem;font-weight:700;color:{_conf_col}'>{_conf_trend}</div>
                          <div style='font-size:0.72rem;color:var(--muted)'>{_c0:.0f}% → {_c1:.0f}%</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                for v in visits:
                    ts = v['timestamp'][:16].replace('T', ' ')
                    zoo_tag  = " 🦠" if v.get('is_zoonotic')  else ""
                    cont_tag = " 🚧" if v.get('is_contagious') else ""
                    with st.expander(
                        f"Visit #{v.get('visit_number', '?')} · {ts} · "
                        f"{v['predicted_disease'][:35]}{zoo_tag}{cont_tag}"
                    ):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            age_disp = "Age unknown" if v.get('age') == 'Unknown' else f"{v.get('age','')} yrs"
                            st.markdown(f"**{age_disp}** / {v.get('gender','')}\n\n"
                                        f"**Weight:** {v.get('weight','')} kg")
                        with c2:
                            st.markdown(f"**Predicted:** {v['predicted_disease']}\n\n"
                                        f"**Confidence:** {v['confidence']}%\n\n"
                                        f"**Risk:** {v['danger_category']}")
                        with c3:
                            st.markdown(f"**Temp:** {v['temperature']} °C\n\n"
                                        f"**HR:** {v['heart_rate']} bpm")
                        if v.get('primary_symptom'):
                            st.caption(f"📋 {v['primary_symptom']}")
                        if v.get('vet_notes'):
                            st.info(f"📝 {v['vet_notes']}")
                        if v.get('top10'):
                            st.markdown("**Top 5 differentials:**")
                            for d in v['top10'][:5]:
                                _z = " 🦠" if is_disease_zoonotic(d['disease']) else ""
                                _c = " 🚧" if is_disease_contagious(d['disease']) else ""
                                st.caption(f"• {d['disease']}{_z}{_c} — {d['probability']}%")

                        st.markdown("---")
                        # ── Per-visit action row: PDF download + Delete ────────
                        _v_id = v.get('id', '')
                        _v_name = v.get('patient_name', '—') or '—'
                        _safe_name = str(_v_name).replace(' ','_').replace('/','_')

                        # Build PDF for this visit
                        _v_treat = v.get('treatment', '')
                        _v_treat_display = ""
                        if _v_treat and str(_v_treat) not in ('Not recorded','Not_Recorded','None','nan'):
                            _v_treat_display = (f"<p style='color:#a0aab8;font-size:13px'>"
                                                f"<strong style='color:#f0f4ff'>Treatment:</strong>"
                                                f" {_v_treat}</p>")
                        _v_top5 = ""
                        if v.get('top10'):
                            for _i, _d in enumerate(v['top10'][:5], 1):
                                _v_top5 += (f"<tr style='background:{'#1c2131' if _i%2==0 else '#131720'}'>"
                                            f"<td style='padding:5px 10px;color:#f0f4ff'>#{_i}</td>"
                                            f"<td style='padding:5px 10px;color:#f0f4ff'>{_d['disease']}</td>"
                                            f"<td style='padding:5px 10px;color:#5b9cf6;text-align:right'>"
                                            f"{_d['probability']}%</td></tr>")
                        _v_zoo_alert = (
                            "<div style='background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.5);"
                            "border-radius:6px;padding:8px 12px;margin:8px 0;color:#ffb3b3;font-size:13px'>"
                            "🦠 Zoonotic disease — inform owner, apply biosafety protocols</div>"
                            if v.get('is_zoonotic') else "")
                        _v_cont_alert = (
                            "<div style='background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.4);"
                            "border-radius:6px;padding:8px 12px;margin:8px 0;color:#ffe99a;font-size:13px'>"
                            "🚧 Contagious — isolate animal, monitor in-contact animals</div>"
                            if v.get('is_contagious') else "")

                        _v_pdf = f"""<!DOCTYPE html>
<html><head><meta charset='UTF-8'>
<title>VetDx Case Report — {_v_name} Visit #{v.get('visit_number','?')}</title>
<style>
body{{margin:0;padding:28px 36px;background:#0c0f18;color:#f0f4ff;
     font-family:'Segoe UI',Arial,sans-serif;font-size:14px;line-height:1.7}}
h1{{font-size:22px;color:#5b9cf6;margin:0 0 4px}}
h2{{font-size:15px;color:#5b9cf6;margin:18px 0 6px;border-bottom:1px solid #2e3850;padding-bottom:4px}}
.header{{display:flex;justify-content:space-between;align-items:flex-start;
         border-bottom:2px solid #5b9cf6;padding-bottom:14px;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:10px 0}}
.cell{{background:#131720;border:1px solid #2e3850;border-radius:8px;padding:10px 14px}}
.cell-label{{font-size:11px;color:#a0aab8;text-transform:uppercase;letter-spacing:.05em}}
.cell-val{{font-size:16px;color:#f0f4ff;font-weight:600;margin-top:2px}}
table{{width:100%;border-collapse:collapse;margin:6px 0}}
th{{background:#1c2131;color:#a0aab8;font-size:12px;text-align:left;padding:7px 10px;
   text-transform:uppercase;letter-spacing:.04em}}
td{{border-bottom:1px solid #1c2131}}
.conf-bar-bg{{background:#2e3850;border-radius:3px;height:6px;margin:6px 0 2px}}
.conf-bar{{height:6px;border-radius:3px;background:linear-gradient(90deg,#5b9cf6,#8b5cf6)}}
.footer{{margin-top:28px;padding-top:12px;border-top:1px solid #2e3850;
         font-size:11px;color:#a0aab8;text-align:center}}
</style></head><body>
<div class='header'>
  <div>
    <h1>VetDx Clinical Case Report</h1>
    <div style='color:#a0aab8;font-size:13px'>
      Record #{_v_id} · Visit #{v.get('visit_number','?')} ·
      {ts} · VetDx FYP — APU 2026
    </div>
  </div>
  <div style='text-align:right'>
    <div style='font-size:20px;font-weight:700;color:#5b9cf6'>{v['confidence']}%</div>
    <div style='font-size:11px;color:#a0aab8'>Confidence</div>
  </div>
</div>
<h2>Patient information</h2>
<div class='grid'>
  <div class='cell'><div class='cell-label'>Patient name</div>
    <div class='cell-val'>{_v_name}</div></div>
  <div class='cell'><div class='cell-label'>Species / Breed</div>
    <div class='cell-val'>{v['species']}</div>
    <div style='font-size:12px;color:#a0aab8'>{v.get('breed','—')}</div></div>
  <div class='cell'><div class='cell-label'>Age / Gender / Weight</div>
    <div class='cell-val'>{v.get('age','—')} yrs · {v.get('gender','—')}</div>
    <div style='font-size:12px;color:#a0aab8'>{v.get('weight','—')} kg</div></div>
</div>
<h2>Primary diagnosis</h2>
<div style='background:#131720;border:1px solid #5b9cf6;border-radius:8px;padding:14px 18px;margin:6px 0'>
  <div style='font-size:20px;font-weight:700;color:#f0f4ff;margin-bottom:4px'>{v['predicted_disease']}</div>
  <div class='conf-bar-bg'><div class='conf-bar' style='width:{min(int(v["confidence"]),100)}%'></div></div>
  <div style='font-size:12px;color:#a0aab8'>Confidence: {v['confidence']}%</div>
</div>
<h2>Vital signs</h2>
<div class='grid'>
  <div class='cell'><div class='cell-label'>Temperature</div>
    <div class='cell-val'>{v['temperature']} °C</div></div>
  <div class='cell'><div class='cell-label'>Heart rate</div>
    <div class='cell-val'>{v['heart_rate']} bpm</div></div>
  <div class='cell'><div class='cell-label'>Risk level</div>
    <div class='cell-val'>{v['danger_category']}</div></div>
</div>
{_v_zoo_alert}{_v_cont_alert}
<h2>Top-5 differential diagnoses</h2>
<table><tr><th>#</th><th>Disease</th><th style='text-align:right'>Probability</th></tr>
{_v_top5}</table>
{_v_treat_display}
{"<h2>Clinical notes</h2><p style='color:#f0f4ff;background:#131720;border:1px solid #2e3850;border-radius:8px;padding:10px 14px'>" + str(v.get('vet_notes','')) + "</p>" if v.get('vet_notes') else ""}
<div class='footer'>⚕️ VetDx research prototype (FYP, APU 2026) — not a licensed medical device.
All predictions are probabilistic estimates to assist, not replace, qualified veterinary clinical judgment.</div>
</body></html>"""

                        _pa1, _pa2 = st.columns(2)
                        with _pa1:
                            st.download_button(
                                label="📄 Download case report (HTML/PDF)",
                                data=_v_pdf,
                                file_name=f"VetDx_{_safe_name}_Visit{v.get('visit_number','?')}_Record{_v_id}.html",
                                mime="text/html",
                                use_container_width=True,
                                key=f"pdf_visit_{_v_id}",
                                help="Open in browser → File → Print → Save as PDF"
                            )
                        with _pa2:
                            if st.button(
                                "🗑️ Delete this visit record",
                                use_container_width=True,
                                key=f"del_visit_btn_{_v_id}"
                            ):
                                st.session_state[f'confirm_del_visit_{_v_id}'] = True

                        if st.session_state.get(f'confirm_del_visit_{_v_id}'):
                            st.warning(
                                f"⚠️ **Confirm delete** — Permanently remove Visit #{v.get('visit_number','?')} "
                                f"({v['predicted_disease']} · {_v_name})? This cannot be undone.")
                            _dy, _dn = st.columns(2)
                            with _dy:
                                if st.button("✅ Yes, delete it", use_container_width=True,
                                             key=f"confirm_del_visit_yes_{_v_id}"):
                                    new_history = [r for r in history if r.get('id') != _v_id]
                                    for _i, _r in enumerate(new_history, 1):
                                        _r['id'] = _i
                                    save_history(new_history)
                                    st.session_state.pop(f'confirm_del_visit_{_v_id}', None)
                                    st.success(f"✅ Visit record deleted.")
                                    st.rerun()
                            with _dn:
                                if st.button("❌ No, keep it", use_container_width=True,
                                             key=f"confirm_del_visit_no_{_v_id}"):
                                    st.session_state.pop(f'confirm_del_visit_{_v_id}', None)
                                    st.rerun()

                st.markdown("---")
                st.caption("Showing this patient's full case history only. "
                           "Switch to 'All Records' to search across every patient.")

        else:
            c1,c2,c3 = st.columns([2,1,1])
            _prefill_search = st.session_state.pop('history_search', '')
            with c1: search   = st.text_input("Search", placeholder="Patient name or disease...",
                                               value=_prefill_search)
            with c2: sp_filt  = st.selectbox("Species",["All"]+list(BREEDS_BY_SPECIES.keys()))
            with c3: sort_ord = st.selectbox("Sort",["Newest first","Oldest first"])

            filtered = [r for r in history
                        if (not search or search.lower() in r.get('patient_name','').lower()
                            or search.lower() in r.get('predicted_disease','').lower())
                        and (sp_filt=="All" or r.get('species')==sp_filt)]
            if sort_ord=="Newest first": filtered=filtered[::-1]
            st.caption(f"Showing {len(filtered)} of {len(history)} records")

            # ── Action bar — Export CSV + Delete All ──────────────────────────
            act1, act2 = st.columns(2)
            with act1:
                if st.button("📥 Export all records as CSV", use_container_width=True):
                    df_h = pd.DataFrame([{
                        'ID':r['id'],'Date':r['timestamp'][:10],
                        'Patient':r.get('patient_name',''),'Species':r['species'],
                        'Breed':r.get('breed',''),'Age':r.get('age',''),
                        'Gender':r.get('gender',''),'Predicted Disease':r['predicted_disease'],
                        'Confidence (%)':r['confidence'],'Risk Level':r['danger_category'],
                        'Temperature':r['temperature'],'Heart Rate':r['heart_rate'],
                        'Zoonotic':r.get('is_zoonotic',False),
                        'Contagious':r.get('is_contagious',False),
                        'Primary Symptom':r.get('primary_symptom',''),
                        'Treatment':r.get('treatment',''),'Vet Notes':r.get('vet_notes',''),
                    } for r in history])
                    st.download_button(
                        "⬇️ Download CSV",
                        df_h.to_csv(index=False),
                        f"vetdx_{datetime.date.today()}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            with act2:
                if st.button("🗑️ Delete entire history",
                             use_container_width=True,
                             key="delete_all_btn"):
                    st.session_state['confirm_delete_all'] = True

            # Delete-all confirmation dialog
            if st.session_state.get('confirm_delete_all'):
                st.error(
                    f"⚠️ **Confirm permanent deletion** — This will erase all "
                    f"**{len(history)} record(s)** from the history file and cannot "
                    f"be undone. Are you absolutely sure?"
                )
                dyes, dno = st.columns(2)
                with dyes:
                    if st.button("🗑️ Yes, delete everything",
                                 use_container_width=True,
                                 key="confirm_delete_all_yes"):
                        save_history([])
                        st.session_state.pop('confirm_delete_all', None)
                        st.session_state.pop('pending_record', None)
                        st.success("✅ History cleared.")
                        st.rerun()
                with dno:
                    if st.button("❌ No, keep my records",
                                 use_container_width=True,
                                 key="confirm_delete_all_no"):
                        st.session_state.pop('confirm_delete_all', None)
                        st.rerun()

            st.markdown("---")

            for rec in filtered:
                ts       = rec['timestamp'][:16].replace('T',' ')
                age_disp = "Age unknown" if rec.get('age')=='Unknown' else f"{rec.get('age','')} yrs"
                zoo_tag  = " 🦠" if rec.get('is_zoonotic')  else ""
                cont_tag = " 🚧" if rec.get('is_contagious') else ""
                visit_tag = (f" · 🔁 Visit #{rec['visit_number']}"
                             if rec.get('visit_number', 1) > 1 else "")
                with st.expander(
                    f"#{rec['id']} · {rec.get('patient_name','—')} · "
                    f"{rec['species']} · {rec['predicted_disease'][:35]}"
                    f"{zoo_tag}{cont_tag}{visit_tag} · {ts}"
                ):
                    c1,c2,c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Patient:** {rec.get('patient_name','—')}\n\n"
                                    f"**Species:** {rec['species']} ({rec.get('breed','')})\n\n"
                                    f"**{age_disp}** / {rec.get('gender','')}\n\n"
                                    f"**Weight:** {rec.get('weight','')} kg")
                    with c2:
                        st.markdown(f"**Predicted:** {rec['predicted_disease']}\n\n"
                                    f"**Confidence:** {rec['confidence']}%\n\n"
                                    f"**Risk:** {rec['danger_category']}\n\n"
                                    f"**Risk Score:** {rec.get('dangerous_level','')}%")
                    with c3:
                        st.markdown(f"**Temperature:** {rec['temperature']} °C\n\n"
                                    f"**Heart Rate:** {rec['heart_rate']} bpm\n\n"
                                    f"**Symptom flags:** {rec['symptom_count']}")
                    if rec.get('is_zoonotic'):
                        st.warning("🦠 Zoonotic disease — potential human transmission risk recorded")
                    if rec.get('is_contagious'):
                        st.info("🚧 Contagious disease — isolation may have been advised")
                    treat_str = rec.get('treatment')
                    if treat_str and str(treat_str) not in ('Not recorded','Not_Recorded','None','nan'):
                        cards = parse_treatments(treat_str)
                        if cards:
                            t_html = '<div style="display:flex;flex-wrap:wrap;gap:5px;margin:6px 0">'
                            for c in cards:
                                t_html += (f'<span style="background:var(--surface2);border:1px solid'
                                           f' var(--border);border-radius:5px;padding:2px 9px;'
                                           f'font-size:0.76rem;color:var(--text)">'
                                           f'{c["icon"]} {c["text"]}</span>')
                            t_html += '</div>'
                            st.markdown("**Treatment:**")
                            st.markdown(t_html, unsafe_allow_html=True)
                    if rec.get('primary_symptom'): st.caption(f"📋 Complaint: {rec['primary_symptom']}")
                    if rec.get('vet_notes'):       st.info(f"📝 {rec['vet_notes']}")
                    if rec.get('top10'):
                        st.markdown("**Top 5 differentials:**")
                        for d in rec['top10'][:5]:
                            zoo = " 🦠" if is_disease_zoonotic(d['disease'])   else ""
                            con = " 🚧" if is_disease_contagious(d['disease']) else ""
                            st.caption(f"• {d['disease']}{zoo}{con} — {d['probability']}%")

                    st.markdown("---")

                    # ── Per-record action row: Delete + PDF download ───────────
                    pa1, pa2 = st.columns(2)

                    # PDF download — build as HTML then offer as downloadable file
                    with pa1:
                        top5_rows = ""
                        if rec.get('top10'):
                            for i, d in enumerate(rec['top10'][:5], 1):
                                zoo_p = " 🦠" if is_disease_zoonotic(d['disease'])   else ""
                                con_p = " 🚧" if is_disease_contagious(d['disease']) else ""
                                top5_rows += (
                                    f"<tr style='background:{'#1c2131' if i%2==0 else '#131720'}'>"
                                    f"<td style='padding:5px 10px;color:#f0f4ff'>#{i}</td>"
                                    f"<td style='padding:5px 10px;color:#f0f4ff'>{d['disease']}{zoo_p}{con_p}</td>"
                                    f"<td style='padding:5px 10px;color:#5b9cf6;text-align:right'>{d['probability']}%</td>"
                                    f"</tr>"
                                )
                        zoonotic_alert = (
                            "<div style='background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.5);"
                            "border-radius:6px;padding:8px 12px;margin:8px 0;color:#ffb3b3;font-size:13px'>"
                            "🦠 Zoonotic disease — inform owner, apply biosafety protocols</div>"
                            if rec.get('is_zoonotic') else ""
                        )
                        contagious_alert = (
                            "<div style='background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.4);"
                            "border-radius:6px;padding:8px 12px;margin:8px 0;color:#ffe99a;font-size:13px'>"
                            "🚧 Contagious — isolate animal, monitor in-contact animals</div>"
                            if rec.get('is_contagious') else ""
                        )
                        treat_display = ""
                        if treat_str and str(treat_str) not in ('Not recorded','Not_Recorded','None','nan'):
                            treat_display = f"<p style='color:#a0aab8;font-size:13px'><strong style='color:#f0f4ff'>Treatment:</strong> {treat_str}</p>"

                        pdf_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='UTF-8'>
<title>VetDx Case Report — {rec.get('patient_name','Patient')} #{rec['id']}</title>
<style>
  body{{margin:0;padding:28px 36px;background:#0c0f18;color:#f0f4ff;
        font-family:'Segoe UI',Arial,sans-serif;font-size:14px;line-height:1.7}}
  h1{{font-size:22px;color:#5b9cf6;margin:0 0 4px}}
  h2{{font-size:15px;color:#5b9cf6;margin:18px 0 6px;border-bottom:1px solid #2e3850;padding-bottom:4px}}
  .header{{display:flex;justify-content:space-between;align-items:flex-start;
            border-bottom:2px solid #5b9cf6;padding-bottom:14px;margin-bottom:18px}}
  .badge{{display:inline-block;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600}}
  .badge-critical{{background:rgba(239,68,68,.2);color:#ffb3b3;border:1px solid rgba(239,68,68,.5)}}
  .badge-high{{background:rgba(251,191,36,.15);color:#ffe99a;border:1px solid rgba(251,191,36,.4)}}
  .badge-moderate{{background:rgba(91,156,246,.15);color:#b8d4ff;border:1px solid rgba(91,156,246,.4)}}
  .badge-low{{background:rgba(52,211,153,.15);color:#9af0d0;border:1px solid rgba(52,211,153,.4)}}
  .grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:10px 0}}
  .cell{{background:#131720;border:1px solid #2e3850;border-radius:8px;padding:10px 14px}}
  .cell-label{{font-size:11px;color:#a0aab8;text-transform:uppercase;letter-spacing:.05em}}
  .cell-val{{font-size:16px;color:#f0f4ff;font-weight:600;margin-top:2px}}
  table{{width:100%;border-collapse:collapse;margin:6px 0}}
  th{{background:#1c2131;color:#a0aab8;font-size:12px;text-align:left;padding:7px 10px;
      text-transform:uppercase;letter-spacing:.04em}}
  td{{border-bottom:1px solid #1c2131}}
  .footer{{margin-top:28px;padding-top:12px;border-top:1px solid #2e3850;
            font-size:11px;color:#a0aab8;text-align:center}}
  .conf-bar-bg{{background:#2e3850;border-radius:3px;height:6px;margin:6px 0 2px}}
  .conf-bar{{height:6px;border-radius:3px;background:linear-gradient(90deg,#5b9cf6,#8b5cf6)}}
</style>
</head>
<body>
<div class='header'>
  <div>
    <h1>VetDx Clinical Case Report</h1>
    <div style='color:#a0aab8;font-size:13px'>
      Record #{rec['id']} &nbsp;·&nbsp; Generated {rec['timestamp'][:16].replace('T',' ')}
      &nbsp;·&nbsp; VetDx FYP — APU 2026
    </div>
  </div>
  <div style='text-align:right'>
    <div style='font-size:20px;font-weight:700;color:#5b9cf6'>{rec['confidence']}%</div>
    <div style='font-size:11px;color:#a0aab8'>Confidence</div>
    <span class='badge badge-{rec['danger_category'].lower()}'>{rec['danger_category']} Risk</span>
  </div>
</div>

<h2>Patient information</h2>
<div class='grid'>
  <div class='cell'>
    <div class='cell-label'>Patient name</div>
    <div class='cell-val'>{rec.get('patient_name','—')}</div>
  </div>
  <div class='cell'>
    <div class='cell-label'>Species / Breed</div>
    <div class='cell-val'>{rec['species']}</div>
    <div style='font-size:12px;color:#a0aab8'>{rec.get('breed','—')}</div>
  </div>
  <div class='cell'>
    <div class='cell-label'>Age / Gender / Weight</div>
    <div class='cell-val'>{rec.get('age','—')} yrs · {rec.get('gender','—')}</div>
    <div style='font-size:12px;color:#a0aab8'>{rec.get('weight','—')} kg</div>
  </div>
</div>

<h2>Primary diagnosis</h2>
<div style='background:#131720;border:1px solid #5b9cf6;border-radius:8px;padding:14px 18px;margin:6px 0'>
  <div style='font-size:20px;font-weight:700;color:#f0f4ff;margin-bottom:4px'>{rec['predicted_disease']}</div>
  <div class='conf-bar-bg'><div class='conf-bar' style='width:{min(rec["confidence"],100)}%'></div></div>
  <div style='font-size:12px;color:#a0aab8'>Model confidence: {rec['confidence']}%</div>
</div>

<h2>Vital signs</h2>
<div class='grid'>
  <div class='cell'>
    <div class='cell-label'>Body temperature</div>
    <div class='cell-val'>{rec['temperature']} °C</div>
  </div>
  <div class='cell'>
    <div class='cell-label'>Heart rate</div>
    <div class='cell-val'>{rec['heart_rate']} bpm</div>
  </div>
  <div class='cell'>
    <div class='cell-label'>Symptom flags</div>
    <div class='cell-val'>{rec['symptom_count']} checked</div>
  </div>
</div>

{zoonotic_alert}
{contagious_alert}

<h2>Top-5 differential diagnoses</h2>
<table>
  <tr><th>#</th><th>Disease</th><th style='text-align:right'>Probability</th></tr>
  {top5_rows}
</table>

{treat_display}

{'<h2>Clinical notes</h2><p style="color:#f0f4ff;background:#131720;border:1px solid #2e3850;border-radius:8px;padding:10px 14px">' + str(rec.get('vet_notes','')) + '</p>' if rec.get('vet_notes') else ''}

{'<h2>Presenting complaint</h2><p style="color:#a0aab8;font-size:13px">' + str(rec.get('primary_symptom','')) + '</p>' if rec.get('primary_symptom') else ''}

<div class='footer'>
  ⚕️ VetDx is a research prototype (FYP, APU 2026) — not a licensed medical device.
  All predictions are probabilistic estimates to assist, not replace, qualified veterinary clinical judgment.
</div>
</body>
</html>"""

                        safe_name = (rec.get('patient_name','patient')
                                     .replace(' ','_').replace('/','_'))
                        pdf_filename = f"VetDx_Report_{safe_name}_Record{rec['id']}.html"
                        st.download_button(
                            label="📄 Download case report (HTML/PDF)",
                            data=pdf_html,
                            file_name=pdf_filename,
                            mime="text/html",
                            use_container_width=True,
                            key=f"pdf_{rec['id']}",
                            help="Save then open in browser → File → Print → Save as PDF"
                        )

                    # Delete single record
                    with pa2:
                        if st.button(
                            f"🗑️ Delete this record",
                            use_container_width=True,
                            key=f"del_btn_{rec['id']}"
                        ):
                            st.session_state[f'confirm_del_{rec["id"]}'] = True

                    if st.session_state.get(f'confirm_del_{rec["id"]}'):
                        st.warning(
                            f"⚠️ **Confirm delete** — Permanently remove Record #{rec['id']} "
                            f"({rec['predicted_disease']} · {rec.get('patient_name','—')})? "
                            f"This cannot be undone."
                        )
                        dy, dn = st.columns(2)
                        with dy:
                            if st.button(
                                "✅ Yes, delete it",
                                use_container_width=True,
                                key=f"confirm_del_yes_{rec['id']}"
                            ):
                                new_history = [r for r in history
                                               if r['id'] != rec['id']]
                                # Re-number IDs sequentially
                                for i, r in enumerate(new_history, 1):
                                    r['id'] = i
                                save_history(new_history)
                                st.session_state.pop(
                                    f'confirm_del_{rec["id"]}', None)
                                st.success(
                                    f"✅ Record #{rec['id']} deleted.")
                                st.rerun()
                        with dn:
                            if st.button(
                                "❌ No, keep it",
                                use_container_width=True,
                                key=f"confirm_del_no_{rec['id']}"
                            ):
                                st.session_state.pop(
                                    f'confirm_del_{rec["id"]}', None)
                                st.rerun()


# ==============================================================================
# PAGE 3 — ANALYTICS
# ==============================================================================
elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    # Reset unique button counter on each page render
    st.session_state['_analytics_btn_n'] = 0
    st.markdown(
        "<p style='color:var(--muted);font-size:0.88rem;margin-top:-0.4rem'>"
        "Comprehensive analysis of all predictions made in this session. "
        "Use the <strong style='color:var(--text)'>drill-down filters</strong> below to focus on a "
        "specific species or disease. All charts update instantly to reflect your selection.</p>",
        unsafe_allow_html=True
    )
    history = load_history()
    if len(history) < 2:
        st.info("Run at least 2 predictions to see analytics.")
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from collections import Counter

        # ── Build full analytics dataframe ─────────────────────────────────────
        df_all = pd.DataFrame([{
            'id':            r.get('id', ''),
            'patient_name':  r.get('patient_name', '—') or '—',
            'date':          r['timestamp'][:10],
            'timestamp':     r['timestamp'],
            'species':       r['species'],
            'breed':         r.get('breed', 'Unknown'),
            'gender':        r.get('gender', 'Unknown'),
            'age':           r.get('age', 0),
            'weight':        r.get('weight', 0),
            'disease':       r['predicted_disease'],
            'confidence':    float(r['confidence']),
            'danger':        r['danger_category'],
            'dangerous_level': float(r.get('dangerous_level', 0)),
            'temp':          float(r.get('temperature', 38.5)),
            'hr':            int(r.get('heart_rate', 80)),
            'symptom_count': int(r.get('symptom_count', 0)),
            'zoonotic':      bool(r.get('is_zoonotic', False)),
            'contagious':    bool(r.get('is_contagious', False)),
            'visit_number':  int(r.get('visit_number', 1)),
            'primary_symptom': r.get('primary_symptom', ''),
            'vet_notes':     r.get('vet_notes', ''),
        } for r in history])

        df_all['date'] = pd.to_datetime(df_all['date'])

        # ══════════════════════════════════════════════════════════════════
        # DRILL-DOWN FILTER BAR
        # ══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🔎 Drill-Down Filters")
        st.caption("Filter all charts below by species and/or disease. "
                   "Select 'All' to see the full session caseload.")

        dcol1, dcol2, dcol3 = st.columns([1, 1, 1])
        with dcol1:
            species_options = ["All species"] + sorted(df_all['species'].unique().tolist())
            selected_species = st.selectbox("🐾 Filter by Species",
                                            species_options, key="analytics_species")
        with dcol2:
            if selected_species != "All species":
                disease_pool = sorted(df_all[df_all['species'] == selected_species]['disease'].unique().tolist())
            else:
                disease_pool = sorted(df_all['disease'].unique().tolist())
            disease_options = ["All diseases"] + disease_pool
            selected_disease = st.selectbox("🦠 Filter by Disease",
                                            disease_options, key="analytics_disease")
        with dcol3:
            risk_options = ["All risk levels", "Critical", "High", "Moderate", "Low"]
            selected_risk = st.selectbox("⚠️ Filter by Risk Level",
                                         risk_options, key="analytics_risk")

        # Apply filters
        df_h = df_all.copy()
        if selected_species != "All species":
            df_h = df_h[df_h['species'] == selected_species]
        if selected_disease != "All diseases":
            df_h = df_h[df_h['disease'] == selected_disease]
        if selected_risk != "All risk levels":
            df_h = df_h[df_h['danger'] == selected_risk]

        # Show filter summary
        if selected_species != "All species" or selected_disease != "All diseases" or selected_risk != "All risk levels":
            active_filters = []
            if selected_species != "All species":   active_filters.append(f"Species = **{selected_species}**")
            if selected_disease != "All diseases":  active_filters.append(f"Disease = **{selected_disease}**")
            if selected_risk != "All risk levels":  active_filters.append(f"Risk = **{selected_risk}**")
            st.markdown(
                f"<div style='background:var(--surface2);border:1px solid var(--accent);"
                f"border-radius:7px;padding:0.55rem 0.9rem;font-size:0.86rem;"
                f"color:var(--muted);margin-bottom:0.5rem'>"
                f"📌 Active filters: {' · '.join(active_filters)} "
                f"— showing <strong style='color:var(--text)'>{len(df_h)}</strong> of "
                f"<strong style='color:var(--text)'>{len(df_all)}</strong> total records</div>",
                unsafe_allow_html=True
            )
            if len(df_h) < 2:
                st.warning("Not enough records match the selected filters. "
                           "Try broadening your selection.")
                st.stop()

        # ── Species drill-down detail panel ────────────────────────────────────
        if selected_species != "All species":
            st.markdown(f"### 🐾 {selected_species} — Detailed Breakdown")
            sp_df = df_h.copy()
            sd1, sd2, sd3, sd4 = st.columns(4)
            with sd1:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{len(sp_df)}</div>"
                    f"<div class='metric-lbl'>{selected_species} Cases</div></div>",
                    unsafe_allow_html=True)
            with sd2:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{sp_df['disease'].nunique()}</div>"
                    f"<div class='metric-lbl'>Unique Diseases</div></div>",
                    unsafe_allow_html=True)
            with sd3:
                avg_conf_sp = sp_df['confidence'].mean()
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{avg_conf_sp:.1f}%</div>"
                    f"<div class='metric-lbl'>Avg Confidence</div></div>",
                    unsafe_allow_html=True)
            with sd4:
                high_risk_sp = sp_df['danger'].isin(['High','Critical']).sum()
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{high_risk_sp}</div>"
                    f"<div class='metric-lbl'>High/Critical Risk</div></div>",
                    unsafe_allow_html=True)

            # Disease breakdown table for selected species
            st.markdown(f"**Disease breakdown for {selected_species}:**")
            sp_disease_df = (sp_df.groupby('disease')
                             .agg(Cases=('disease','count'),
                                  Avg_Confidence=('confidence','mean'),
                                  Avg_Risk=('dangerous_level','mean'),
                                  Zoonotic=('zoonotic','any'))
                             .sort_values('Cases', ascending=False)
                             .reset_index())
            sp_disease_df['Avg_Confidence'] = sp_disease_df['Avg_Confidence'].round(1).astype(str) + '%'
            sp_disease_df['Avg_Risk']        = sp_disease_df['Avg_Risk'].round(1).astype(str) + '%'
            sp_disease_df['Zoonotic']        = sp_disease_df['Zoonotic'].map({True:'🦠 Yes', False:'No'})
            sp_disease_df.columns           = ['Disease','Cases','Avg Confidence','Avg Risk Score','Zoonotic']
            st.dataframe(sp_disease_df, use_container_width=True, hide_index=True)

            # Breed breakdown if multiple breeds present
            if sp_df['breed'].nunique() > 1:
                st.markdown(f"**Breed distribution for {selected_species} cases:**")
                breed_counts = sp_df['breed'].value_counts().reset_index()
                breed_counts.columns = ['Breed', 'Cases']
                st.dataframe(breed_counts, use_container_width=True, hide_index=True)
            st.markdown("---")

        df_h['date'] = pd.to_datetime(df_h['date'])

        zoo_count  = df_h['zoonotic'].sum()
        cont_count = df_h['contagious'].sum()
        avg_conf   = df_h['confidence'].mean()
        high_risk  = df_h['danger'].isin(['High', 'Critical']).sum()
        returning  = (df_h['visit_number'] > 1).sum()

        # ── KPI row ───────────────────────────────────────────────────────────
        kpi_cols = st.columns(6)
        kpis = [
            (len(df_h),                "Filtered Cases"),
            (df_h['species'].nunique(), "Species"),
            (df_h['disease'].nunique(), "Unique Diseases"),
            (f"{avg_conf:.1f}%",        "Avg Confidence"),
            (high_risk,                 "High/Critical Risk"),
            (f"{zoo_count}+{cont_count}", "Zoo / Contagious"),
        ]
        for col, (val, lbl) in zip(kpi_cols, kpis):
            with col:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-val">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

        # ── Session risk alert banner ──────────────────────────────────────────
        critical_count = df_h['danger'].isin(['Critical']).sum()
        if critical_count > 0 and (zoo_count > 0 or cont_count > 0):
            st.markdown(
                f"<div class='alert-zoonotic' style='font-size:0.92rem;margin:0.6rem 0'>"
                f"⚠️ <strong>Clinical alert:</strong> This session contains "
                f"<strong>{critical_count} Critical-risk</strong> case(s) and "
                f"<strong>{zoo_count + cont_count} biosafety-flagged</strong> case(s) "
                f"requiring immediate attention. Review Sections 2 and 6 below.</div>",
                unsafe_allow_html=True)
        elif critical_count > 0:
            st.markdown(
                f"<div class='urgency-Critical' style='margin:0.6rem 0'>"
                f"⚠️ <strong>{critical_count} Critical-risk</strong> case(s) in this session "
                f"— review Section 2 for details.</div>",
                unsafe_allow_html=True)
        elif zoo_count > 0 or cont_count > 0:
            st.markdown(
                f"<div class='alert-contagion' style='margin:0.6rem 0'>"
                f"🦠 <strong>{zoo_count} zoonotic</strong> and "
                f"<strong>{cont_count} contagious</strong> case(s) flagged this session "
                f"— review Section 6 for biosafety guidance.</div>",
                unsafe_allow_html=True)

        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        PAL  = ['#5b9cf6','#8b5cf6','#34d399','#fbbf24','#f87171',
                '#fb923c','#a78bfa','#38bdf8','#4ade80','#e879f9']
        RISK_COLORS = {'Critical':'#ef4444','High':'#fbbf24',
                       'Moderate':'#5b9cf6','Low':'#34d399'}

        PLOTLY_LAYOUT = dict(
            paper_bgcolor='#131720', plot_bgcolor='#1a2030',
            font=dict(color='#f0f4ff', size=13),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor='#1a2030', bordercolor='#2e3850',
                        borderwidth=1, font=dict(size=12)),
            hoverlabel=dict(bgcolor='#1a2030', bordercolor='#5b9cf6',
                            font=dict(color='#f0f4ff', size=13)),
        )

        def apply_layout(fig, title='', height=380):
            fig.update_layout(**PLOTLY_LAYOUT, title=dict(
                text=title, font=dict(size=14, color='#f0f4ff'), x=0.01), height=height)
            fig.update_xaxes(gridcolor='#2e3850', zerolinecolor='#2e3850',
                             tickfont=dict(size=12))
            fig.update_yaxes(gridcolor='#2e3850', zerolinecolor='#2e3850',
                             tickfont=dict(size=12))
            return fig

        # Helper: build drill-down detail table for a clicked item
        def drill_detail(df_sub, label):
            """Return a styled dataframe for the drill-down expander."""
            cols = ['patient_name','species','breed','disease',
                    'confidence','danger','temp','hr','symptom_count',
                    'zoonotic','contagious']
            cols = [c for c in cols if c in df_sub.columns]
            return df_sub[cols].rename(columns={
                'patient_name':'Patient','species':'Species','breed':'Breed',
                'disease':'Disease','confidence':'Confidence %','danger':'Risk',
                'temp':'Temp °C','hr':'HR bpm','symptom_count':'Symptoms',
                'zoonotic':'Zoonotic','contagious':'Contagious'
            })

        # ══════════════════════════════════════════════════════════════════
        # SECTION 1 — DISEASE & SPECIES DISTRIBUTION  (Plotly — interactive)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🦠 Disease & Species Distribution")
        st.caption("Click any bar or slice to drill down into the individual cases behind that data point.")

        col_s1a, col_s1b = st.columns(2)

        # 1a — Top diseases horizontal bar (CLICKABLE)
        with col_s1a:
            top_d = df_h["disease"].value_counts().head(12).reset_index()
            top_d.columns = ["Disease", "Cases"]
            top_d["Hover"] = top_d.apply(
                lambda r: (f"<b>{r.Disease}</b><br>Cases: {r.Cases}<br>"
                           f"Avg confidence: {df_h[df_h['disease']==r.Disease]['confidence'].mean():.1f}%<br>"
                           f"Zoonotic: {'Yes 🦠' if df_h[df_h['disease']==r.Disease]['zoonotic'].any() else 'No'}"),
                axis=1)
            fig_d = px.bar(top_d, x="Cases", y="Disease", orientation="h",
                           color="Cases", color_continuous_scale=["#2e3850","#5b9cf6","#8b5cf6"],
                           hover_name="Disease", hover_data={"Cases":True,"Disease":False,"Hover":False},
                           custom_data=["Disease","Hover"])
            fig_d.update_traces(
                hovertemplate="%{customdata[1]}<extra></extra>",
                marker_line_width=0
            )
            fig_d.update_coloraxes(showscale=False)
            apply_layout(fig_d, "Top 12 Predicted Diseases — click a bar to drill down", height=400)
            fig_d.update_yaxes(categoryorder="total ascending")
            clicked_disease = st.plotly_chart(fig_d, use_container_width=True,
                                              on_select="rerun", key="chart_diseases",
                                              selection_mode="points")
            # Drill-down on click
            sel_pts = (clicked_disease.selection.get("points", [])
                       if clicked_disease and clicked_disease.selection else [])
            if sel_pts:
                clicked_name = sel_pts[0].get("customdata", [None])[0]
                if clicked_name:
                    sub = df_h[df_h["disease"] == clicked_name].copy().reset_index(drop=True)
                    with st.expander(f"📊 Drill-down: {clicked_name} — {len(sub)} case(s)", expanded=True):
                        st.markdown(f"""
                        <div style='display:flex;gap:20px;flex-wrap:wrap;margin-bottom:0.7rem'>
                            <span style='font-size:0.9rem;color:var(--muted)'>Cases: <strong style='color:var(--text)'>{len(sub)}</strong></span>
                            <span style='font-size:0.9rem;color:var(--muted)'>Avg confidence: <strong style='color:var(--accent)'>{sub["confidence"].mean():.1f}%</strong></span>
                            <span style='font-size:0.9rem;color:var(--muted)'>Avg risk score: <strong style='color:#fbbf24'>{sub["dangerous_level"].mean():.1f}</strong></span>
                            <span style='font-size:0.9rem;color:var(--muted)'>Species seen: <strong style='color:var(--text)'>{"、".join(sub["species"].unique())}</strong></span>
                            {"<span style='font-size:0.9rem;color:#ffb3b3'>🦠 Zoonotic</span>" if sub["zoonotic"].any() else ""}
                        </div>""", unsafe_allow_html=True)
                        st.caption("Expand a patient row below to see full clinical detail.")
                        for _, row in sub.iterrows():
                            _pn  = str(row.get("patient_name","—") or "—")
                            _zt  = " 🦠" if row.get("zoonotic")   else ""
                            _ct  = " 🚧" if row.get("contagious") else ""
                            with st.expander(
                                f"🐾 {_pn}  ·  {row.get('species','')} ({row.get('breed','')})  ·  "
                                f"{row.get('confidence','')}%  ·  {row.get('danger','')}{_zt}{_ct}",
                                expanded=False):
                                ca, cb, cc = st.columns(3)
                                with ca:
                                    st.markdown(f"**Patient:** {_pn}  \n**Species:** {row.get('species','')} ({row.get('breed','')})  \n**Age:** {row.get('age','')} yrs · {row.get('gender','')}  \n**Weight:** {row.get('weight','')} kg")
                                with cb:
                                    st.markdown(f"**Disease:** {row.get('disease','')}  \n**Confidence:** {row.get('confidence','')}%  \n**Risk:** {row.get('danger','')}  \n**Risk score:** {row.get('dangerous_level','')}%")
                                with cc:
                                    st.markdown(f"**Temp:** {row.get('temp','')} °C  \n**HR:** {row.get('hr','')} bpm  \n**Symptoms:** {row.get('symptom_count','')} flags  \n**Date:** {str(row.get('timestamp',''))[:10]}")
                                if row.get('zoonotic'):  st.warning("🦠 Zoonotic — advise owner hygiene")
                                if row.get('contagious'): st.info("🚧 Contagious — consider isolation")
                                if row.get('primary_symptom'): st.caption(f"📋 {row['primary_symptom']}")
                                if row.get('vet_notes'): st.info(f"📝 {row['vet_notes']}")
                                st.session_state['_analytics_btn_n'] += 1
                                _btn_key = f"gh_{st.session_state['_analytics_btn_n']}"
                                if st.button(f"📋 Go to History for {_pn}",
                                             key=_btn_key, use_container_width=True):
                                    st.session_state['history_search'] = _pn
                                    st.session_state['_page_override'] = '📋 History'
                                    st.rerun()

        # 1b — Species donut (CLICKABLE)
        with col_s1b:
            sp_c = df_h["species"].value_counts().reset_index()
            sp_c.columns = ["Species", "Cases"]
            sp_c["Hover"] = sp_c.apply(
                lambda r: (f"<b>{r.Species}</b><br>Cases: {r.Cases}<br>"
                           f"Unique diseases: {df_h[df_h['species']==r.Species]['disease'].nunique()}<br>"
                           f"Avg confidence: {df_h[df_h['species']==r.Species]['confidence'].mean():.1f}%"),
                axis=1)
            fig_sp = px.pie(sp_c, values="Cases", names="Species",
                            color_discrete_sequence=PAL,
                            hole=0.5,
                            custom_data=["Species","Hover"],
                            hover_data={"Cases": True})
            fig_sp.update_traces(
                textposition="outside",
                textinfo="label+percent",
                hovertext=sp_c["Hover"].tolist(),
                hovertemplate="%{hovertext}<extra></extra>",
                customdata=sp_c[["Species","Hover"]].values,
            )
            apply_layout(fig_sp, "Cases by Species — click a slice to drill down", height=400)
            clicked_species = st.plotly_chart(fig_sp, use_container_width=True,
                                              on_select="rerun", key="chart_species",
                                              selection_mode="points")
            sel_sp = (clicked_species.selection.get("points", [])
                      if clicked_species and clicked_species.selection else [])
            if sel_sp:
                sp_name = sel_sp[0].get("customdata", [None])[0]
                if sp_name:
                    sub_sp = df_h[df_h["species"] == sp_name]
                    with st.expander(f"📊 Drill-down: {sp_name} — {len(sub_sp)} case(s)", expanded=True):
                        st.markdown(f"""
                        <div style='display:flex;gap:20px;flex-wrap:wrap;margin-bottom:0.7rem'>
                            <span style='font-size:0.9rem;color:var(--muted)'>
                                Cases: <strong style='color:var(--text)'>{len(sub_sp)}</strong></span>
                            <span style='font-size:0.9rem;color:var(--muted)'>
                                Unique diseases: <strong style='color:var(--accent)'>{sub_sp["disease"].nunique()}</strong></span>
                            <span style='font-size:0.9rem;color:var(--muted)'>
                                Avg confidence: <strong style='color:var(--accent)'>{sub_sp["confidence"].mean():.1f}%</strong></span>
                        </div>""", unsafe_allow_html=True)
                        disease_tbl = (sub_sp.groupby("disease")
                                       .agg(Cases=("disease","count"),
                                            Avg_Conf=("confidence","mean"))
                                       .sort_values("Cases", ascending=False)
                                       .reset_index())
                        disease_tbl["Avg_Conf"] = disease_tbl["Avg_Conf"].round(1)
                        disease_tbl.columns = ["Disease","Cases","Avg Confidence %"]
                        st.dataframe(disease_tbl, use_container_width=True, hide_index=True)

        # 1c — Pareto curve (hover only — shows threshold context)
        disease_freq = df_h["disease"].value_counts()
        cum_pct = (disease_freq.cumsum() / disease_freq.sum() * 100).reset_index()
        cum_pct.columns = ["Disease","Cumulative %"]
        cum_pct["Rank"] = range(1, len(cum_pct)+1)
        fig_par = go.Figure()
        fig_par.add_trace(go.Scatter(
            x=cum_pct["Rank"], y=cum_pct["Cumulative %"],
            mode="lines", fill="tozeroy",
            line=dict(color="#5b9cf6", width=2),
            fillcolor="rgba(91,156,246,0.15)",
            hovertemplate=(
                "Rank #%{x}<br>Disease: %{customdata}<br>"
                "Cumulative coverage: %{y:.1f}%<extra></extra>"),
            customdata=cum_pct["Disease"]
        ))
        fig_par.add_hline(y=80, line_dash="dash", line_color="#fbbf24",
                          annotation_text="80% of cases", annotation_font_color="#fbbf24")
        apply_layout(fig_par, "Disease Concentration — hover to see which disease is at each rank", height=280)
        st.plotly_chart(fig_par, use_container_width=True, key="chart_pareto")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 2 — CLINICAL RISK PROFILING  (Plotly — interactive)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🚨 Clinical Risk Profiling")
        st.caption("Click any risk level bar to see the individual cases at that risk level.")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            dc = df_h["danger"].value_counts().reset_index()
            dc.columns = ["Risk Level","Cases"]
            dc["Color"] = dc["Risk Level"].map(RISK_COLORS)
            fig_risk = px.bar(dc, x="Risk Level", y="Cases",
                              color="Risk Level",
                              color_discrete_map=RISK_COLORS,
                              text="Cases",
                              custom_data=["Risk Level"])
            fig_risk.update_traces(
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Cases: %{y}<extra></extra>",
            )
            apply_layout(fig_risk, "Risk Level Distribution — click a bar to drill down", height=340)
            clicked_risk_bar = st.plotly_chart(fig_risk, use_container_width=True,
                                               on_select="rerun", key="chart_risk_dist",
                                               selection_mode="points")
            sel_risk = (clicked_risk_bar.selection.get("points",[])
                        if clicked_risk_bar and clicked_risk_bar.selection else [])
            if sel_risk:
                risk_name = sel_risk[0].get("customdata",[None])[0]
                if risk_name:
                    sub_r = df_h[df_h["danger"] == risk_name]
                    with st.expander(f"📊 {risk_name} Risk Cases — {len(sub_r)} record(s)", expanded=True):
                        st.markdown(f"""
                        <div style='display:flex;gap:20px;flex-wrap:wrap;margin-bottom:0.7rem'>
                            <span style='font-size:0.9rem;color:var(--muted)'>Avg confidence:
                                <strong style='color:var(--accent)'>{sub_r["confidence"].mean():.1f}%</strong></span>
                            <span style='font-size:0.9rem;color:var(--muted)'>Avg risk score:
                                <strong style='color:#fbbf24'>{sub_r["dangerous_level"].mean():.1f}</strong></span>
                            <span style='font-size:0.9rem;color:var(--muted)'>Species:
                                <strong style='color:var(--text)'>{", ".join(sub_r["species"].unique())}</strong></span>
                        </div>""", unsafe_allow_html=True)
                        st.dataframe(drill_detail(sub_r, risk_name),
                                     use_container_width=True, hide_index=True)

        with col_r2:
            fig_temp_hr = px.scatter(df_h, x="temp", y="hr",
                                     color="danger",
                                     color_discrete_map=RISK_COLORS,
                                     hover_name="disease",
                                     hover_data={"species":True,"confidence":True,
                                                 "temp":True,"hr":True,"danger":True},
                                     labels={"temp":"Body Temperature (°C)",
                                             "hr":"Heart Rate (bpm)",
                                             "danger":"Risk Level"})
            apply_layout(fig_temp_hr,
                         "Temperature vs Heart Rate — hover for case details", height=340)
            st.plotly_chart(fig_temp_hr, use_container_width=True, key="chart_temp_hr")

        # Risk score histogram
        fig_dl = px.histogram(df_h, x="dangerous_level", nbins=20,
                              color_discrete_sequence=["#5b9cf6"],
                              labels={"dangerous_level":"Risk Score (0–100)"})
        fig_dl.add_vline(x=df_h["dangerous_level"].mean(), line_dash="dash",
                         line_color="#fbbf24",
                         annotation_text=f"Mean: {df_h['dangerous_level'].mean():.1f}",
                         annotation_font_color="#fbbf24")
        apply_layout(fig_dl, "Risk Score Distribution", height=260)
        st.plotly_chart(fig_dl, use_container_width=True, key="chart_risk_hist")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3 — CONFIDENCE ANALYSIS  (Plotly — interactive)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🎯 Model Confidence Analysis")
        st.caption("Click any species bar to see its individual confidence distribution.")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_conf_hist = px.histogram(df_h, x="confidence", nbins=20,
                                         color_discrete_sequence=["#5b9cf6"],
                                         labels={"confidence":"Confidence (%)"})
            for pct, col_v, lbl in [(25,"#a78bfa","P25"),(50,"#fbbf24","Median"),
                                     (75,"#f87171","P75")]:
                v = float(df_h["confidence"].quantile(pct/100))
                fig_conf_hist.add_vline(x=v, line_dash="dot", line_color=col_v,
                                        annotation_text=f"{lbl}: {v:.1f}%",
                                        annotation_font_color=col_v)
            apply_layout(fig_conf_hist, "Confidence Score Distribution", height=320)
            st.plotly_chart(fig_conf_hist, use_container_width=True, key="chart_conf_hist")

        with col_c2:
            sp_conf = (df_h.groupby("species")["confidence"]
                       .agg(["mean","std","count"])
                       .sort_values("mean", ascending=True)
                       .reset_index())
            sp_conf.columns = ["Species","Mean Confidence","Std Dev","Cases"]
            sp_conf["Std Dev"] = sp_conf["Std Dev"].fillna(0)
            fig_sp_conf = px.bar(sp_conf, x="Mean Confidence", y="Species",
                                 orientation="h",
                                 error_x="Std Dev",
                                 color="Mean Confidence",
                                 color_continuous_scale=["#2e3850","#5b9cf6"],
                                 hover_data={"Cases":True,"Std Dev":":.1f"},
                                 custom_data=["Species","Cases"])
            fig_sp_conf.update_coloraxes(showscale=False)
            fig_sp_conf.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>Mean confidence: %{x:.1f}%<br>"
                    "Cases: %{customdata[1]}<extra></extra>"))
            apply_layout(fig_sp_conf,
                         "Avg Confidence by Species (± std dev) — click to drill down", height=320)
            clicked_sp_conf = st.plotly_chart(fig_sp_conf, use_container_width=True,
                                              on_select="rerun", key="chart_sp_conf",
                                              selection_mode="points")
            sel_spc = (clicked_sp_conf.selection.get("points",[])
                       if clicked_sp_conf and clicked_sp_conf.selection else [])
            if sel_spc:
                sp_clicked = sel_spc[0].get("customdata",[None])[0]
                if sp_clicked:
                    sub_spc = df_h[df_h["species"]==sp_clicked]
                    with st.expander(f"📊 Confidence detail: {sp_clicked}", expanded=True):
                        fig_box = px.box(sub_spc, x="confidence", color_discrete_sequence=["#5b9cf6"],
                                         labels={"confidence":"Confidence (%)"})
                        apply_layout(fig_box, f"Confidence distribution — {sp_clicked}", height=220)
                        st.plotly_chart(fig_box, use_container_width=True, key="chart_conf_box")
                        st.dataframe(drill_detail(sub_spc, sp_clicked),
                                     use_container_width=True, hide_index=True)

        # ── Confidence tier breakdown ──────────────────────────────────────────
        st.markdown("**Confidence tier breakdown**")
        tier_counts = {
            "🟢 High (≥70%)":     int((df_h['confidence'] >= 70).sum()),
            "🟡 Moderate (30–70%)": int(((df_h['confidence'] >= 30) & (df_h['confidence'] < 70)).sum()),
            "🔴 Low (<30%)":       int((df_h['confidence'] < 30).sum()),
        }
        t1, t2, t3 = st.columns(3)
        for col, (label, count) in zip([t1, t2, t3], tier_counts.items()):
            pct = count / len(df_h) * 100 if len(df_h) > 0 else 0
            color = "#34d399" if "High" in label else "#fbbf24" if "Moderate" in label else "#f87171"
            col.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-val' style='color:{color}'>{count}</div>"
                f"<div class='metric-lbl'>{label}</div>"
                f"<div style='font-size:0.72rem;color:var(--muted);margin-top:2px'>{pct:.0f}% of cases</div>"
                f"</div>", unsafe_allow_html=True)
        st.caption("High confidence (≥70%) = model strongly discriminates. "
                   "Low confidence (<30%) = high diagnostic ambiguity — "
                   "expand differential list and consider additional workup.")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 4 — TEMPORAL TRENDS  (Plotly — interactive)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📅 Temporal Trends")
        st.caption("Hover over data points to see exact values. Click a point on the daily volume chart to see cases from that date.")

        daily = df_h.groupby("date").size().reset_index(name="count")
        daily = daily.sort_values("date")

        if len(daily) >= 2:
            daily["rolling"] = daily["count"].rolling(3, min_periods=1).mean()
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=daily["date"], y=daily["count"],
                mode="lines+markers", name="Daily cases",
                line=dict(color="#5b9cf6", width=2),
                fill="tozeroy", fillcolor="rgba(91,156,246,0.12)",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Cases: %{y}<extra></extra>",
                customdata=daily["date"].astype(str)
            ))
            if len(daily) >= 3:
                fig_time.add_trace(go.Scatter(
                    x=daily["date"], y=daily["rolling"],
                    mode="lines", name="3-day avg",
                    line=dict(color="#fbbf24", width=1.5, dash="dash"),
                    hovertemplate="3-day avg: %{y:.1f}<extra></extra>"
                ))
            apply_layout(fig_time, "Daily Case Volume — click a point to see that day's cases", height=300)
            clicked_time = st.plotly_chart(fig_time, use_container_width=True,
                                           on_select="rerun", key="chart_time",
                                           selection_mode="points")
            sel_time = (clicked_time.selection.get("points",[])
                        if clicked_time and clicked_time.selection else [])
            if sel_time:
                date_str = str(sel_time[0].get("customdata",""))[:10]
                if date_str:
                    sub_date = df_h[df_h["date"].astype(str).str[:10] == date_str]
                    with st.expander(f"📊 Cases on {date_str} — {len(sub_date)} record(s)", expanded=True):
                        st.dataframe(drill_detail(sub_date, date_str),
                                     use_container_width=True, hide_index=True)

            conf_time = df_h.groupby("date")["confidence"].mean().reset_index()
            conf_time = conf_time.sort_values("date")
            fig_conf_time = px.line(conf_time, x="date", y="confidence",
                                    markers=True,
                                    color_discrete_sequence=["#34d399"],
                                    labels={"confidence":"Avg Confidence (%)",
                                            "date":"Date"})
            fig_conf_time.add_hline(y=float(df_h["confidence"].mean()),
                                    line_dash="dash", line_color="#fbbf24",
                                    annotation_text=f"Overall avg: {df_h['confidence'].mean():.1f}%",
                                    annotation_font_color="#fbbf24")
            apply_layout(fig_conf_time, "Average Model Confidence Over Time", height=260)
            st.plotly_chart(fig_conf_time, use_container_width=True, key="chart_conf_time")
        else:
            st.info("Run predictions on multiple dates to see temporal trends.")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 5 — PHYSIOLOGICAL MEASUREMENTS  (Plotly — interactive)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🌡️ Physiological Measurements")
        st.caption("Hover over data points to see species, temperature, heart rate, and risk. "
                   "Click a point to see full case details.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_temp_sp = px.box(df_h, x="species", y="temp",
                                 color="species",
                                 color_discrete_sequence=PAL,
                                 labels={"temp":"Body Temperature (°C)","species":""},
                                 points="all")
            apply_layout(fig_temp_sp,
                         "Temperature by Species — hover over points for case detail", height=340)
            fig_temp_sp.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_temp_sp, use_container_width=True, key="chart_temp_sp")

        with col_p2:
            fig_hr_sp = px.box(df_h, x="species", y="hr",
                               color="species",
                               color_discrete_sequence=PAL,
                               labels={"hr":"Heart Rate (bpm)","species":""},
                               points="all")
            apply_layout(fig_hr_sp,
                         "Heart Rate by Species — hover over points for case detail", height=340)
            fig_hr_sp.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_hr_sp, use_container_width=True, key="chart_hr_sp")

        # Age vs Weight bubble — clickable
        df_h_num = df_h.copy()
        df_h_num["age_num"] = pd.to_numeric(df_h_num["age"], errors="coerce").fillna(3)
        fig_aw = px.scatter(df_h_num, x="age_num", y="weight",
                            color="dangerous_level",
                            color_continuous_scale="RdYlGn_r",
                            size="dangerous_level",
                            size_max=25,
                            hover_name="disease",
                            hover_data={"species":True,"confidence":True,
                                        "temp":True,"hr":True,"dangerous_level":True},
                            labels={"age_num":"Age (years)","weight":"Weight (kg)",
                                    "dangerous_level":"Risk Score"})
        apply_layout(fig_aw,
                     "Age vs Weight — bubble size & colour = risk score", height=320)
        st.plotly_chart(fig_aw, use_container_width=True, key="chart_age_weight")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 6 — BIOSAFETY & EPIDEMIOLOGY  (Plotly — interactive)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🛡️ Biosafety & Epidemiological Risk")
        st.caption("Click a biosafety category or zoonotic disease bar to see the affected cases.")

        zoo_count_local  = int(df_h["zoonotic"].sum())
        cont_count_local = int(df_h["contagious"].sum())
        safe_count_local = int((~df_h["contagious"] & ~df_h["zoonotic"]).sum())

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            bio_df = pd.DataFrame({
                "Category": ["Zoonotic","Contagious only","Neither"],
                "Cases":    [zoo_count_local,
                             int((df_h["contagious"] & ~df_h["zoonotic"]).sum()),
                             safe_count_local],
                "Color":    ["#ef4444","#fbbf24","#34d399"]
            })
            fig_bio = px.pie(bio_df, values="Cases", names="Category",
                             color="Category",
                             color_discrete_map={"Zoonotic":"#ef4444",
                                                 "Contagious only":"#fbbf24",
                                                 "Neither":"#34d399"},
                             hole=0.5,
                             custom_data=["Category"])
            fig_bio.update_traces(
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Cases: %{value}<extra></extra>"
            )
            apply_layout(fig_bio, "Biosafety Risk Distribution — click a slice to drill down", height=340)
            clicked_bio = st.plotly_chart(fig_bio, use_container_width=True,
                                          on_select="rerun", key="chart_bio_pie",
                                          selection_mode="points")
            sel_bio = (clicked_bio.selection.get("points",[])
                       if clicked_bio and clicked_bio.selection else [])
            if sel_bio:
                cat = sel_bio[0].get("customdata",[None])[0]
                if cat == "Zoonotic":
                    sub_bio = df_h[df_h["zoonotic"]]
                elif cat == "Contagious only":
                    sub_bio = df_h[df_h["contagious"] & ~df_h["zoonotic"]]
                else:
                    sub_bio = df_h[~df_h["contagious"] & ~df_h["zoonotic"]]
                with st.expander(f"📊 {cat} cases — {len(sub_bio)} record(s)", expanded=True):
                    st.dataframe(drill_detail(sub_bio, cat),
                                 use_container_width=True, hide_index=True)

        with col_b2:
            zoo_df = df_h[df_h["zoonotic"]]
            if len(zoo_df) > 0:
                zd = zoo_df["disease"].value_counts().head(8).reset_index()
                zd.columns = ["Disease","Cases"]
                zd["Hover"] = zd.apply(
                    lambda r: f"<b>{r.Disease}</b><br>Cases: {r.Cases}",
                    axis=1)
                fig_zoo = px.bar(zd, x="Cases", y="Disease", orientation="h",
                                 color_discrete_sequence=["#ef4444"],
                                 text="Cases",
                                 custom_data=["Disease","Hover"])
                fig_zoo.update_traces(
                    textposition="outside",
                    hovertemplate="%{customdata[1]}<extra></extra>")
                apply_layout(fig_zoo,
                             "🦠 Top Zoonotic Diseases — click to see affected cases", height=340)
                fig_zoo.update_yaxes(categoryorder="total ascending")
                clicked_zoo = st.plotly_chart(fig_zoo, use_container_width=True,
                                              on_select="rerun", key="chart_zoo_bar",
                                              selection_mode="points")
                sel_zoo = (clicked_zoo.selection.get("points",[])
                           if clicked_zoo and clicked_zoo.selection else [])
                if sel_zoo:
                    z_disease = sel_zoo[0].get("customdata",[None])[0]
                    if z_disease:
                        sub_z = df_h[df_h["disease"]==z_disease]
                        with st.expander(f"📊 Zoonotic drill-down: {z_disease}", expanded=True):
                            st.warning(f"🦠 **{z_disease}** is zoonotic — "
                                       "inform owners and apply clinic biosafety protocols.")
                            st.dataframe(drill_detail(sub_z, z_disease),
                                         use_container_width=True, hide_index=True)
            else:
                st.info("No zoonotic cases in current filtered view.")


        # ══════════════════════════════════════════════════════════════════
        # SECTION 7 — SUMMARY STATISTICS TABLE
        # ══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📋 Summary Statistics Table")
        st.caption("Descriptive statistics for all numerical clinical measurements across the session caseload.")

        num_cols = ['confidence','dangerous_level','temp','hr','symptom_count']
        num_labels = {
            'confidence':    'Confidence (%)',
            'dangerous_level': 'Risk Score (0–100)',
            'temp':          'Temperature (°C)',
            'hr':            'Heart Rate (bpm)',
            'symptom_count': 'Symptom Count'
        }
        stats_rows = []
        for col, label in num_labels.items():
            if col in df_h.columns:
                s = df_h[col].describe()
                stats_rows.append({
                    'Metric':     label,
                    'N':          int(s['count']),
                    'Mean':       f"{s['mean']:.2f}",
                    'Std Dev':    f"{s['std']:.2f}",
                    'Min':        f"{s['min']:.2f}",
                    'Q1 (25%)':   f"{s['25%']:.2f}",
                    'Median':     f"{s['50%']:.2f}",
                    'Q3 (75%)':   f"{s['75%']:.2f}",
                    'Max':        f"{s['max']:.2f}",
                })
        if stats_rows:
            st.dataframe(pd.DataFrame(stats_rows).set_index('Metric'),
                         use_container_width=True)
            st.caption("Q1/Q3 = 25th/75th percentile. Std Dev = standard deviation. "
                       "These statistics characterise the clinical measurement distributions across all predictions made in this session.")

        # Zoonotic text summary
        if zoo_count > 0 or cont_count > 0:
            st.markdown("---")
            st.markdown("**⚠️ Biosafety Case Summary**")
            col_z, col_c = st.columns(2)
            with col_z:
                zoo_cases = df_h[df_h['zoonotic']]['disease'].value_counts()
                if len(zoo_cases) > 0:
                    st.caption(f"🦠 **Zoonotic cases detected: {zoo_count}**")
                    for d, c in zoo_cases.items():
                        st.caption(f"  • {d}: {c} case{'s' if c>1 else ''}")
                    st.caption("_These diseases can transmit to humans. Inform owners and apply clinic biosafety protocols._")
            with col_c:
                cont_cases = df_h[df_h['contagious']]['disease'].value_counts()
                if len(cont_cases) > 0:
                    st.caption(f"🚧 **Contagious animal-to-animal cases: {cont_count}**")
                    for d, c in cont_cases.items():
                        st.caption(f"  • {d}: {c} case{'s' if c>1 else ''}")
                    st.caption("_Isolate affected animals and monitor in-contact herd/flock members._")

        # ══════════════════════════════════════════════════════════════════
        # SECTION 8 — END-OF-SESSION CLINICAL SUMMARY REPORT
        # ══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📄 End-of-Session Clinical Summary")
        st.caption("Download a formatted HTML report summarising all predictions, biosafety flags, "
                   "and clinical recommendations for this session. Open in browser and print as PDF.")

        _total      = len(df_h)
        _species_n  = df_h['species'].nunique()
        _diseases_n = df_h['disease'].nunique()
        _avg_conf   = df_h['confidence'].mean()
        _high_risk  = int(df_h['danger'].isin(['High','Critical']).sum())
        _zoo_n      = int(df_h['zoonotic'].sum())
        _cont_n     = int(df_h['contagious'].sum())
        _top_disease = df_h['disease'].value_counts().index[0] if _total > 0 else "—"

        _case_rows = ""
        for _, row in df_h.sort_values('timestamp', ascending=False).iterrows():
            _zoo_tag  = "🦠 Zoonotic" if row.get('zoonotic') else ""
            _cont_tag = "🚧 Contagious" if row.get('contagious') else ""
            _flags    = " · ".join(filter(None, [_zoo_tag, _cont_tag])) or "—"
            _case_rows += f"""
            <tr style='background:{"#1c2131" if int(row.name) % 2 == 0 else "#131720"}'>
              <td style='padding:5px 10px;color:#f0f4ff'>#{row.get('id','')}</td>
              <td style='padding:5px 10px;color:#f0f4ff'>{row.get('patient_name','—')}</td>
              <td style='padding:5px 10px;color:#a0aab8'>{row.get('species','')} · {row.get('breed','')}</td>
              <td style='padding:5px 10px;color:#f0f4ff'>{row.get('disease','')}</td>
              <td style='padding:5px 10px;color:#5b9cf6;text-align:right'>{row.get('confidence','')}%</td>
              <td style='padding:5px 10px;color:#fbbf24'>{row.get('danger','')}</td>
              <td style='padding:5px 10px;color:#ffb3b3;font-size:11px'>{_flags}</td>
            </tr>"""

        _session_html = f"""<!DOCTYPE html>
<html><head><meta charset='UTF-8'>
<title>VetDx Session Summary — {datetime.date.today()}</title>
<style>
body{{margin:0;padding:28px 36px;background:#0c0f18;color:#f0f4ff;
     font-family:'Segoe UI',Arial,sans-serif;font-size:13px;line-height:1.7}}
h1{{font-size:20px;color:#5b9cf6;margin:0 0 4px}}
h2{{font-size:14px;color:#5b9cf6;margin:18px 0 6px;border-bottom:1px solid #2e3850;padding-bottom:4px}}
.header{{display:flex;justify-content:space-between;align-items:flex-start;
         border-bottom:2px solid #5b9cf6;padding-bottom:14px;margin-bottom:18px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0}}
.kpi{{background:#131720;border:1px solid #2e3850;border-radius:7px;padding:10px 14px;text-align:center}}
.kpi-val{{font-size:22px;font-weight:700;color:#5b9cf6}}
.kpi-lbl{{font-size:10px;color:#a0aab8;text-transform:uppercase;letter-spacing:.05em}}
table{{width:100%;border-collapse:collapse;margin:8px 0}}
th{{background:#1c2131;color:#a0aab8;font-size:11px;padding:6px 10px;text-align:left;
   text-transform:uppercase;letter-spacing:.04em}}
td{{border-bottom:1px solid #1c2131}}
.alert-zoo{{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.4);
           border-radius:6px;padding:8px 12px;margin:8px 0;color:#ffb3b3;font-size:12px}}
.alert-cont{{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.35);
            border-radius:6px;padding:8px 12px;margin:8px 0;color:#ffe99a;font-size:12px}}
.footer{{margin-top:24px;padding-top:10px;border-top:1px solid #2e3850;
         font-size:10px;color:#a0aab8;text-align:center}}
</style></head><body>
<div class='header'>
  <div>
    <h1>VetDx — Session Clinical Summary</h1>
    <div style='color:#a0aab8;font-size:12px'>
      Date: {datetime.date.today()} &nbsp;·&nbsp;
      Generated by VetDx FYP · APU 2026 &nbsp;·&nbsp;
      Not a licensed medical device
    </div>
  </div>
  <div style='text-align:right'>
    <div style='font-size:22px;font-weight:700;color:#5b9cf6'>{_total}</div>
    <div style='font-size:11px;color:#a0aab8'>Total cases</div>
  </div>
</div>

<h2>Session Overview</h2>
<div class='kpi-grid'>
  <div class='kpi'><div class='kpi-val'>{_total}</div><div class='kpi-lbl'>Cases</div></div>
  <div class='kpi'><div class='kpi-val'>{_species_n}</div><div class='kpi-lbl'>Species</div></div>
  <div class='kpi'><div class='kpi-val'>{_diseases_n}</div><div class='kpi-lbl'>Unique Diseases</div></div>
  <div class='kpi'><div class='kpi-val'>{_avg_conf:.1f}%</div><div class='kpi-lbl'>Avg Confidence</div></div>
  <div class='kpi'><div class='kpi-val' style='color:#f87171'>{_high_risk}</div><div class='kpi-lbl'>High/Critical Risk</div></div>
  <div class='kpi'><div class='kpi-val' style='color:#ffb3b3'>{_zoo_n}</div><div class='kpi-lbl'>Zoonotic Cases</div></div>
  <div class='kpi'><div class='kpi-val' style='color:#ffe99a'>{_cont_n}</div><div class='kpi-lbl'>Contagious Cases</div></div>
  <div class='kpi'><div class='kpi-val' style='font-size:14px;color:#f0f4ff'>{_top_disease[:20]}</div><div class='kpi-lbl'>Most Predicted</div></div>
</div>

{f"<div class='alert-zoo'>🦠 <strong>Zoonotic diseases detected ({_zoo_n} cases):</strong> " + ", ".join(df_h[df_h['zoonotic']]['disease'].unique()) + "<br>Advise owner hygiene precautions. Report to public health authorities if required.</div>" if _zoo_n > 0 else ""}
{f"<div class='alert-cont'>🚧 <strong>Contagious diseases detected ({_cont_n} cases):</strong> " + ", ".join(df_h[df_h['contagious']]['disease'].unique()) + "<br>Ensure affected animals are isolated. Monitor in-contact animals.</div>" if _cont_n > 0 else ""}

<h2>All Cases This Session</h2>
<table>
  <tr>
    <th>#</th><th>Patient</th><th>Species · Breed</th>
    <th>Predicted Disease</th><th style='text-align:right'>Confidence</th>
    <th>Risk</th><th>Biosafety Flags</th>
  </tr>
  {_case_rows}
</table>

<div class='footer'>
  ⚕️ VetDx is a research prototype (FYP, APU 2026). All predictions are probabilistic estimates
  to assist — not replace — qualified veterinary clinical judgment.
  AUC-ROC 0.9843 · Top-5 Accuracy 99.00% · 162 diseases · 16 species.
</div>
</body></html>"""

        st.download_button(
            label="📥 Download session summary (HTML → print as PDF)",
            data=_session_html,
            file_name=f"VetDx_Session_{datetime.date.today()}.html",
            mime="text/html",
            use_container_width=True,
            help="Open the downloaded file in Chrome and use File → Print → Save as PDF"
        )


# ==============================================================================
# PAGE 4 — ABOUT
# ==============================================================================
elif page == "ℹ️ About":
    st.markdown("## About VetDx")
    st.markdown("""
    <div class="card">
    <p class="card-title">🎯 System Overview</p>
    <p style="color:var(--muted);font-size:0.86rem;line-height:1.7">
    VetDx is a veterinary disease prediction system developed as part of a
    B.Sc. Computer Science (Data Analytics) Final Year Project at Asia Pacific University
    of Technology and Innovation (APU), 2026. The system supports clinical decision-making
    by generating probabilistic disease predictions from symptom descriptions across
    16 animal species and 162 disease categories, with integrated zoonotic disease
    alerts aligned with the One Health framework and UN SDG 3.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
        <p class="card-title">🔬 Technical Architecture</p>
        <ul style="color:var(--muted);font-size:0.83rem;line-height:1.9">
            <li><strong>NLP encoder:</strong> Fine-tuned BERT (selected over BioBERT &amp; PubMedBERT — Val F1 0.0897)</li>
            <li><strong>Classifiers compared:</strong> XGBoost · Random Forest · MLP · Logistic Regression</li>
            <li><strong>Production classifier:</strong> XGBoost (5-fold CV F1 = 0.2490, won all folds)</li>
            <li><strong>Feature fusion:</strong> BERT embeddings (768) + Structured clinical (26, incl. 8 engineered)
                + Species/Breed/Gender (3) = <strong>797 total features</strong></li>
            <li><strong>Engineered features:</strong> Temp_Dev_Species · HR_Dev_Species · GI_Combo · Resp_Combo
                · Severity×Age · Temp×Symptoms · Multi_System · HR×Risk</li>
            <li><strong>Dataset:</strong> 59,314 cases · 16 species · 162 diseases</li>
            <li><strong>Methodology:</strong> CRISP-DM</li>
            <li><strong>AUC-ROC (Macro):</strong> 0.9843 · Top-5 Accuracy: 99.00%</li>
            <li><strong>F1 Macro:</strong> 0.1868 (56.4× better than random baseline)</li>
            <li><strong>Cross-validation:</strong> 5-fold Stratified · XGBoost won all 5 folds</li>
            <li><strong>Statistical tests:</strong> McNemar's (p&lt;0.05) + Wilcoxon signed-rank</li>
            <li><strong>Explainability:</strong> SHAP TreeExplainer (per-prediction)</li>
            <li><strong>Zoonotic alerts:</strong> 25 diseases flagged</li>
            <li><strong>Contagion alerts:</strong> 54 diseases flagged</li>
        </ul>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
        <p class="card-title">🌍 SDG 3 Alignment</p>
        <p style="color:var(--muted);font-size:0.83rem;line-height:1.7">
        Supports <strong>UN SDG 3: Good Health and Well-being</strong>:</p>
        <ul style="color:var(--muted);font-size:0.83rem;line-height:1.9">
            <li>Early zoonotic disease detection — SDG Target 3.3</li>
            <li>One Health framework integration</li>
            <li>Expanding diagnostic access in resource-limited settings — SDG 3.8</li>
            <li>Multi-species surveillance supporting global health security</li>
            <li>Biosafety alerts for 25 zoonotic and 54 contagious diseases</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    # ── Model evaluation results card ─────────────────────────────────────────
    st.markdown("""
    <div class="card">
    <p class="card-title">📊 Model Evaluation Results (Test Set — 8,892 records)</p>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:0.8rem">
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.8rem;text-align:center">
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:#34d399">0.9843</div>
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-top:3px">AUC-ROC Macro</div>
            <div style="font-size:0.7rem;color:#34d399;margin-top:2px">Target ≥ 0.90 ✅</div>
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.8rem;text-align:center">
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:#34d399">99.00%</div>
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-top:3px">Top-5 Accuracy</div>
            <div style="font-size:0.7rem;color:#34d399;margin-top:2px">8,803 / 8,892 cases ✅</div>
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.8rem;text-align:center">
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:#fbbf24">0.1868</div>
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-top:3px">F1 Macro</div>
            <div style="font-size:0.7rem;color:#fbbf24;margin-top:2px">56.4× better than random</div>
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.8rem;text-align:center">
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:#5b9cf6">73.66%</div>
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-top:3px">Top-3 Accuracy</div>
            <div style="font-size:0.7rem;color:#5b9cf6;margin-top:2px">6,564 / 8,892 cases</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.65rem 0.8rem">
            <span style="font-size:0.72rem;color:var(--muted)">Statistical validation</span><br>
            <span style="font-size:0.82rem;color:var(--text)">McNemar's test <strong style="color:#34d399">p = 0.0296</strong> · Wilcoxon <strong style="color:#34d399">p = 0.002</strong></span>
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.65rem 0.8rem">
            <span style="font-size:0.72rem;color:var(--muted)">Cross-validation (XGBoost 5-fold)</span><br>
            <span style="font-size:0.82rem;color:var(--text)">CV F1 = <strong style="color:#5b9cf6">0.2490 ± 0.0157</strong> · Won all 5 folds</span>
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.65rem 0.8rem">
            <span style="font-size:0.72rem;color:var(--muted)">Ablation study (fusion gain)</span><br>
            <span style="font-size:0.82rem;color:var(--text)">Full fusion F1 <strong style="color:#34d399">+0.0075</strong> over BERT-only</span>
        </div>
    </div>
    <p style="font-size:0.72rem;color:var(--muted);margin-top:0.6rem;margin-bottom:0">
    F1 Macro 0.1868 is the macro average across all 162 disease classes including 96 classes with zero test support.
    The system performs 56.4× better than random chance (1/162 = 0.617%). AUC-ROC and Top-5 Accuracy are the
    primary performance indicators for a differential diagnosis support system.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    ⚕️ <strong>Important disclaimer:</strong> VetDx is a research prototype developed as a
    Final Year Project (FYP) at Asia Pacific University of Technology and Innovation (APU), 2026.
    It is a clinical decision support tool only — not a licensed medical device — and does not
    replace professional veterinary diagnosis, physical examination, or laboratory investigation.
    All predictions are probabilistic estimates intended to assist, not substitute, qualified
    veterinary clinical judgment. Zoonotic and contagion alerts are based on disease classification
    and should be confirmed by a qualified veterinarian before action is taken.
    </div>""", unsafe_allow_html=True)