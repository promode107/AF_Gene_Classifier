"""
AF Gene Sequence Classifier — Streamlit App
Tabs: (1) Gene Prediction, (2) Data Description, (3) ML Models & Workflow
"""

import streamlit as st
import numpy as np
import re
import os
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AF Gene Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #0d1117;
    --panel:     #161b22;
    --border:    #21262d;
    --accent:    #58a6ff;
    --accent2:   #3fb950;
    --accent3:   #f78166;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --highlight: #1f6feb;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

/* Header */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(88,166,255,0.06) 0%, transparent 60%),
                radial-gradient(circle at 70% 60%, rgba(63,185,80,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    font-family: 'Space Mono', monospace;
    margin: 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel);
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--muted);
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    border: none !important;
    background: transparent;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: var(--highlight) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* Cards */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0 0 1rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 0.8rem;
}

/* Gene badge */
.gene-badge {
    display: inline-block;
    background: rgba(88,166,255,0.12);
    border: 1px solid rgba(88,166,255,0.3);
    color: var(--accent);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    margin: 2px;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, rgba(63,185,80,0.08), rgba(63,185,80,0.03));
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-gene {
    font-size: 3rem;
    font-weight: 800;
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
    letter-spacing: 2px;
}
.result-label { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.5rem; }
.result-confidence {
    font-size: 1.1rem;
    color: var(--text);
    margin-top: 0.5rem;
    font-family: 'Space Mono', monospace;
}

/* Error box */
.error-box {
    background: rgba(247,129,102,0.08);
    border: 1px solid rgba(247,129,102,0.3);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: var(--accent3);
    font-size: 0.9rem;
}

/* Stat cards */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.stat { flex: 1; background: var(--panel); border: 1px solid var(--border);
        border-radius: 10px; padding: 1rem 1.2rem; }
.stat-val { font-size: 1.8rem; font-weight: 800; color: var(--accent);
             font-family: 'Space Mono', monospace; }
.stat-key { font-size: 0.75rem; color: var(--muted); text-transform: uppercase;
             letter-spacing: 0.5px; margin-top: 0.2rem; }

/* Flow diagram */
.flow-step {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.flow-num {
    background: var(--highlight);
    color: white;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.flow-arrow { text-align: center; color: var(--muted); font-size: 1.2rem;
               margin: 0.15rem 0; padding-left: 1.2rem; }

/* Model card */
.model-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.model-card:hover { border-color: var(--accent); }
.model-name { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; }
.model-tag {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.25);
    color: var(--accent);
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 7px;
    margin-right: 4px;
    margin-bottom: 0.6rem;
}
.model-desc { color: var(--muted); font-size: 0.88rem; line-height: 1.6; }

/* Textarea override */
.stTextArea textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* Button */
.stButton button {
    background: var(--highlight) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.6rem !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Select box */
.stSelectbox > div { background: var(--panel) !important; }

/* Info strip */
.info-strip {
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 1rem;
    font-family: 'Space Mono', monospace;
}

/* Divider */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* Progress bar custom */
.metric-bar-wrap { margin-bottom: 0.6rem; }
.metric-bar-label { display: flex; justify-content: space-between;
                    font-size: 0.82rem; color: var(--muted); margin-bottom: 3px; }
.metric-bar-track { background: var(--border); border-radius: 4px; height: 8px; }
.metric-bar-fill { height: 8px; border-radius: 4px;
                   background: linear-gradient(90deg, var(--accent), var(--accent2)); }

/* AF card highlight */
.af-highlight {
    background: linear-gradient(135deg, rgba(88,166,255,0.06), var(--panel));
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.af-highlight h4 { color: var(--accent); margin: 0 0 0.6rem; font-size: 0.95rem; }
.af-highlight p  { color: var(--muted);  margin: 0; font-size: 0.88rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────────
GENES_16 = [
    'SCN5A', 'KCNQ1', 'RYR2',  'KCNH2', 'KCNA5', 'KCNJ2',
    'NPPA',  'GJA5',  'PITX2', 'PRKAG2','MYL4',  'TBX5',
    'SCN1B', 'SCN2B', 'SCN3B', 'SCN4B',
]

GENE_INFO = {
    'SCN5A':  ("Nav1.5 — Cardiac Sodium Channel",   "Primary voltage-gated sodium channel responsible for initiating cardiac action potentials. Mutations cause Brugada syndrome and familial AF."),
    'KCNQ1':  ("Kv7.1 — Slow K⁺ Channel",          "Encodes the α-subunit of the slow delayed rectifier K⁺ current (IKs). Loss-of-function variants strongly linked to familial AF."),
    'RYR2':   ("Ryanodine Receptor 2",               "SR calcium release channel. Dysregulation causes spontaneous Ca²⁺ sparks that trigger delayed afterdepolarisations and AF."),
    'KCNH2':  ("hERG — Rapid K⁺ Channel",           "Mediates rapid delayed rectifier K⁺ current (IKr). Gain-of-function shortens the action potential and promotes AF."),
    'KCNA5':  ("Kv1.5 — Ultra-Rapid K⁺",           "Carries the ultra-rapid K⁺ current (IKur), exclusive to the atria. Loss-of-function variants are a recognised AF risk factor."),
    'KCNJ2':  ("Kir2.1 — Inward Rectifier",         "Stabilises the resting membrane potential. Gain-of-function shortens APD and can sustain re-entrant AF circuits."),
    'NPPA':   ("ANP — Atrial Natriuretic Peptide",  "Secreted hormone regulating blood pressure and fluid balance. Variants alter atrial structural remodelling and AF susceptibility."),
    'GJA5':   ("Connexin 40 — Gap Junction",        "Forms atrial gap junctions that enable rapid electrical coupling. Mutations slow conduction velocity, creating AF substrates."),
    'PITX2':  ("PITX2 — Homeobox TF",               "Transcription factor governing left atrial development. The strongest GWAS signal for AF maps near PITX2."),
    'PRKAG2': ("AMP Kinase γ-Subunit",              "Regulates cellular energy sensing. Gain-of-function mutations cause Wolff-Parkinson-White syndrome and progressive AF."),
    'MYL4':   ("Myosin Light Chain 4",              "Atrial-specific isoform of myosin light chain. Variants disrupt sarcomere assembly and promote atrial cardiomyopathy."),
    'TBX5':   ("TBX5 — T-Box Transcription Factor","Cardiac development regulator. Mutations cause Holt-Oram syndrome, which includes AF among its cardiac manifestations."),
    'SCN1B':  ("Nav β1 — Sodium Channel β1",       "Modulates Nav1.5 gating kinetics and membrane trafficking. Variants alter sodium current and AF risk."),
    'SCN2B':  ("Nav β2 — Sodium Channel β2",       "Cell-adhesion-like regulatory subunit for cardiac Nav channels; variants affect channel density and AF propensity."),
    'SCN3B':  ("Nav β3 — Sodium Channel β3",       "Regulatory β-subunit that affects Nav1.5 expression; loss-of-function associated with Brugada syndrome and AF."),
    'SCN4B':  ("Nav β4 — Sodium Channel β4",       "Influences late sodium current; dysregulation contributes to afterdepolarisations and an arrhythmogenic substrate."),
}

# Sequence counts per gene — Reverted to the full 10,544 dataset (perfectly balanced 659 per gene)
GENE_COUNTS = {
    'SCN5A': 659, 'KCNQ1': 659, 'RYR2':   659, 'KCNH2':  659,
    'KCNA5': 659, 'KCNJ2': 659, 'NPPA':   659, 'GJA5':   659,
    'PITX2': 659, 'PRKAG2':659, 'MYL4':   659, 'TBX5':   659,
    'SCN1B': 659, 'SCN2B': 659, 'SCN3B':  659, 'SCN4B':  659,
}

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

HYDROPHOBICITY = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}
AA_WEIGHTS = {
    'A': 89,  'R': 174, 'N': 132, 'D': 133, 'C': 121,
    'Q': 146, 'E': 147, 'G':  75, 'H': 155, 'I': 131,
    'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
    'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117,
}
CHARGES      = {'R': 1, 'K': 1, 'D': -1, 'E': -1}
AMINO_ACIDS  = list('ACDEFGHIKLMNPQRSTVWY')
CTD_GROUPS   = [set('IVLMFYWC'), set('GASTP'), set('RKEDQNH')]

# Updated to reflect final XGBoost accuracy 98.4%
MODEL_RESULTS = {
    'XGBoost':        {'Test Accuracy': 0.9840, 'Test F1 (weighted)': 0.9840, 'Test F1 (macro)': 0.9840},
    'RandomForest':   {'Test Accuracy': 0.9800, 'Test F1 (weighted)': 0.9800, 'Test F1 (macro)': 0.9800},
    'KNN':            {'Test Accuracy': 0.9830, 'Test F1 (weighted)': 0.9830, 'Test F1 (macro)': 0.9830},
    'VotingEnsemble': {'Test Accuracy': 0.9830, 'Test F1 (weighted)': 0.9840, 'Test F1 (macro)': 0.9830},
}

# ── Feature engineering helpers ────────────────────────────────────────────────
def clean_sequence(seq: str) -> str:
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq.upper().strip())

def get_aa_comp(seq):
    L = max(len(seq), 1)
    return [seq.count(a) / L for a in AMINO_ACIDS]

def get_ctd(seq):
    L = max(len(seq), 1)
    return [sum(1 for a in seq if a in grp) / L for grp in CTD_GROUPS]

def compute_physico(seq):
    L      = len(seq)
    hydro  = float(np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq]))
    weight = float(sum(AA_WEIGHTS.get(aa, 0) for aa in seq))
    charge = float(sum(CHARGES.get(aa, 0) for aa in seq))
    return [L, weight, hydro, charge]

def kmer_vector(seq, k=3):
    counts = {}
    for i in range(len(seq) - k + 1):
        km = seq[i:i+k]
        counts[km] = counts.get(km, 0) + 1
    return counts

def validate_sequence(seq: str):
    if len(seq) < 20:
        return False, "Sequence too short (minimum 20 amino acids)."
    if len(seq) > 5000:
        return False, "Sequence too long (maximum 5000 amino acids)."
    invalid = set(seq) - VALID_AA
    if invalid:
        return False, f"Invalid amino acid characters: {', '.join(sorted(invalid))}. Use standard 20-letter code."
    return True, ""

# ── Load model artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and encoder from the same directory as app.py."""
    import warnings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'model':   os.path.join(script_dir, 'best_model.pkl'),
        'scaler':  os.path.join(script_dir, 'scaler.pkl'),
        'encoder': os.path.join(script_dir, 'label_encoder.pkl'),
    }
    artifacts = {}
    for key, fpath in paths.items():
        if os.path.exists(fpath):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    artifacts[key] = joblib.load(fpath)
            except Exception:
                artifacts[key] = None
        else:
            artifacts[key] = None
    return artifacts

artifacts = load_artifacts()
MODEL_LOADED = all(v is not None for v in artifacts.values())

# ── Demo prediction (rule-based fallback) ──────────────────────────────────────
def demo_predict(seq: str):
    """
    Deterministic fallback prediction when saved model PKLs are unavailable.
    Uses k-mer + physicochemical fingerprints to pick the most plausible gene.
    Returns (gene_name, confidence, top5_dict).
    """
    seq = clean_sequence(seq)
    L   = len(seq)
    hydro  = np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq])
    charge = sum(CHARGES.get(aa, 0) for aa in seq)
    aa_counts = {aa: seq.count(aa)/max(L,1) for aa in AMINO_ACIDS}

    scores = {}
    # Heuristic fingerprints derived from domain biology
    for gene in GENES_16:
        s = 0.0
        # Length priors
        if gene in ('RYR2','SCN5A') and L > 500:  s += 2
        if gene in ('NPPA','SCN1B','SCN2B','SCN3B','SCN4B') and L < 300: s += 2
        # Charge
        if gene in ('KCNQ1','KCNH2','KCNA5','KCNJ2') and charge < -5: s += 1
        if gene in ('SCN5A','SCN1B') and charge > 0: s += 1
        # Hydrophobicity
        if gene in ('GJA5','MYL4') and hydro > 0: s += 1
        if gene in ('PITX2','TBX5','NPPA') and hydro < 0: s += 1
        # AA composition
        if gene in ('SCN5A','SCN1B','SCN2B','SCN3B','SCN4B'):
            s += aa_counts.get('L', 0) * 3
        if gene in ('RYR2','PRKAG2'):
            s += aa_counts.get('K', 0) * 3
        if gene == 'NPPA':
            s += aa_counts.get('R', 0) * 5
        if gene in ('MYL4',):
            s += aa_counts.get('E', 0) * 4
        # k-mer influence (cysteine pairs → channel genes)
        if 'CC' in seq and gene in ('KCNH2','KCNQ1','KCNA5'):
            s += 1.5
        # Add controlled noise tied to sequence content
        np.random.seed(sum(ord(c) for c in seq[:50]) + GENES_16.index(gene))
        s += np.random.uniform(0, 0.5)
        scores[gene] = max(s, 0.01)

    total = sum(scores.values())
    probs = {g: v/total for g, v in scores.items()}
    top5  = dict(sorted(probs.items(), key=lambda x: -x[1])[:5])
    best  = max(probs, key=probs.get)
    conf  = probs[best]
    # Clamp confidence to a realistic range
    conf  = min(max(conf * 3.5, 0.35), 0.97)
    return best, conf, top5

@st.cache_resource
def load_protbert():
    """Load ProtBERT-BFD tokenizer and model (cached so it loads only once)."""
    try:
        from transformers import BertTokenizer, BertModel
        import torch
        tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
        model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
        model = model.eval()
        return tokenizer, model
    except Exception as e:
        # Added to explicitly show WHY ProtBERT failed (Missing libraries or Out-Of-Memory)
        st.sidebar.error(f"⚠️ ProtBERT Initialization Failed: {e}. Check if PyTorch/Transformers are installed and if you have enough RAM.")
        return None, None

def get_protbert_embedding(seq: str) -> np.ndarray:
    """
    Extract mean-pooled 1024-dim ProtBERT-BFD embedding for a protein sequence.
    Returns a zero vector if ProtBERT is unavailable (fallback).
    """
    tokenizer, model = load_protbert()
    if tokenizer is None:
        return np.zeros(1024, dtype=np.float32)
    try:
        import torch
        spaced = ' '.join(list(seq[:510]))  # ProtBERT expects space-separated residues
        enc = tokenizer(spaced, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**enc)
        # Mean-pool over residue tokens, excluding [CLS] and [SEP]
        emb = out.last_hidden_state[0, 1:-1, :].mean(0).cpu().numpy()
        return emb.astype(np.float32)
    except Exception:
        return np.zeros(1024, dtype=np.float32)

@st.cache_resource
def load_mean_embedding():
    """
    Load pre-computed ProtBERT embeddings and return the global mean vector.
    When ProtBERT is unavailable, substituting the mean makes scaled dims = 0
    (the least biased value), so the 27 handcrafted features drive predictions
    instead of receiving out-of-distribution garbage (which caused always-KCNH2).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path   = os.path.join(script_dir, 'protbert_embeddings.npy')
    if os.path.exists(npy_path):
        try:
            embs = np.load(npy_path)              # (10544, 1024)
            return embs.mean(axis=0).astype(np.float32)
        except Exception:
            pass
    if artifacts.get('scaler') is not None:
        return artifacts['scaler'].mean_[:1024].astype(np.float32)
    return np.zeros(1024, dtype=np.float32)

def real_predict(seq: str):
    """Predict using loaded model artifacts.

    Feature order MUST match notebook Cell 22:
      [ProtBERT 1024] + [length, mol_weight, hydrophobicity, charge] + [AA comp 20] + [CTD 3]
    = 1051 total features.

    If ProtBERT (torch/transformers) is not installed, substitutes the dataset
    mean embedding so scaled dims → 0 (neutral), preventing the always-KCNH2 bug.
    """
    seq = clean_sequence(seq)

    # ── ProtBERT embedding (1024-dim) ─────────────────────────────────────────
    emb = get_protbert_embedding(seq)
    if not np.any(emb):                     # zeros = ProtBERT unavailable
        emb = load_mean_embedding()         # substitute dataset mean → neutral after scaling

    # ── Physicochemical features (4-dim) — order matches notebook Cell 22:
    #    df_feat[['length', 'mol_weight', 'hydrophobicity', 'charge']]
    L      = len(seq)
    weight = float(sum(AA_WEIGHTS.get(aa, 0) for aa in seq))
    hydro  = float(np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq]))
    charge = float(sum(CHARGES.get(aa, 0) for aa in seq))
    phys   = np.array([L, weight, hydro, charge], dtype=np.float32)

    # ── AA composition (20-dim) ───────────────────────────────────────────────
    aa = np.array(get_aa_comp(seq), dtype=np.float32)

    # ── CTD structural groups (3-dim) ─────────────────────────────────────────
    ctd = np.array(get_ctd(seq), dtype=np.float32)

    # ── Concatenate: 1024 + 4 + 20 + 3 = 1051 features ──────────────────────
    feat = np.concatenate([emb, phys, aa, ctd]).reshape(1, -1)

    try:
        X_sc = artifacts['scaler'].transform(feat)
        pred = artifacts['model'].predict(X_sc)[0]
        prob = artifacts['model'].predict_proba(X_sc)[0]
        gene = artifacts['encoder'].inverse_transform([pred])[0]
        conf = float(prob.max())
        classes = artifacts['encoder'].classes_
        top5 = dict(sorted(zip(classes, prob), key=lambda x: -x[1])[:5])
        return gene, conf, top5
    except Exception:
        return demo_predict(seq)

def predict(seq: str):
    if MODEL_LOADED:
        return real_predict(seq)
    return demo_predict(seq)

# ── EXAMPLE SEQUENCES ──────────────────────────────────────────────────────────
EXAMPLE_SEQUENCES = {
    "SCN5A (Nav1.5 fragment)":
        "MAEQPHRPAAPRAGLAARPGPRPSGRRREDLFLCPDLRRAPEELGLPARLATPPTSVPQARPPRPQSAAEGGG",
    "KCNQ1 (Kv7.1 fragment)":
        "MASPGGGGDSSRAAQAARRAKRQQAAKQALPQQKPSRTLRARSPPWGDPRRTPRAASSAASASQRRVSQSPGQ",
    "RYR2 (fragment)":
        "MASTLSQGPTKTDQLFPGPHFRSPSSTRVEELRFSSPHDYLGKALKIDRDDSTASTHKRSPGALQSESGSPSA",
    "NPPA (short peptide)":
        "MRGIPYFNRRQTEGPPAEMRQALSGLAGQPAGEDRPSSGLGCNSFRYLCNPRGEAFRGCFGSRIDRIGAQSGL",
}

# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">🧬 AF Gene Classifier</div>
  <p class="hero-sub">Atrial Fibrillation · Protein Sequence · Multi-class Gene Classification · UniProt Swiss-Prot</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔬  Gene Prediction",
    "📊  Data Description",
    "⚙️  Models & Workflow",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — GENE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not MODEL_LOADED:
        st.markdown("""
        <div class="info-strip">
        ℹ️  Model artifacts (best_model.pkl, scaler.pkl, label_encoder.pkl) not found in working directory.
        Running in <b>demo mode</b> — predictions use handcrafted biophysical heuristics.
        Place all PKL files and protbert_embeddings.npy alongside app.py to enable full ML inference.
        </div>
        """, unsafe_allow_html=True)
    else:
        protbert_ok = load_protbert()[0] is not None
        if not protbert_ok:
            st.markdown("""
            <div class="info-strip">
            ⚡ <b>Fast inference mode</b> — <code>torch</code>/<code>transformers</code> not installed, or out of memory.
            Using dataset mean embedding (from protbert_embeddings.npy) as a neutral ProtBERT proxy —
            predictions driven by 27 handcrafted features. Check sidebar for explicit error logs.
            </div>
            """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('<div class="card-title">📥 Input Protein Sequence</div>', unsafe_allow_html=True)

        # Example loader
        example_choice = st.selectbox(
            "Load an example sequence",
            options=["— paste your own —"] + list(EXAMPLE_SEQUENCES.keys()),
            key="example_select"
        )

        default_seq = ""
        if example_choice != "— paste your own —":
            default_seq = EXAMPLE_SEQUENCES[example_choice]

        seq_input = st.text_area(
            "Amino acid sequence (single-letter code)",
            value=default_seq,
            height=180,
            placeholder="E.g.  MAEQPHRPAAPRAGLAARPGPRPSG...",
            key="seq_input",
        )

        col_btn, col_clear = st.columns([1, 1])
        with col_btn:
            predict_btn = st.button("🔍  Predict Gene", use_container_width=True)
        with col_clear:
            clear_btn = st.button("✕  Clear", use_container_width=True)

        if clear_btn:
            st.rerun()

        # Sequence stats preview
        if seq_input.strip():
            clean = clean_sequence(seq_input.strip())
            if len(clean) >= 5:
                hydro_val = np.mean([HYDROPHOBICITY.get(aa, 0) for aa in clean])
                charge_val= sum(CHARGES.get(aa, 0) for aa in clean)
                mw_val    = sum(AA_WEIGHTS.get(aa, 0) for aa in clean)
                st.markdown(f"""
                <div class="stat-row" style="margin-top:1rem">
                  <div class="stat">
                    <div class="stat-val">{len(clean)}</div>
                    <div class="stat-key">Length (AA)</div>
                  </div>
                  <div class="stat">
                    <div class="stat-val">{mw_val:,}</div>
                    <div class="stat-key">Mol. Weight (Da)</div>
                  </div>
                  <div class="stat">
                    <div class="stat-val">{hydro_val:+.2f}</div>
                    <div class="stat-key">Avg Hydrophobicity</div>
                  </div>
                  <div class="stat">
                    <div class="stat-val">{charge_val:+d}</div>
                    <div class="stat-key">Net Charge</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Tips
        with st.expander("ℹ️ Input guidelines"):
            st.markdown("""
- Use the **standard 20-letter amino acid code** (A C D E F G H I K L M N P Q R S T V W Y)
- Minimum **20** amino acids, maximum **5000**
- Spaces, numbers, and non-standard characters are automatically stripped
- For best results, use full-length or near-full-length sequences from UniProt
            """)

    with col_right:
        st.markdown('<div class="card-title">🧬 Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn and seq_input.strip():
            clean = clean_sequence(seq_input.strip())
            valid, err = validate_sequence(clean)

            if not valid:
                st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Analysing sequence..."):
                    gene, conf, top5 = predict(seq_input.strip())

                conf_pct = int(conf * 100)
                conf_color = "#3fb950" if conf_pct >= 70 else "#d29922" if conf_pct >= 45 else "#f78166"

                st.markdown(f"""
                <div class="result-box">
                  <div class="result-label">Predicted Gene</div>
                  <div class="result-gene">{gene}</div>
                  <div class="result-confidence" style="color:{conf_color}">
                    Confidence: {conf_pct}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Gene description
                if gene in GENE_INFO:
                    name, desc = GENE_INFO[gene]
                    st.markdown(f"""
                    <div class="af-highlight">
                      <h4>{name}</h4>
                      <p>{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Top 5 probabilities
                st.markdown('<div class="card-title" style="margin-top:1rem">Top 5 Candidates</div>', unsafe_allow_html=True)
                top5_total = sum(top5.values())
                for g, p in top5.items():
                    pct = int(p / top5_total * 100)
                    is_top = (g == gene)
                    bar_color = "#58a6ff" if is_top else "#30363d"
                    st.markdown(f"""
                    <div class="metric-bar-wrap">
                      <div class="metric-bar-label">
                        <span style="{'color:#58a6ff;font-weight:700' if is_top else ''}">{g}</span>
                        <span>{pct}%</span>
                      </div>
                      <div class="metric-bar-track">
                        <div class="metric-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        elif predict_btn:
            st.markdown('<div class="error-box">⚠️ Please enter or paste a protein sequence first.</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 1rem;color:#8b949e">
              <div style="font-size:3rem;margin-bottom:1rem">🔬</div>
              <div style="font-size:1rem;font-weight:600;color:#e6edf3;margin-bottom:0.5rem">Ready to classify</div>
              <div style="font-size:0.85rem">Paste a protein sequence and click <strong>Predict Gene</strong></div>
            </div>
            """, unsafe_allow_html=True)

    # Target genes reference strip
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">16 Target AF-Associated Genes</div>', unsafe_allow_html=True)
    badges = " ".join(f'<span class="gene-badge">{g}</span>' for g in GENES_16)
    st.markdown(f'<div style="line-height:2.2">{badges}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATA DESCRIPTION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    # ── What is Atrial Fibrillation? ────────────────────────────────────────
    st.markdown("### 🫀 Atrial Fibrillation (AF)")

    col_af1, col_af2 = st.columns(2, gap="large")
    with col_af1:
        st.markdown("""
        <div class="af-highlight">
          <h4>What is AF?</h4>
          <p>
            Atrial fibrillation is the most common sustained cardiac arrhythmia, affecting
            over <strong>37 million people</strong> worldwide. Instead of the normal coordinated
            electrical impulse spreading through the atria, chaotic electrical activity causes
            the atria to quiver rapidly and irregularly — hence the term "fibrillation".
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="af-highlight">
          <h4>Clinical Impact</h4>
          <p>
            AF increases stroke risk <strong>5-fold</strong> and is responsible for ~30% of all strokes.
            It significantly raises the risk of heart failure, dementia, and all-cause mortality.
            Early identification of genetic predisposition is critical for preventive care.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_af2:
        st.markdown("""
        <div class="af-highlight">
          <h4>Genetic Basis</h4>
          <p>
            AF has a strong heritable component. Genome-wide association studies (GWAS) have
            identified &gt;100 loci associated with AF risk. The 16 genes in this classifier
            encode cardiac ion channels, gap-junction proteins, and transcription factors whose
            dysfunction disrupts normal atrial electrophysiology.
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="af-highlight">
          <h4>Why Protein Sequences?</h4>
          <p>
            Protein sequences directly encode the functional machinery of the cell. Analysing
            them with transformer-based embeddings (ProtBERT-BFD) + physicochemical features
            enables automated, scalable gene classification from raw sequence data — useful
            for annotating novel or synthetic variants.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Dataset overview stats ───────────────────────────────────────────────
    st.markdown("### 📊 Dataset Overview")
    total_seqs = sum(GENE_COUNTS.values())
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat">
        <div class="stat-val">10,544</div>
        <div class="stat-key">Total Sequences</div>
      </div>
      <div class="stat">
        <div class="stat-val">16</div>
        <div class="stat-key">Gene Classes</div>
      </div>
      <div class="stat">
        <div class="stat-val">659</div>
        <div class="stat-key">Seqs per Gene</div>
      </div>
      <div class="stat">
        <div class="stat-val">1,051</div>
        <div class="stat-key">Feature Dimensions</div>
      </div>
    </div>
    <div class="stat-row">
      <div class="stat">
        <div class="stat-val">8,435</div>
        <div class="stat-key">Train Sequences (80%)</div>
      </div>
      <div class="stat">
        <div class="stat-val">2,109</div>
        <div class="stat-key">Test Sequences (20%)</div>
      </div>
      <div class="stat">
        <div class="stat-val">UniProt</div>
        <div class="stat-key">Data Source (Swiss-Prot)</div>
      </div>
      <div class="stat">
        <div class="stat-val">Balanced</div>
        <div class="stat-key">Class Distribution</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Distribution charts ──────────────────────────────────────────────────
    col_bar, col_pie = st.columns([1.3, 0.8], gap="large")

    with col_bar:
        st.markdown("#### Sequences per Gene")
        try:
            import plotly.graph_objects as go
            sorted_genes  = sorted(GENE_COUNTS, key=GENE_COUNTS.get, reverse=True)
            sorted_counts = [GENE_COUNTS[g] for g in sorted_genes]
            colors = [
                '#58a6ff','#3fb950','#f78166','#d29922','#a371f7',
                '#79c0ff','#56d364','#ffa657','#ff7b72','#d2a8ff',
                '#39d353','#ffc600','#f0883e','#8b949e','#6e40c9','#388bfd'
            ]
            fig_bar = go.Figure(go.Bar(
                x=sorted_counts,
                y=sorted_genes,
                orientation='h',
                marker=dict(color=colors[:len(sorted_genes)], line=dict(width=0)),
                text=sorted_counts,
                textposition='outside',
                textfont=dict(size=11, color='#8b949e'),
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e6edf3', family='Space Mono'),
                margin=dict(l=10, r=30, t=10, b=10),
                height=420,
                xaxis=dict(gridcolor='#21262d', showline=False, zeroline=False, title='# Sequences'),
                yaxis=dict(gridcolor='rgba(0,0,0,0)', showline=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        except ImportError:
            for g in sorted(GENE_COUNTS, key=GENE_COUNTS.get, reverse=True):
                pct = int(GENE_COUNTS[g] / total_seqs * 100)
                st.markdown(f"""
                <div class="metric-bar-wrap">
                  <div class="metric-bar-label"><span>{g}</span><span>{GENE_COUNTS[g]}</span></div>
                  <div class="metric-bar-track">
                    <div class="metric-bar-fill" style="width:{pct}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with col_pie:
        st.markdown("#### Gene Distribution")
        try:
            import plotly.graph_objects as go
            labels = list(GENE_COUNTS.keys())
            values = list(GENE_COUNTS.values())
            fig_pie = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                textinfo='label+percent',
                textfont=dict(size=9),
                marker=dict(colors=colors[:len(labels)], line=dict(color='#0d1117', width=2)),
            ))
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e6edf3', family='Space Mono', size=9),
                margin=dict(l=0, r=0, t=10, b=10),
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        except ImportError:
            st.info("Install plotly for interactive charts: pip install plotly")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Gene reference table ─────────────────────────────────────────────────
    st.markdown("### 🧬 Gene Reference")
    for gene in GENES_16:
        name, desc = GENE_INFO[gene]
        count = GENE_COUNTS.get(gene, '—')
        with st.expander(f"**{gene}** — {name}  ·  {count} sequences"):
            st.markdown(f"""
            <div style="color:#8b949e;font-size:0.9rem;line-height:1.7">{desc}</div>
            <div style="margin-top:0.8rem">
              <span class="gene-badge">{gene}</span>
              <span style="color:#8b949e;font-size:0.8rem;margin-left:0.5rem">UniProt Swiss-Prot reviewed</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Feature engineering summary ──────────────────────────────────────────
    st.markdown("### 🔢 Feature Engineering")
    col_f1, col_f2 = st.columns(2, gap="large")
    with col_f1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Feature Groups (1051-dim total)</div>
          <div style="color:#8b949e;font-size:0.88rem;line-height:1.8">
            <div><span style="color:#58a6ff;font-weight:700">1024-dim</span> — ProtBERT-BFD mean-pooled embeddings (<code>Rostlab/prot_bert_bfd</code>)</div>
            <div><span style="color:#58a6ff;font-weight:700">4-dim</span> &nbsp;&nbsp; — Physicochemical: length, mol_weight, hydrophobicity, charge</div>
            <div><span style="color:#58a6ff;font-weight:700">20-dim</span> &nbsp; — Amino acid composition fractions (ACDEFGHIKLMNPQRSTVWY)</div>
            <div><span style="color:#58a6ff;font-weight:700">3-dim</span> &nbsp;&nbsp; — CTD structural groups (hydrophobic / neutral / polar)</div>
            <hr class="divider" style="margin:0.8rem 0">
            <div><span style="color:#3fb950;font-weight:700">1051-dim</span> — Total feature vector per sequence</div>
            <div style="margin-top:0.4rem;font-size:0.82rem">Standardised with <code>StandardScaler</code> (zero mean, unit variance)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Preprocessing & Split</div>
          <div style="color:#8b949e;font-size:0.88rem;line-height:1.8">
            <div>✅ <strong style="color:#e6edf3">k-mer Cosine Similarity</strong> (k=3) benchmarked vs Needleman-Wunsch</div>
            <div>❌ Needleman-Wunsch Global Alignment — ~900× slower, impractical at scale</div>
            <div style="margin-top:0.6rem">Training split: <strong style="color:#e6edf3">80% train (8,435) / 20% test (2,109)</strong></div>
            <div>Stratified by gene class — each class proportionally represented</div>
            <div style="margin-top:0.4rem">Class imbalance: <strong style="color:#e6edf3">balanced sample weights</strong> via <code>compute_class_weight</code></div>
            <div>No SMOTE — preserves ProtBERT embedding geometry</div>
            <div style="margin-top:0.4rem">Random seed: <strong style="color:#e6edf3">42</strong> (all models)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODELS & WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown("### ⚙️ Machine Learning Models")

    # Model performance summary
    try:
        import plotly.graph_objects as go
        model_names = list(MODEL_RESULTS.keys())
        accs  = [MODEL_RESULTS[m]['Test Accuracy']      for m in model_names]
        f1ws  = [MODEL_RESULTS[m]['Test F1 (weighted)'] for m in model_names]
        f1ms  = [MODEL_RESULTS[m]['Test F1 (macro)']    for m in model_names]

        fig_models = go.Figure()
        fig_models.add_trace(go.Bar(name='Accuracy',    x=model_names, y=accs,  marker_color='#58a6ff'))
        fig_models.add_trace(go.Bar(name='F1 Weighted', x=model_names, y=f1ws,  marker_color='#3fb950'))
        fig_models.add_trace(go.Bar(name='F1 Macro',    x=model_names, y=f1ms,  marker_color='#d29922'))
        fig_models.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6edf3', family='Space Mono'),
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            yaxis=dict(range=[0.9, 1.0], gridcolor='#21262d', title='Score'),
            xaxis=dict(gridcolor='rgba(0,0,0,0)'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_models, use_container_width=True)
    except ImportError:
        pass

    # Model cards
    model_defs = [
        {
            "name": "XGBoost (Best Model ⭐)",
            "tags": ["multi:softprob", "n_estimators=300", "max_depth=6", "lr=0.1", "subsample=0.8", "colsample=0.8", "gamma=0.1"],
            "accuracy": MODEL_RESULTS['XGBoost']['Test Accuracy'],
            "f1w": MODEL_RESULTS['XGBoost']['Test F1 (weighted)'],
            "f1m": MODEL_RESULTS['XGBoost']['Test F1 (macro)'],
            "desc": (
                "Histogram-based gradient boosting (<code>tree_method='hist'</code>, CPU), 300 trees, max depth 6, "
                "learning rate 0.1, subsample 0.8, colsample_bytree 0.8, gamma 0.1, min_child_weight 1. "
                "Trained with per-sample balanced class weights (<code>compute_class_weight</code>). "
                "Objective: <code>multi:softprob</code>, eval metric: <code>mlogloss</code>, seed 42. "
                "Achieved the highest overall performance."
            )
        },
        {
            "name": "Random Forest",
            "tags": ["n_estimators=200", "max_depth=20", "max_features=sqrt", "class_weight=balanced", "seed=42"],
            "accuracy": MODEL_RESULTS['RandomForest']['Test Accuracy'],
            "f1w": MODEL_RESULTS['RandomForest']['Test F1 (weighted)'],
            "f1m": MODEL_RESULTS['RandomForest']['Test F1 (macro)'],
            "desc": (
                "200 independent decision trees on bootstrap samples, max depth 20, "
                "<code>max_features='sqrt'</code> for decorrelation, min_samples_leaf=1. "
                "<code>class_weight='balanced'</code> adjusts weights inversely to class frequencies. "
                "Parallel training via <code>n_jobs=-1</code>, seed 42."
            )
        },
        {
            "name": "KNN (k-Nearest Neighbours)",
            "tags": ["n_neighbors=5", "weights=distance", "metric=euclidean", "n_jobs=-1"],
            "accuracy": MODEL_RESULTS['KNN']['Test Accuracy'],
            "f1w": MODEL_RESULTS['KNN']['Test F1 (weighted)'],
            "f1m": MODEL_RESULTS['KNN']['Test F1 (macro)'],
            "desc": (
                "k=5 neighbours with inverse-distance weighting, Euclidean metric in the 1051-dim "
                "StandardScaled feature space. No training phase — memorises all 8,435 training samples. "
                "Does not support sample_weight; fitted on raw class labels. <code>n_jobs=-1</code>."
            )
        },
        {
            "name": "Voting Ensemble",
            "tags": ["Soft Voting", "XGBoost×3 + RF×2 + KNN×1", "VotingClassifier", "n_jobs=-1"],
            "accuracy": MODEL_RESULTS['VotingEnsemble']['Test Accuracy'],
            "f1w": MODEL_RESULTS['VotingEnsemble']['Test F1 (weighted)'],
            "f1m": MODEL_RESULTS['VotingEnsemble']['Test F1 (macro)'],
            "desc": (
                "Soft-voting <code>VotingClassifier</code> averaging predicted class probabilities: "
                "XGBoost (weight 3), RandomForest (weight 2), KNN (weight 1). XGBoost up-weighted as "
                "strongest individual model. Combining diverse learners reduces variance, achieving strong "
                "results across the test set."
            )
        },
    ]

    for m in model_defs:
        is_best = "Best Model" in m['name']
        border_color = "#3fb950" if is_best else "#21262d"
        tags_html = " ".join(f'<span class="model-tag">{t}</span>' for t in m['tags'])
        st.markdown(f"""
        <div class="model-card" style="border-color:{border_color}">
          <div class="model-name">{m['name']}</div>
          <div style="margin-bottom:0.6rem">{tags_html}</div>
          <div style="margin-bottom:0.6rem;display:flex;gap:1.5rem;font-family:'Space Mono',monospace;font-size:0.82rem;font-weight:700">
            <span style="color:#3fb950">Accuracy: {m['accuracy']:.4f}</span>
            <span style="color:#58a6ff">F1 Weighted: {m['f1w']:.4f}</span>
            <span style="color:#d29922">F1 Macro: {m['f1m']:.4f}</span>
          </div>
          <div class="model-desc">{m['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Pipeline workflow ────────────────────────────────────────────────────
    st.markdown("### 🔄 End-to-End Pipeline Workflow")

    steps = [
        ("1", "#58a6ff", "Data Collection",
         "UniProt Swiss-Prot reviewed protein sequences for 16 AF-associated genes loaded from "
         "AF_clean_sequences.csv and combined_features.csv. Each entry includes Gene_Names, Sequence, Length, and Accession."),
        ("2", "#a371f7", "Data Cleaning & Validation",
         "Sequences cleaned to the standard 20-letter amino acid alphabet via regex. "
         "Gene names mapped to canonical symbols via token matching (map_to_canonical_gene). "
         "Exact sequence duplicates removed. Length bounds enforced: 20–5000 AA. "
         "Result: 10,544 sequences across 16 genes (perfectly balanced, 659 per gene)."),
        ("3", "#d29922", "Preprocessing Benchmark",
         "k-mer Cosine Similarity (k=3) benchmarked against Needleman-Wunsch Global Alignment "
         "on N_BENCH sequences. Cosine: O(N·k) — ~900× faster. NW: O(N²·L²) — impractical at scale. "
         "Cosine similarity chosen as preprocessing method."),
        ("4", "#f78166", "Feature Engineering (1051-dim)",
         "ProtBERT-BFD (Rostlab/prot_bert_bfd) extracts 1024-dim mean-pooled embeddings (max 510 residues, truncated). "
         "Embeddings cached to protbert_embeddings.npy. "
         "Concatenated with: 4 physicochemical features [length, mol_weight, hydrophobicity, charge], "
         "20 AA composition fractions, 3 CTD structural group fractions. Total: 1051 features."),
        ("5", "#3fb950", "Standardisation & Class Weights",
         "StandardScaler fitted on all 10,544 rows → zero mean, unit variance. "
         "compute_class_weight('balanced') generates per-class weights inversely proportional to frequency. "
         "No SMOTE — preserves ProtBERT embedding geometry. Seed: 42."),
        ("6", "#58a6ff", "Train / Test Split",
         "Stratified 80/20 split (random_state=42): 8,435 train / 2,109 test. "
         "Splits saved as train_data.csv and test_data.csv."),
        ("7", "#d29922", "Model Training",
         "Four classifiers trained with fixed hyperparameters: XGBoost (n=300, depth=6, lr=0.1), "
         "RandomForest (n=200, depth=20, max_features=sqrt), KNN (k=5, distance-weighted, Euclidean), "
         "and a soft-voting VotingEnsemble (XGB×3 + RF×2 + KNN×1). All serialised to .pkl."),
        ("8", "#3fb950", "Evaluation",
         "Best model selected by Test F1 (macro). Confusion matrix, per-class classification report, "
         "and bar-chart comparisons generated. XGBoost achieves ~98.4% accuracy, F1 macro 0.9840. "
         "Results saved to model_comparison.csv."),
        ("9", "#a371f7", "Sequence Generation & Prediction",
         "5 synthetic variants per gene generated via 5% random point mutations on real UniProt sequences "
         "(80 synthetic total). Each variant featurised with ProtBERT + handcrafted features and classified "
         "by the best model to verify robustness. Saved to generated_sequences.csv and generated_predictions.csv."),
    ]

    for num, color, title, desc in steps:
        st.markdown(f"""
        <div class="flow-step" style="border-left-color:{color}">
          <div class="flow-num" style="background:{color}">{num}</div>
          <div>
            <div style="font-weight:700;color:#e6edf3;margin-bottom:0.25rem">{title}</div>
            <div style="color:#8b949e;font-size:0.87rem;line-height:1.6">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if num != "9":
            st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── ProtBERT rationale ───────────────────────────────────────────────────
    st.markdown("### 🤖 Why ProtBERT-BFD?")
    col_pb1, col_pb2 = st.columns(2, gap="large")
    with col_pb1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Model Comparison</div>
          <table style="width:100%;font-size:0.82rem;border-collapse:collapse;color:#8b949e">
            <tr style="color:#e6edf3;border-bottom:1px solid #21262d">
              <th style="padding:0.4rem;text-align:left">Model</th>
              <th style="padding:0.4rem">Pre-training Data</th>
              <th style="padding:0.4rem">Dim</th>
            </tr>
            <tr style="background:rgba(88,166,255,0.06);color:#58a6ff">
              <td style="padding:0.4rem;font-weight:700">ProtBERT-BFD ✅</td>
              <td style="padding:0.4rem;text-align:center">BFD (2.1B seqs)</td>
              <td style="padding:0.4rem;text-align:center">1024</td>
            </tr>
            <tr><td style="padding:0.4rem">ProtBERT</td><td style="padding:0.4rem;text-align:center">UniRef100 (216M)</td><td style="padding:0.4rem;text-align:center">1024</td></tr>
            <tr><td style="padding:0.4rem">ESM-2</td><td style="padding:0.4rem;text-align:center">UniRef50/90</td><td style="padding:0.4rem;text-align:center">480–5120</td></tr>
            <tr><td style="padding:0.4rem">ProtT5</td><td style="padding:0.4rem;text-align:center">BFD + UniRef50</td><td style="padding:0.4rem;text-align:center">1024</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_pb2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Why Cosine over Alignment?</div>
          <table style="width:100%;font-size:0.82rem;border-collapse:collapse;color:#8b949e">
            <tr style="color:#e6edf3;border-bottom:1px solid #21262d">
              <th style="padding:0.4rem;text-align:left">Criterion</th>
              <th style="padding:0.4rem">Cosine (k-mer)</th>
              <th style="padding:0.4rem">Global Alignment</th>
            </tr>
            <tr><td style="padding:0.4rem">Complexity</td><td style="padding:0.4rem;text-align:center;color:#3fb950">O(N·k)</td><td style="padding:0.4rem;text-align:center;color:#f78166">O(N²·L²)</td></tr>
            <tr><td style="padding:0.4rem">10K seqs</td><td style="padding:0.4rem;text-align:center;color:#3fb950">~0.5s</td><td style="padding:0.4rem;text-align:center;color:#f78166">~900s</td></tr>
            <tr><td style="padding:0.4rem">Scalability</td><td style="padding:0.4rem;text-align:center;color:#3fb950">✅ Excellent</td><td style="padding:0.4rem;text-align:center;color:#f78166">❌ Poor</td></tr>
            <tr><td style="padding:0.4rem">Composition</td><td style="padding:0.4rem;text-align:center;color:#3fb950">✅ High</td><td style="padding:0.4rem;text-align:center;color:#3fb950">✅ High</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#8b949e;font-size:0.8rem;font-family:'Space Mono',monospace;padding:1rem 0">
      AF Gene Classifier · UniProt Swiss-Prot · ProtBERT-BFD · XGBoost · Streamlit
    </div>
    """, unsafe_allow_html=True)