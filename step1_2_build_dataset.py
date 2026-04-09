"""
SLD Risk Prediction Benchmark
Step 1: Build and Merge Dataset
Step 2: Create Labels

Dataset: NHANES 2017-2018
Author: Sachin Parajuli
Course: CSE 6361

Required files (place in same directory or update DATA_DIR):
    DEMO_J.xpt, BMX_J.xpt, BPX_J.xpt, BIOPRO_J.xpt,
    GLU_J.xpt, GHB_J.xpt, ALQ_J.xpt, LUX_J.xpt

Output:
    sld_dataset.csv  — merged, labeled, reliability-filtered dataset
"""

import pandas as pd
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR  = "."          # folder containing all .xpt files
OUTPUT    = "sld_dataset.csv"

IQR_THRESHOLD   = 0.30   # drop FibroScan rows where IQR/median >= 0.30
CAP_THRESHOLD   = 248    # CAP >= 248 dB/m  → steatosis positive
LSM_THRESHOLD   = 8.0    # LSM >= 8.0 kPa   → fibrosis positive


# ── Helper ───────────────────────────────────────────────────────────────────
def load(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_sas(path, format='xport', encoding='utf-8')
    print(f"  Loaded {filename}: {df.shape}")
    return df


# ── Step 1: Load & Select Columns ────────────────────────────────────────────
print("\n[Step 1] Loading data files...")

# 1. FibroScan (labels source)
lux = load("LUX_J.xpt")
lux = lux[['SEQN', 'LUXSMED', 'LUXSIQR', 'LUXCAPM']].copy()

# 2. Demographics
demo = load("DEMO_J.xpt")
demo = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR']].copy()
demo.columns = ['SEQN', 'AGE', 'GENDER', 'RACE', 'POVERTY_RATIO']

# 3. Anthropometrics
bmx = load("BMX_J.xpt")
bmx = bmx[['SEQN', 'BMXBMI', 'BMXWAIST']].copy()
bmx.columns = ['SEQN', 'BMI', 'WAIST_CM']

# 4. Blood Pressure — average across up to 4 readings
bpx = load("BPX_J.xpt")
bpx['SYS_BP'] = bpx[['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']].mean(axis=1)
bpx['DIA_BP'] = bpx[['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']].mean(axis=1)
bpx = bpx[['SEQN', 'SYS_BP', 'DIA_BP']].copy()

# 5. Biochemistry
#    Note: ALT and HDL not present in BIOPRO_J for this cycle — skipped.
biopro = load("BIOPRO_J.xpt")
biopro = biopro[[
    'SEQN',
    'LBXSATSI',   # AST
    'LBXSAL',     # Albumin
    'LBXSAPSI',   # ALP (Alkaline Phosphatase)
    'LBXSGTSI',   # GGT
    'LBXSTB',     # Total Bilirubin
    'LBXSTP',     # Total Protein
    'LBXSGB',     # Globulin
    'LBXSCH',     # Total Cholesterol
    'LBXSTR',     # Triglyceride
]].copy()
biopro.columns = [
    'SEQN', 'AST', 'ALBUMIN', 'ALP', 'GGT',
    'TOTAL_BILIRUBIN', 'TOTAL_PROTEIN', 'GLOBULIN',
    'TOTAL_CHOLESTEROL', 'TRIGLYCERIDE'
]

# 6. Fasting Glucose
glu = load("GLU_J.xpt")
glu = glu[['SEQN', 'LBXGLU']].copy()
glu.columns = ['SEQN', 'GLUCOSE']

# 7. HbA1c
ghb = load("GHB_J.xpt")
ghb = ghb[['SEQN', 'LBXGH']].copy()
ghb.columns = ['SEQN', 'HBAC1']

# 8. Alcohol Use
#    ALQ111 = ever drank (1=yes, 2=no); ALQ130 = avg drinks per day
#    ALQ101 not present in this cycle — ALQ111 used as equivalent.
alq = load("ALQ_J.xpt")
alq = alq[['SEQN', 'ALQ111', 'ALQ130']].copy()

def alcohol_category(row):
    """
    Categorize alcohol use:
        0 = non-drinker
        1 = light    (<= 1 drink/day)
        2 = moderate (<= 2 drinks/day)
        3 = heavy    (> 2 drinks/day)
    Returns NaN for coded missing values (777, 999).
    """
    if row['ALQ111'] == 2:          # never drank
        return 0
    if pd.isna(row['ALQ130']):      # drank but frequency missing → light
        return 1
    x = row['ALQ130']
    if x >= 777:                    # coded missing
        return np.nan
    if x == 0:
        return 0
    elif x <= 1:
        return 1
    elif x <= 2:
        return 2
    else:
        return 3

alq['ALCOHOL_CAT'] = alq.apply(alcohol_category, axis=1)
alq = alq[['SEQN', 'ALCOHOL_CAT']]


# ── Merge all files on SEQN ──────────────────────────────────────────────────
print("\n[Step 1] Merging all files on SEQN...")

df = lux.copy()
for other in [demo, bmx, bpx, biopro, glu, ghb, alq]:
    df = df.merge(other, on='SEQN', how='left')

print(f"  Shape after merge: {df.shape}")


# ── Step 1 (cont): Filter unreliable FibroScan measurements ─────────────────
print("\n[Step 1] Filtering unreliable FibroScan measurements...")
print(f"  Rows before filter : {len(df)}")

df['IQR_RATIO'] = df['LUXSIQR'] / df['LUXSMED']
df = df[df['IQR_RATIO'] < IQR_THRESHOLD].copy()
df.drop(columns=['IQR_RATIO'], inplace=True)

print(f"  Rows after filter  : {len(df)}  (IQR/median < {IQR_THRESHOLD})")


# ── Step 2: Create Binary Labels ─────────────────────────────────────────────
print("\n[Step 2] Creating binary labels...")

df['y_steatosis'] = (df['LUXCAPM'] >= CAP_THRESHOLD).astype(int)
df['y_fibrosis']  = (df['LUXSMED'] >= LSM_THRESHOLD).astype(int)

print(f"  y_steatosis — positive: {df['y_steatosis'].sum()} "
      f"/ {len(df)} ({df['y_steatosis'].mean()*100:.1f}%)")
print(f"  y_fibrosis  — positive: {df['y_fibrosis'].sum()} "
      f"/ {len(df)} ({df['y_fibrosis'].mean()*100:.1f}%)")

# Drop raw FibroScan measurement columns (labels already created)
df.drop(columns=['LUXSMED', 'LUXSIQR', 'LUXCAPM'], inplace=True)


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n[Summary]")
print(f"  Final shape  : {df.shape}")
print(f"  Columns      : {df.columns.tolist()}")
print(f"\n  Missing values per column:")
print(df.isnull().sum().to_string())


# ── Save ─────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)
print(f"\n✅  Saved: {OUTPUT}")
