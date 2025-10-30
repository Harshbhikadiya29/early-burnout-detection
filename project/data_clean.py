# project/data_clean.py
import re
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw/Impact_of_Remote_Work_on_Mental_Health.csv")
OUT = Path("data/processed/burnout_demo.csv")

# ---- helpers ---------------------------------------------------------------
def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def choose_col(df, candidates):
    """Pick the first column whose normalized name matches any candidate pattern."""
    norm_map = {c: _norm(c) for c in df.columns}
    inv = {v: k for k, v in norm_map.items()}
    for patt in candidates:
        # exact or contains
        for norm_name in norm_map.values():
            if norm_name == patt or patt in norm_name:
                return inv[norm_name]
    return None

def to_1_10(val):
    """Map string or numeric to a 1..10 scale."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        # assume already 1..10 or 1..5; scale 1..5 to ~5..9
        v = float(val)
        if 1 <= v <= 10:
            return v
        if 1 <= v <= 5:
            return 4 + v  # 1->5, 5->9 (simple spread)
        return np.nan
    s = str(val).strip().lower()
    m = {
        "very low": 2, "low": 3, "medium": 6, "moderate": 6, "mid": 6,
        "high": 9, "very high": 10, "poor": 3, "fair": 5, "good": 7, "great": 8, "excellent": 9
    }
    # common tokens
    for k, v in m.items():
        if k in s:
            return v
    # direct digits in text
    dig = re.findall(r"\d+", s)
    if dig:
        n = float(dig[0])
        if 1 <= n <= 10:
            return n
        if 1 <= n <= 5:
            return 4 + n
    return np.nan

# ---- main ------------------------------------------------------------------
def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Missing dataset at: {RAW}")

    df = pd.read_csv(RAW)

    # Try to detect columns (robust to naming)
    hours_col     = choose_col(df, ["hours_worked_per_week", "avg_weekly_hours", "hours_worked", "work_hours"])
    wlb_col       = choose_col(df, ["work_life_balance_rating", "work_life_balance", "wlb", "work_life"])
    sleep_col     = choose_col(df, ["sleep_quality", "sleep", "sleep_score"])
    stress_col    = choose_col(df, ["stress_level", "stress", "stress_score"])

    # Build a lightweight free_text from a few common descriptors if available
    role_col      = choose_col(df, ["job_role", "role", "position", "job_title"])
    ind_col       = choose_col(df, ["industry"])
    loc_col       = choose_col(df, ["work_location", "location", "remote_type"])
    support_col   = choose_col(df, ["company_support_for_remote_work", "company_support", "support_remote"])

    missing = [name for name, c in {
        "hours": hours_col, "work_life_balance": wlb_col, "sleep": sleep_col, "stress": stress_col
    }.items() if c is None]
    if missing:
        raise ValueError(f"Could not auto-detect required columns: {missing}\n"
                         f"Found columns: {list(df.columns)}")

    # Feature mapping
    hours = pd.to_numeric(df[hours_col], errors="coerce").fillna(40).clip(lower=0)
    # Workload (1..10) from inverse of work-life balance (assume higher WLB => lower workload)
    wlb_raw = df[wlb_col].apply(to_1_10)
    workload = (11 - wlb_raw).clip(1, 10)  # invert: 10 -> 1 workload, 1 -> 10 workload
    sleep = df[sleep_col].apply(to_1_10).fillna(6).clip(1, 10)
    stress = df[stress_col].apply(to_1_10).fillna(5).clip(1, 10)

    # free_text synthesis (optional)
    def synth(row):
        parts = []
        for c in [role_col, ind_col, loc_col, support_col]:
            if c and pd.notna(row.get(c, np.nan)):
                parts.append(str(row[c]))
        return " | ".join(parts) if parts else ""
    free_text = df.apply(synth, axis=1)

    # Heuristic label (0/1) for a demo training target
    cond = ((stress >= 8) & (sleep <= 5)) | (workload >= 8) | (hours >= 55)
    burnout_label = cond.astype(int)

    tidy = pd.DataFrame({
        "free_text": free_text,
        "hours": hours.astype(float),
        "workload": workload.astype(float),
        "sleep": sleep.astype(float),
        "stress": stress.astype(float),
        "burnout_label": burnout_label.astype(int),
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT, index=False)
    print(f"Wrote: {OUT.resolve()}  rows={len(tidy)}")
    print(tidy.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
