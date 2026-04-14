# src/infer.py
"""
Strava workout type classifier — inference CLI

Usage:
    python src/infer.py --file activities/11147594123.fit.gz
    python src/infer.py --file activities/10078375825.gpx
    python src/infer.py --file activities/11147594123.fit.gz --verbose
"""

import argparse
import gzip
import json
import sys
import warnings
from pathlib import Path

import fitparse
import numpy as np
import pandas as pd
import onnxruntime as rt
import xml.etree.ElementTree as ET

warnings.filterwarnings('ignore')

# ── Constants (same as Phase 1 parsers) ──────────────────────────────────────
SEMICIRCLE_TO_DEG = 180 / (2 ** 31)
RECORD_FIELDS = {
    'timestamp', 'heart_rate', 'cadence', 'fractional_cadence',
    'enhanced_speed', 'enhanced_altitude', 'distance',
    'position_lat', 'position_long',
}
NS = {
    'gpx':    'http://www.topografix.com/GPX/1/1',
    'gpxtpx': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
}
MAX_HR = 176   # from your data — update if your max HR changes


# ── Parsers (copied from Phase 1 — self-contained) ───────────────────────────
def parse_fit(filepath):
    filepath = Path(filepath)
    try:
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rb') as f:
                raw = f.read()
            fit = fitparse.FitFile(raw)
        else:
            fit = fitparse.FitFile(str(filepath))
    except Exception as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        return None

    rows = []
    for msg in fit.get_messages("record"):
        row = {}
        for field in msg:
            if field.name in RECORD_FIELDS and field.value is not None:
                row[field.name] = field.value
        if 'timestamp' in row and 'enhanced_speed' in row:
            rows.append(row)

    if len(rows) < 10:
        return None

    df = pd.DataFrame(rows)
    if 'position_lat' in df.columns:
        df['lat'] = df['position_lat'] * SEMICIRCLE_TO_DEG
        df['lon'] = df['position_long'] * SEMICIRCLE_TO_DEG
        df.drop(columns=['position_lat', 'position_long'], inplace=True)
    if 'cadence' in df.columns and 'fractional_cadence' in df.columns:
        df['cadence'] = df['cadence'] + df['fractional_cadence'].fillna(0)
        df.drop(columns=['fractional_cadence'], inplace=True)

    df['pace_min_km'] = np.where(
        df['enhanced_speed'] > 0.1,
        1000 / (df['enhanced_speed'] * 60), np.nan)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df.sort_index()


def _compute_speed(df):
    lat = np.radians(df['lat'].values)
    lon = np.radians(df['lon'].values)
    lat_prev = np.concatenate([[lat[0]], lat[:-1]])
    lon_prev = np.concatenate([[lon[0]], lon[:-1]])
    dlat, dlon = lat - lat_prev, lon - lon_prev
    a = np.sin(dlat/2)**2 + np.cos(lat_prev) * np.cos(lat) * np.sin(dlon/2)**2
    dist_m = 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dist_m[0] = 0
    idx = df.index.tz_convert('UTC').tz_localize(None) if df.index.tz else df.index
    times_s = idx.astype(np.int64) // 1_000_000_000
    dt = np.diff(times_s, prepend=times_s[0])
    dt[0] = 1
    dt = np.where(dt <= 0, 1, dt)
    raw = dist_m / dt
    smoothed = pd.Series(raw, index=df.index).rolling(5, center=True, min_periods=1).median()
    return np.clip(smoothed.values, 0, 7)


def parse_gpx(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing GPX: {e}", file=sys.stderr)
        return None

    rows = []
    for trkpt in root.findall('.//gpx:trkpt', NS):
        row = {}
        try:
            row['lat'] = float(trkpt.attrib['lat'])
            row['lon'] = float(trkpt.attrib['lon'])
        except (KeyError, ValueError):
            continue
        ele = trkpt.find('gpx:ele', NS)
        if ele is not None and ele.text:
            row['enhanced_altitude'] = float(ele.text)
        t = trkpt.find('gpx:time', NS)
        if t is not None and t.text:
            row['timestamp'] = pd.to_datetime(t.text, utc=True)
        else:
            continue
        ext = trkpt.find('.//gpxtpx:TrackPointExtension', NS)
        if ext is not None:
            hr  = ext.find('gpxtpx:hr',  NS)
            cad = ext.find('gpxtpx:cad', NS)
            if hr  is not None and hr.text:  row['heart_rate'] = int(hr.text)
            if cad is not None and cad.text: row['cadence']    = float(cad.text)
        rows.append(row)

    if len(rows) < 10:
        return None

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df['enhanced_speed'] = _compute_speed(df)
    df['pace_min_km'] = np.where(df['enhanced_speed'] > 0.1,
                                  1000 / (df['enhanced_speed'] * 60), np.nan)
    return df


# ── Feature extraction (mirrors Phase 2) ─────────────────────────────────────
def extract_features(df, summary):
    """
    Extract the same features as Phase 2 from a single activity.
    summary: dict with keys distance_km, moving_time_min, elevation_gain_m,
             hr_max, moving_ratio, atl_pre, ctl_pre, tsb_pre,
             weekly_km_7d, runs_7d, days_since_last_run, weekly_km_delta_pct
    All rolling features default to 0 if not provided (standalone use).
    """
    feats = {}

    # Pace features
    moving = df['pace_min_km'].dropna()
    feats['pace_mean'] = moving.mean() if len(moving) > 0 else np.nan
    feats['pace_std']  = moving.std()  if len(moving) > 1 else np.nan
    feats['pace_cv']   = (feats['pace_std'] / feats['pace_mean']
                          if feats['pace_mean'] and feats['pace_mean'] > 0 else np.nan)
    feats['pace_p10']  = moving.quantile(0.10) if len(moving) > 0 else np.nan
    feats['pace_p90']  = moving.quantile(0.90) if len(moving) > 0 else np.nan

    # HR features
    hr = df['heart_rate'].dropna() if 'heart_rate' in df.columns else pd.Series([], dtype=float)
    feats['hr_mean']         = hr.mean() if len(hr) > 0 else np.nan
    feats['hr_std']          = hr.std()  if len(hr) > 1 else np.nan
    feats['hr_max_recorded'] = hr.max()  if len(hr) > 0 else np.nan
    feats['hr_coverage']     = len(hr) / len(df)

    # HR zones
    thresholds = [0, 0.60, 0.70, 0.80, 0.90, 1.01]
    for i in range(1, 6):
        lo, hi = thresholds[i-1] * MAX_HR, thresholds[i] * MAX_HR
        feats[f'hr_zone{i}_frac'] = (
            ((hr >= lo) & (hr < hi)).sum() / len(hr) if len(hr) > 0 else np.nan)

    # HR-pace decoupling
    paired = df[['pace_min_km', 'heart_rate']].dropna() if 'heart_rate' in df.columns else pd.DataFrame()
    if len(paired) > 30:
        mid = len(paired) // 2
        r1 = paired.iloc[:mid]['heart_rate'].mean()  / paired.iloc[:mid]['pace_min_km'].mean()
        r2 = paired.iloc[mid:]['heart_rate'].mean() / paired.iloc[mid:]['pace_min_km'].mean()
        feats['hr_pace_decoupling'] = (r2 - r1) / r1 if r1 > 0 else np.nan
    else:
        feats['hr_pace_decoupling'] = np.nan

    # Cadence
    cad = df['cadence'].dropna() if 'cadence' in df.columns else pd.Series([], dtype=float)
    feats['cadence_mean'] = cad.mean() if len(cad) > 10 else np.nan
    feats['cadence_std']  = cad.std()  if len(cad) > 10 else np.nan

    # Elevation
    alt = df['enhanced_altitude'].dropna() if 'enhanced_altitude' in df.columns else pd.Series([], dtype=float)
    if len(alt) > 10:
        diffs = alt.diff().dropna()
        feats['elevation_gain_stream'] = diffs[diffs > 0].sum()
        feats['elevation_loss_stream'] = diffs[diffs < 0].abs().sum()
    else:
        feats['elevation_gain_stream'] = np.nan
        feats['elevation_loss_stream'] = np.nan

    feats['moving_frac'] = (df['enhanced_speed'] > 0.5).sum() / len(df)

    # Data quality flags
    feats['has_hr']      = int(len(hr) > 0)
    feats['has_cadence'] = int(len(cad) > 10)
    feats['has_gps']     = int('lat' in df.columns and df['lat'].notna().sum() > 10)

    # Summary + rolling features (from argument or defaults)
    for key, default in [
        ('distance_km', np.nan), ('moving_time_min', np.nan),
        ('elevation_gain_m', np.nan), ('hr_max', np.nan),
        ('moving_ratio', np.nan), ('atl_pre', 0.0),
        ('ctl_pre', 0.0), ('tsb_pre', 0.0), ('weekly_km_7d', 0.0),
        ('runs_7d', 0), ('days_since_last_run', np.nan),
        ('weekly_km_delta_pct', np.nan),
    ]:
        feats[key] = summary.get(key, default)

    return feats
def estimate_current_vdot(race_df, current_ctl, current_date=None):
    """
    Estimate current VDOT given training load context.
    
    Strategy:
    1. Weight each past race by recency and CTL similarity
    2. Adjust upward if current CTL > race CTL (fitter now)
    3. Return estimate with uncertainty bounds
    
    With only 5 races we use a weighted average, not a regression.
    This is honest — we don't overfit a line to 5 points.
    """
    import pandas as pd
    
    if current_date is None:
        current_date = pd.Timestamp.now()
    
    df = race_df.copy()
    
    # Recency weight — more recent races matter more
    # Half-life of 180 days: a race 180 days ago gets half the weight
    days_ago = (current_date - df['date']).dt.days.values
    recency_weight = np.exp(-days_ago / 180)
    
    # CTL similarity weight — races run at similar fitness are more predictive
    # Avoid division by zero if CTL values are identical
    ctl_range = df['ctl_pre'].max() - df['ctl_pre'].min()
    if ctl_range > 0:
        ctl_diff = np.abs(df['ctl_pre'].values - current_ctl)
        ctl_weight = np.exp(-ctl_diff / (ctl_range * 0.5))
    else:
        ctl_weight = np.ones(len(df))
    
    # Combined weight
    weights = recency_weight * ctl_weight
    weights = weights / weights.sum()
    
    # Weighted VDOT estimate from past races
    base_vdot = np.average(df['vdot'].values, weights=weights)
    
    # CTL adjustment — if you're fitter now than at your reference races,
    # adjust VDOT upward. We use the CTL→VDOT slope from the data,
    # but cap it to avoid overconfident extrapolation
    # Slope from our regression: ~4.1 VDOT per CTL unit
    # We use a conservative 3.0 to avoid overclaiming
    weighted_ctl = np.average(df['ctl_pre'].values, weights=weights)
    ctl_delta = current_ctl - weighted_ctl
    ctl_adjustment = np.clip(ctl_delta * 3.0, -5.0, 5.0)  # cap at ±5 VDOT
    
    adjusted_vdot = base_vdot + ctl_adjustment
    
    # Uncertainty: std of weighted residuals + model uncertainty
    # With 5 races we're honest about wide bounds
    residuals = df['vdot'].values - base_vdot
    weighted_std = np.sqrt(np.average(residuals**2, weights=weights))
    uncertainty = max(weighted_std, 1.5)  # minimum ±1.5 VDOT
    
    return {
        'vdot_estimate':  adjusted_vdot,
        'vdot_low':       adjusted_vdot - uncertainty,
        'vdot_high':      adjusted_vdot + uncertainty,
        'base_vdot':      base_vdot,
        'ctl_adjustment': ctl_adjustment,
        'uncertainty':    uncertainty,
        'most_similar_race': df.loc[weights.argmax(), 'name'],
    }


def predict_race_times(vdot_result, distances_km=None):
    """
    Given a VDOT estimate with uncertainty, predict finish times
    at common race distances with confidence ranges.
    """
    if distances_km is None:
        distances_km = [1.0, 3.0, 5.0, 10.0, 21.1]
    
    predictions = []
    for dist in distances_km:
        t_est  = predict_time_from_vdot(vdot_result['vdot_estimate'], dist * 1000)
        t_fast = predict_time_from_vdot(vdot_result['vdot_high'],     dist * 1000)
        t_slow = predict_time_from_vdot(vdot_result['vdot_low'],      dist * 1000)
        
        def fmt(t):
            if t >= 60:
                h = int(t // 60)
                m = int(t % 60)
                s = int((t - int(t)) * 60)
                return f"{h}:{m:02d}:{s:02d}"
            else:
                m = int(t)
                s = int((t - m) * 60)
                return f"{m}:{s:02d}"
        
        predictions.append({
            'distance_km': dist,
            'predicted':   fmt(t_est),
            'fast_bound':  fmt(t_fast),
            'slow_bound':  fmt(t_slow),
            'pace_min_km': t_est / dist,
        })
    
    return pd.DataFrame(predictions)
def compute_readiness(features_row, race_df=None):
    """
    Compute a 0-100 training readiness score from training load features.
    
    Components:
    1. Form (TSB) — are you fresh or fatigued?           weight 0.40
    2. Fitness (CTL trend) — is fitness building?        weight 0.25  
    3. Rest (days since last run) — adequate recovery?   weight 0.20
    4. Load stability (week delta) — sustainable ramp?   weight 0.15
    
    Each component is normalised to 0-100 before weighting.
    """
    
    tsb        = features_row.get('tsb_pre',             0.0)
    ctl        = features_row.get('ctl_pre',             0.0)
    atl        = features_row.get('atl_pre',             0.0)
    days_rest  = features_row.get('days_since_last_run', 1.0)
    wk_delta   = features_row.get('weekly_km_delta_pct', 0.0)
    weekly_km  = features_row.get('weekly_km_7d',        0.0)

    scores = {}
    explanations = []

    # ── Component 1: Form (TSB) ───────────────────────────────────────────────
    # TSB range in your data: roughly -4 to +3
    # Peak readiness: TSB around +0.5 to +1.5 (fresh but not detrained)
    # Poor readiness: TSB < -2 (deep fatigue) or TSB > 3 (detrained)
    # Map to 0-100 with peak at TSB=+1
    if pd.isna(tsb):
        scores['form'] = 50
    else:
        # Gaussian-shaped scoring — peaks at TSB=+1, falls off both sides
        # Width parameter controls how quickly score drops
        form_score = 100 * np.exp(-((tsb - 1.0) ** 2) / (2 * 1.5**2))
        scores['form'] = np.clip(form_score, 0, 100)
    
    if tsb < -2:
        explanations.append("high accumulated fatigue")
    elif tsb < -0.5:
        explanations.append("moderate fatigue — consider an easy day")
    elif tsb > 2.5:
        explanations.append("very fresh — fitness may be declining from rest")
    else:
        explanations.append("good form balance")

    # ── Component 2: Fitness base (CTL) ──────────────────────────────────────
    # Higher CTL = better base = more ready to handle training
    # Normalise against your personal max CTL from race data
    if race_df is not None:
        personal_max_ctl = race_df['ctl_pre'].max()
    else:
        personal_max_ctl = 3.0   # fallback estimate
    
    if pd.isna(ctl) or personal_max_ctl == 0:
        scores['fitness'] = 50
    else:
        ctl_pct = ctl / personal_max_ctl
        scores['fitness'] = np.clip(ctl_pct * 100, 0, 100)
    
    if ctl < personal_max_ctl * 0.3:
        explanations.append("low fitness base — build gradually")
    elif ctl > personal_max_ctl * 0.8:
        explanations.append("strong fitness base")

    # ── Component 3: Rest adequacy ────────────────────────────────────────────
    # 1-2 days rest: optimal for most training runs
    # 0 days (ran yesterday hard): might need more recovery
    # 5+ days: well rested but possibly stale
    if pd.isna(days_rest):
        scores['rest'] = 60
    else:
        # Peak at 1-2 days rest
        if days_rest <= 0:
            rest_score = 40
        elif days_rest <= 2:
            rest_score = 95
        elif days_rest <= 4:
            rest_score = 80
        elif days_rest <= 7:
            rest_score = 65
        else:
            rest_score = max(40, 65 - (days_rest - 7) * 3)
        scores['rest'] = rest_score
    
    if days_rest and days_rest >= 3:
        explanations.append(f"{int(days_rest)} days since last run — well recovered")
    elif days_rest and days_rest <= 1:
        explanations.append("ran recently — monitor fatigue")

    # ── Component 4: Load ramp (sustainability) ───────────────────────────────
    # Sudden volume spikes increase injury risk and reduce readiness
    # > 30% week-over-week increase is the "10% rule" danger zone
    # Negative delta (reducing load) is fine
    if pd.isna(wk_delta):
        scores['ramp'] = 75   # unknown — assume moderate
    else:
        if wk_delta > 0.5:      # >50% increase — high risk
            ramp_score = 30
            explanations.append("large volume spike this week — injury risk elevated")
        elif wk_delta > 0.3:    # 30-50% — caution zone
            ramp_score = 60
            explanations.append("volume increasing quickly — monitor closely")
        elif wk_delta > -0.3:   # -30% to +30% — sustainable
            ramp_score = 90
        else:                   # reducing load — recovering
            ramp_score = 80
        scores['ramp'] = ramp_score

    # ── Weighted composite ────────────────────────────────────────────────────
    weights = {'form': 0.40, 'fitness': 0.25, 'rest': 0.20, 'ramp': 0.15}
    composite = sum(scores[k] * weights[k] for k in weights)
    composite = np.clip(composite, 0, 100)

    # ── Category ──────────────────────────────────────────────────────────────
    if composite >= 80:
        category = "PEAK"
        category_desc = "Ready for hard training or racing"
    elif composite >= 65:
        category = "GOOD"
        category_desc = "Good to train — moderate to hard effort fine"
    elif composite >= 45:
        category = "MODERATE"
        category_desc = "Train easy — prioritise recovery"
    else:
        category = "LOW"
        category_desc = "Rest or very easy activity only"

    return {
        'score':         round(composite, 1),
        'category':      category,
        'description':   category_desc,
        'components':    scores,
        'explanations':  explanations,
        'tsb':           tsb,
        'ctl':           ctl,
        'atl':           atl,
    }
def suggest_workout(readiness_result, recent_labels, vdot_result=None):
    """
    Suggest the next workout type based on readiness and recent history.
    
    readiness_result: output of compute_readiness()
    recent_labels: list of recent workout labels, most recent last
                   e.g. ['EASY', 'TEMPO', 'EASY', 'LONG']
    vdot_result: optional, used to suggest specific paces
    
    Logic: rule-based decision tree grounded in periodisation principles.
    We use rules not a model — with 197 runs the sequence patterns
    are too sparse to learn reliably. Rules are interpretable and correct.
    """
    score    = readiness_result['score']
    category = readiness_result['category']
    tsb      = readiness_result['tsb']
    ctl      = readiness_result['ctl']

    # Analyse recent history
    n_recent = min(7, len(recent_labels))
    recent   = recent_labels[-n_recent:] if recent_labels else []

    recent_hard = sum(1 for l in recent if l in ('TEMPO', 'INTERVAL'))
    recent_long = sum(1 for l in recent if l == 'LONG')
    last        = recent[-1] if recent else None
    second_last = recent[-2] if len(recent) >= 2 else None

    suggestion  = None
    reason      = None
    paces       = None

    # ── Rule 1: Low readiness → always EASY ──────────────────────────────────
    if score < 45:
        suggestion = 'EASY'
        reason = (f"Readiness is low ({score:.0f}/100). "
                  f"Your body needs recovery before quality work.")

    # ── Rule 2: Never hard back-to-back ──────────────────────────────────────
    elif last in ('TEMPO', 'INTERVAL'):
        suggestion = 'EASY'
        reason = (f"Your last run was {last}. "
                  f"At least one easy day between hard efforts reduces injury risk.")

    # ── Rule 3: Long run spacing — once per week maximum ─────────────────────
    elif last == 'LONG' or second_last == 'LONG':
        if score >= 65:
            suggestion = 'EASY'
            reason = "Recovery run after your long effort. Keep it genuinely easy."
        else:
            suggestion = 'EASY'
            reason = "Still recovering from your long run. Easy only."

    # ── Rule 4: High readiness + no recent hard work → quality session ────────
    elif score >= 80 and recent_hard == 0:
        # Alternate between TEMPO and INTERVAL based on recent history
        # If haven't done intervals in a while, suggest them
        recent_intervals = sum(1 for l in recent if l == 'INTERVAL')
        if recent_intervals == 0 and ctl > 1.0:
            suggestion = 'INTERVAL'
            reason = (f"Peak readiness ({score:.0f}/100) and no recent intervals. "
                      f"Good time for speed work.")
        else:
            suggestion = 'TEMPO'
            reason = (f"Peak readiness ({score:.0f}/100). "
                      f"Sustained tempo effort will build threshold fitness.")

    # ── Rule 5: Good readiness + one recent hard session → easy or long ───────
    elif score >= 65 and recent_hard == 1:
        if recent_long == 0 and ctl > 0.8:
            suggestion = 'LONG'
            reason = (f"Good readiness ({score:.0f}/100), no long run this week, "
                      f"and solid fitness base. Long run will build aerobic capacity.")
        else:
            suggestion = 'EASY'
            reason = (f"Good readiness ({score:.0f}/100) but recent hard effort. "
                      f"Easy run maintains volume without adding stress.")

    # ── Rule 6: Good readiness + already done hard + long → easy ─────────────
    elif score >= 65 and recent_hard >= 2:
        suggestion = 'EASY'
        reason = (f"You've done {recent_hard} hard sessions recently. "
                  f"Back off and let the adaptation happen.")

    # ── Default: moderate readiness → easy ────────────────────────────────────
    else:
        suggestion = 'EASY'
        reason = (f"Moderate readiness ({score:.0f}/100). "
                  f"Easy running builds base without digging a deeper hole.")

    # ── Pace guidance from VDOT ───────────────────────────────────────────────
    if vdot_result is not None:
        vdot = vdot_result['vdot_estimate']
        # Jack Daniels training paces derived from VDOT
        # Easy: 59-74% vVO2max — roughly 70-75s/km slower than 5km race pace
        t_5k    = predict_time_from_vdot(vdot, 5000)
        pace_5k = t_5k / 5.0

        pace_guidance = {
            'EASY':     (pace_5k + 1.5, pace_5k + 2.5),   # 90-150s/km slower than 5km
            'LONG':     (pace_5k + 1.2, pace_5k + 2.0),   # similar to easy
            'TEMPO':    (pace_5k + 0.3, pace_5k + 0.8),   # comfortably hard
            'INTERVAL': (pace_5k - 0.2, pace_5k + 0.2),   # 5km race pace ± 12s
        }

        lo, hi = pace_guidance[suggestion]

        def fmt_pace(p):
            m = int(p)
            s = int((p - m) * 60)
            return f"{m}:{s:02d}"

        paces = f"{fmt_pace(lo)} – {fmt_pace(hi)} min/km"

    return {
        'suggestion': suggestion,
        'reason':     reason,
        'paces':      paces,
        'readiness':  score,
        'category':   category,
    }
def vdot_from_race(distance_m, time_min):
    """
    Estimate VDOT from a race performance.
    distance_m: race distance in metres
    time_min: finish time in minutes
    Returns VDOT (ml/kg/min equivalent)
    """
    # Oxygen cost of running at race pace
    # velocity in metres per minute
    v = distance_m / time_min
    
    # VO2 at race velocity (Daniels formula)
    vo2 = -4.60 + 0.182258 * v + 0.000104 * v**2
    
    # Percent VO2max utilised at race duration
    # (longer races use a lower % of max)
    pct_max = (0.8 + 0.1894393 * np.exp(-0.012778 * time_min)
                   + 0.2989558 * np.exp(-0.1932605 * time_min))
    
    return vo2 / pct_max


def predict_time_from_vdot(vdot, distance_m):
    """
    Given VDOT, predict finish time at a target distance.
    Uses Newton's method to invert the VDOT formula numerically.
    Returns predicted time in minutes.
    """
    # Initial guess: assume 5:00/km pace
    t = distance_m / 1000 * 5.0
    
    for _ in range(50):   # Newton iterations
        v      = distance_m / t
        vo2    = -4.60 + 0.182258 * v + 0.000104 * v**2
        pct    = (0.8 + 0.1894393 * np.exp(-0.012778 * t)
                      + 0.2989558 * np.exp(-0.1932605 * t))
        f      = vo2 / pct - vdot

        # Numerical derivative
        dt     = 0.001
        v2     = distance_m / (t + dt)
        vo2_2  = -4.60 + 0.182258 * v2 + 0.000104 * v2**2
        pct_2  = (0.8 + 0.1894393 * np.exp(-0.012778 * (t+dt))
                      + 0.2989558 * np.exp(-0.1932605 * (t+dt)))
        f2     = vo2_2 / pct_2 - vdot
        df     = (f2 - f) / dt

        t = t - f / df
        if abs(f) < 1e-8:
            break
    return t


    
def compute_rolling_features(activity_date, runs_csv_path):
    """
    Given an activity date and a path to runs.csv, compute
    ATL/CTL/TSB and other rolling load features as of that date.
    
    This is the same logic as Phase 2 notebooks/02_features.ipynb
    but self-contained — no pandas ewm magic, just the math.
    """
    runs = pd.read_csv(runs_csv_path, parse_dates=['date'])
    runs = runs.sort_values('date').reset_index(drop=True)

    # Only use runs BEFORE the activity date — no leakage
    prior = runs[runs['date'].dt.normalize() < pd.Timestamp(activity_date).normalize()].copy()

    if len(prior) == 0:
        # First ever run — no history
        return {
            'atl_pre': 0.0, 'ctl_pre': 0.0, 'tsb_pre': 0.0,
            'weekly_km_7d': 0.0, 'runs_7d': 0,
            'days_since_last_run': np.nan,
            'weekly_km_delta_pct': np.nan,
        }

    # Build daily distance series up to day before activity
    date_min = prior['date'].min().normalize()
    date_max = pd.Timestamp(activity_date).normalize() - pd.Timedelta(days=1)
    all_days = pd.date_range(date_min, date_max, freq='D')

    daily = (prior.groupby(prior['date'].dt.normalize())['distance_km']
                  .sum()
                  .reindex(all_days, fill_value=0.0))

    # Exponential weighted load — same spans as Phase 2
    atl = daily.ewm(span=7,  adjust=False).mean().iloc[-1]
    ctl = daily.ewm(span=42, adjust=False).mean().iloc[-1]
    tsb = ctl - atl

    # Weekly volume and frequency
    day       = pd.Timestamp(activity_date).normalize()
    week_ago  = day - pd.Timedelta(days=7)
    last_week = daily.loc[week_ago:day - pd.Timedelta(days=1)]
    weekly_km = last_week.sum()
    runs_7d   = int((last_week > 0).sum())

    # Days since last run
    prior_run_days = daily[daily > 0]
    days_since = (day - prior_run_days.index[-1]).days if len(prior_run_days) > 0 else np.nan

    # Week over week volume change
    two_weeks_ago = day - pd.Timedelta(days=14)
    prev_week = daily.loc[two_weeks_ago:week_ago - pd.Timedelta(days=1)]
    prev_km   = prev_week.sum()
    wk_delta  = (weekly_km - prev_km) / prev_km if prev_km > 0 else np.nan

    return {
        'atl_pre':             float(atl),
        'ctl_pre':             float(ctl),
        'tsb_pre':             float(tsb),
        'weekly_km_7d':        float(weekly_km),
        'runs_7d':             runs_7d,
        'days_since_last_run': float(days_since) if not np.isnan(days_since) else np.nan,
        'weekly_km_delta_pct': float(wk_delta)   if not np.isnan(wk_delta)   else np.nan,
    }
# ── Inference ─────────────────────────────────────────────────────────────────
def classify_activity(filepath, summary=None, verbose=False,
                      history_csv=None, predict_race=False, suggest=False):

    filepath = Path(filepath)
    summary  = summary or {}

    # Parse file
    name = filepath.name.lower()
    if name.endswith('.fit.gz') or name.endswith('.fit'):
        df = parse_fit(filepath)
    elif name.endswith('.gpx'):
        df = parse_gpx(filepath)
    else:
        print(f"Unsupported file type: {filepath.suffix}", file=sys.stderr)
        return None

    if df is None:
        print("Could not parse file or too few records.", file=sys.stderr)
        return None

    # Derive basic summary fields from stream
    if 'distance_km' not in summary:
        if 'distance' in df.columns:
            summary['distance_km'] = df['distance'].max() / 1000
        else:
            summary['distance_km'] = (df['enhanced_speed'].fillna(0) * 1.0).sum() / 1000

    if 'moving_time_min' not in summary:
        summary['moving_time_min'] = (df['enhanced_speed'] > 0.5).sum() / 60

    if 'moving_ratio' not in summary:
        summary['moving_ratio'] = (df['enhanced_speed'] > 0.5).sum() / len(df)

    # Rolling features from history
    race_df_local  = None
    rolling        = {}
    readiness      = None
    vdot_result    = None
    suggestion     = None
    recent_labels  = []

    if history_csv is not None:
        try:
            # Parse activity date from stream
            _peek = df
            activity_date = _peek.index[0]

            # Rolling load features
            rolling = compute_rolling_features(activity_date, history_csv)
            summary.update(rolling)

            # Load runs for race predictor + readiness
            runs_hist = pd.read_csv(history_csv, parse_dates=['date'])

            # Rebuild race_df from history
            race_keywords = ['race', 'Race', '5k', '5K', '10k', '10K',
                             'half', 'Half', 'marathon', 'Marathon',
                             'parkrun', 'Parkrun', 'Park Run']
            race_mask = runs_hist['name'].str.contains(
                '|'.join(race_keywords), na=False, case=False)
            race_runs = runs_hist[race_mask & (runs_hist['distance_km'] >= 3)].copy()

            if len(race_runs) >= 2:
                race_rows = []
                for _, r in race_runs.iterrows():
                    if pd.notna(r['avg_pace_min_km']) and r['avg_pace_min_km'] > 0:
                        t_min = r['distance_km'] * r['avg_pace_min_km']
                        v = vdot_from_race(r['distance_km'] * 1000, t_min)
                        race_rows.append({
                            'date':        r['date'],
                            'name':        r['name'],
                            'distance_km': r['distance_km'],
                            'time_min':    t_min,
                            'vdot':        v,
                            'ctl_pre':     r.get('ctl_pre', 0.0),
                        })
                race_df_local = pd.DataFrame(race_rows)

            # Recent workout labels for suggester
            # Approximate labels from pace_cv and distance
            def quick_label(r):
                cv   = r.get('pace_cv',    0.2)
                dist = r.get('distance_km', 5.0)
                if pd.isna(cv): cv = 0.2
                if dist >= 10:                      return 'LONG'
                if cv > 0.49:                       return 'INTERVAL'
                if r.get('avg_pace_min_km', 6) < 5.2 and dist >= 3: return 'TEMPO'
                return 'EASY'

            recent_labels = (runs_hist
                             .sort_values('date')
                             .tail(7)
                             .apply(quick_label, axis=1)
                             .tolist())

        except Exception as e:
            if verbose:
                print(f"[history] Error: {e}", file=sys.stderr)

    # Extract features and classify
    feats = extract_features(df, summary)

    model_dir   = Path(__file__).parent.parent / 'models'
    sess        = rt.InferenceSession(str(model_dir / 'classifier.onnx'))
    with open(model_dir / 'feature_cols.json') as f:
        feature_cols = json.load(f)
    with open(model_dir / 'label_map.json') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    X           = np.array([[feats.get(c, np.nan) for c in feature_cols]],
                            dtype=np.float32)
    input_name  = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    outputs     = sess.run(output_names, {input_name: X})
    pred_enc    = outputs[0][0]
    pred_label  = label_map[int(pred_enc)]

    proba = None
    if len(outputs) > 1:
        prob_array = outputs[1]
        if hasattr(prob_array[0], 'values'):
            proba = {label_map[i]: v for i, v in enumerate(prob_array[0].values())}
        else:
            proba = {label_map[i]: float(v) for i, v in enumerate(prob_array[0])}

    # Race prediction
    if predict_race and race_df_local is not None and len(race_df_local) >= 2:
        current_ctl  = rolling.get('ctl_pre', 0.0)
        vdot_result  = estimate_current_vdot(
            race_df_local, current_ctl,
            pd.Timestamp(df.index[0]).tz_localize(None)
            if df.index[0].tzinfo is None else
            pd.Timestamp(df.index[0]).tz_convert(None))

    # Readiness
    if history_csv is not None:
        readiness = compute_readiness(
            {**rolling, 'ctl_pre': rolling.get('ctl_pre', 0),
             'atl_pre': rolling.get('atl_pre', 0),
             'tsb_pre': rolling.get('tsb_pre', 0)},
            race_df_local)

    # Workout suggestion
    if suggest and readiness is not None:
        suggestion = suggest_workout(readiness, recent_labels, vdot_result)

    # ── Verbose output ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'═'*48}")
        print(f"  STRAVA ML — ACTIVITY ANALYSIS")
        print(f"{'═'*48}")
        print(f"  File:      {filepath.name}")
        print(f"  Duration:  {summary.get('moving_time_min', len(df)/60):.0f} min  "
              f"│  Distance: {summary.get('distance_km', 0):.1f} km")
        print(f"  Pace:      {feats.get('pace_mean', 0):.2f} min/km  "
              f"(cv={feats.get('pace_cv', 0):.2f})")
        if feats['has_hr']:
            z45 = feats.get('hr_zone4_frac', 0) + feats.get('hr_zone5_frac', 0)
            print(f"  HR mean:   {feats.get('hr_mean', 0):.0f} bpm  │  Z4+Z5: {z45:.0%}")

        if rolling:
            print(f"\n{'─'*48}")
            print(f"  TRAINING LOAD")
            print(f"  CTL (fitness):  {rolling.get('ctl_pre', 0):.2f}  │  "
                  f"ATL (fatigue): {rolling.get('atl_pre', 0):.2f}  │  "
                  f"TSB (form): {rolling.get('tsb_pre', 0):+.2f}")
            print(f"  Weekly km: {rolling.get('weekly_km_7d', 0):.1f}  │  "
                  f"Days rest: {rolling.get('days_since_last_run', 0):.0f}")

        if readiness:
            print(f"\n{'─'*48}")
            print(f"  READINESS: {readiness['score']:.0f}/100 [{readiness['category']}]")
            print(f"  {readiness['description']}")
            for e in readiness['explanations']:
                print(f"  · {e}")

        print(f"\n{'─'*48}")
        print(f"  WORKOUT TYPE: {pred_label}")
        if proba:
            for lbl, prob in sorted(proba.items(), key=lambda x: -x[1]):
                bar = '█' * int(prob * 24)
                print(f"  {lbl:<10} {prob:.2f}  {bar}")

        if vdot_result:
            print(f"\n{'─'*48}")
            print(f"  RACE PREDICTIONS  (VDOT {vdot_result['vdot_estimate']:.1f})")
            preds = predict_race_times(vdot_result)
            for _, row in preds.iterrows():
                print(f"  {row['distance_km']:5.1f} km  →  {row['predicted']:>8}  "
                      f"({row['fast_bound']} – {row['slow_bound']})")

        if suggestion:
            print(f"\n{'─'*48}")
            print(f"  NEXT WORKOUT: {suggestion['suggestion']}")
            if suggestion['paces']:
                print(f"  Target pace:  {suggestion['paces']}")
            print(f"  {suggestion['reason']}")

        print(f"{'═'*48}\n")

    return pred_label


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Strava ML — Workout classifier + Training Intelligence')
    parser.add_argument('--file',         required=True,
                        help='Path to .fit, .fit.gz, or .gpx file')
    parser.add_argument('--history',      default=None,
                        help='Path to runs.csv for training load context')
    parser.add_argument('--predict-race', action='store_true',
                        help='Show race time predictions')
    parser.add_argument('--suggest',      action='store_true',
                        help='Suggest next workout type')
    parser.add_argument('--verbose',      action='store_true',
                        help='Print full analysis')
    args = parser.parse_args()

    result = classify_activity(
        args.file,
        verbose=args.verbose,
        history_csv=args.history,
        predict_race=args.predict_race,
        suggest=args.suggest,
    )
    if result:
        print(result)