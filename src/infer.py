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


# ── Inference ─────────────────────────────────────────────────────────────────
def classify_activity(filepath, summary=None, verbose=False):
    """
    Main entry point. Parse a .fit/.fit.gz/.gpx file and return predicted label.
    """
    filepath = Path(filepath)
    summary  = summary or {}

    # Parse
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

    # Derive summary fields from stream if not provided
    if 'distance_km' not in summary:
        if 'distance' in df.columns:
            summary['distance_km'] = df['distance'].max() / 1000
        else:
            # estimate from GPS speed
            speed = df['enhanced_speed'].fillna(0)
            dt_s  = 1.0  # 1 Hz recording
            summary['distance_km'] = (speed * dt_s).sum() / 1000

    if 'moving_time_min' not in summary:
        summary['moving_time_min'] = (df['enhanced_speed'] > 0.5).sum() / 60

    if 'moving_ratio' not in summary:
        summary['moving_ratio'] = (df['enhanced_speed'] > 0.5).sum() / len(df)

    # Extract features
    feats = extract_features(df, summary)

    # Load model artifacts
    model_dir = Path(__file__).parent.parent / 'models'
    sess      = rt.InferenceSession(str(model_dir / 'classifier.onnx'))
    with open(model_dir / 'feature_cols.json') as f:
        feature_cols = json.load(f)
    with open(model_dir / 'label_map.json') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    # Build feature vector in correct order
    X = np.array([[feats.get(c, np.nan) for c in feature_cols]], dtype=np.float32)

    # Infer
    input_name   = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    outputs      = sess.run(output_names, {input_name: X})

    pred_enc   = outputs[0][0]
    pred_label = label_map[int(pred_enc)]

    # Get probabilities if available
    proba = None
    if len(outputs) > 1:
        prob_array = outputs[1]  # shape: (1, n_classes)
        if hasattr(prob_array[0], 'values'):
            # zipmap=True returns list of dicts
            proba = {label_map[i]: v for i, v in enumerate(prob_array[0].values())}
        else:
            # zipmap=False returns plain numpy array
            proba = {label_map[i]: float(v) for i, v in enumerate(prob_array[0])}

    if verbose:
        print(f"\n{'─'*45}")
        print(f"File:      {filepath.name}")
        print(f"Duration:  {summary.get('moving_time_min', len(df)/60):.0f} min")
        print(f"Distance:  {summary.get('distance_km', 0):.1f} km")
        print(f"Pace:      {feats.get('pace_mean', 0):.2f} min/km  "
              f"(cv={feats.get('pace_cv', 0):.2f})")
        if feats['has_hr']:
            print(f"HR mean:   {feats.get('hr_mean', 0):.0f} bpm  "
                  f"Z4+Z5: {(feats.get('hr_zone4_frac',0)+feats.get('hr_zone5_frac',0)):.0%}")
        print(f"{'─'*45}")
        print(f"Predicted: {pred_label}")
        if proba:
            print("Confidence:")
            for lbl, prob in sorted(proba.items(), key=lambda x: -x[1]):
                bar = '█' * int(prob * 20)
                print(f"  {lbl:<10} {prob:.2f}  {bar}")
        print(f"{'─'*45}\n")

    return pred_label


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify a running activity')
    parser.add_argument('--file',    required=True, help='Path to .fit, .fit.gz, or .gpx file')
    parser.add_argument('--verbose', action='store_true', help='Print detailed breakdown')
    args = parser.parse_args()

    result = classify_activity(args.file, verbose=args.verbose)
    if result:
        print(result)