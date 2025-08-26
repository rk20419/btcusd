import csv

# Timeframe-wise paths
live_files = {
     '1m': 'data/live/1m.csv',
}

historical_files = {
    tf: f"data/historical/BTCUSDT_1m_100.csv" for tf in live_files
}

# Load each row as a raw string
def load_rows_as_strings(path):
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = [','.join(r).strip() for r in reader]
            return headers, rows
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return [], []

# Compare full string rows
def audit_string_match(tf, hist_lines, live_lines):
    hist_set = set(hist_lines)
    live_set = set(live_lines)
    matched = hist_set & live_set
    total = len(hist_lines)
    match_count = len(matched)

    percent = (match_count / total) * 100 if total > 0 else 0

    print(f"\n🧾 Timeframe: {tf}")
    print(f"📦 Historical rows: {total}")
    print(f"📦 Live rows: {len(live_lines)}")
    print(f"✅ Exact matches: {match_count}")
    print(f"🎯 Match percent: {percent:.2f}%")

# Master audit runner
def run_string_match_audit():
    for tf in live_files:
        hist_path = historical_files[tf]
        live_path = live_files[tf]

        hist_headers, hist_lines = load_rows_as_strings(hist_path)
        live_headers, live_lines = load_rows_as_strings(live_path)

        if hist_headers != live_headers:
            print(f"\n⚠️ Header mismatch in {tf}, skipping.")
            continue

        audit_string_match(tf, hist_lines, live_lines)

run_string_match_audit()