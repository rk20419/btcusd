import csv
from datetime import datetime

# 📁 Your timeframe paths
old_path = {
    '1m': 'data/historical/BTCUSDT_1m_100.csv',
    
}

live_path = {
    '1m': 'data/live/1m.csv',
}

def merge_and_overwrite_tf(tf, old_file, live_file):
    print(f"\n🔄 Processing TF: {tf}")

    # Load old
    with open(old_file, 'r') as f:
        old_rows = list(csv.reader(f))
    header = old_rows[0]
    old_data = old_rows[1:]
    old_timestamps = set(row[0] for row in old_data)
    print(f"[{tf}] Loaded {len(old_data)} old rows")

    # Load live
    with open(live_file, 'r') as f:
        live_rows = list(csv.reader(f))
    live_data = live_rows[1:]
    print(f"[{tf}] Loaded {len(live_data)} live rows")

    # Filter live data
    filtered_live = []
    removed_count = 0
    for row in live_data:
        ts = row[0]
        if ts in old_timestamps:
            print(f"[{tf}] 🛑 Skipped duplicate timestamp from live: {ts}")
            removed_count += 1
        else:
            filtered_live.append(row)

    print(f"[{tf}] Final live rows after filter: {len(filtered_live)}")
    print(f"[{tf}] Old rows kept: {len(old_data)}")
    print(f"[{tf}] Total merged: {len(filtered_live) + len(old_data)}")

    # Overwrite live.csv with merged result
    try:
        with open(live_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(old_data + filtered_live)
        print(f"[{tf}] ✅ live.csv overwritten successfully.")
    except Exception as e:
        print(f"[{tf}] ❌ Error writing live file — {e}")

# 🔁 Run for all TFs
for tf in old_path:
    merge_and_overwrite_tf(tf, old_path[tf], live_path[tf])