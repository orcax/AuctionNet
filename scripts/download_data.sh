#!/usr/bin/env bash
#
# Download the AuctionNet pre-generated dataset (~80GB across 11 zips) and place
# the period-*.csv files where the training pipeline expects them:
#   strategy_train_env/data/traffic/period-7.csv ... period-27.csv
#
# Source links: pre_generated_dataset/readme_dataset.md (Alibaba OSS, public, no auth).
# Downloads are resumable: re-run the script to continue interrupted transfers.
#
# Usage:
#   bash scripts/download_data.sh            # download all 11 zips + extract
#   KEEP_ZIPS=1 bash scripts/download_data.sh   # keep the .zip files after extracting
#   bash scripts/download_data.sh 13 14-15      # only the given period chunks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final"
ZIP_DIR="$REPO_ROOT/strategy_train_env/data/zips"
TRAFFIC_DIR="$REPO_ROOT/strategy_train_env/data/traffic"

ALL_CHUNKS=(7-8 9-10 11-12 13 14-15 16-17 18-19 20-21 22-23 24-25 26-27)
CHUNKS=("${@:-}")
[ -z "${CHUNKS[*]}" ] && CHUNKS=("${ALL_CHUNKS[@]}")

mkdir -p "$ZIP_DIR" "$TRAFFIC_DIR"

for chunk in "${CHUNKS[@]}"; do
    fname="autoBidding_general_track_final_data_period_${chunk}.zip"
    url="$BASE_URL/$fname"
    zip_path="$ZIP_DIR/$fname"

    echo ">>> Downloading $fname"
    # Retry across connection drops: -C - resumes the partial file each attempt,
    # --retry-all-errors covers mid-transfer resets (curl exit 56) that --retry alone skips.
    attempt=0
    until curl -fL -C - --retry 10 --retry-all-errors --retry-delay 5 -o "$zip_path" "$url"; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge 10 ]; then
            echo "!!! Giving up on $fname after $attempt attempts" >&2
            exit 1
        fi
        echo ">>> Connection dropped on $fname; retry #$attempt in 10s (resuming)..." >&2
        sleep 10
    done

    echo ">>> Extracting $fname -> $TRAFFIC_DIR"
    # Use Python's zipfile (no `unzip` binary required). Extract period-*.csv flat,
    # skipping the __MACOSX junk dir.
    python3 - "$zip_path" "$TRAFFIC_DIR" <<'PY'
import os, sys, zipfile
zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    for member in z.namelist():
        base = os.path.basename(member)
        if member.startswith("__MACOSX/") or not base.startswith("period-") or not base.endswith(".csv"):
            continue
        out = os.path.join(dest, base)
        with z.open(member) as src, open(out, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        print("  extracted", base)
PY

    if [ "${KEEP_ZIPS:-0}" != "1" ]; then
        rm -f "$zip_path"
        echo ">>> Removed $fname (set KEEP_ZIPS=1 to keep)"
    fi
done

# Drop the zips dir if empty.
rmdir "$ZIP_DIR" 2>/dev/null || true

echo
echo "Done. CSV files in $TRAFFIC_DIR:"
ls -1 "$TRAFFIC_DIR"/period-*.csv 2>/dev/null | sed 's#.*/#  #' || echo "  (none found)"
