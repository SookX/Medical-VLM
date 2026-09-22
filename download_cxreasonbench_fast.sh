#!/usr/bin/env bash
set -Eeuo pipefail

base_url="https://physionet.org/files/chexstruct-cxreasonbench/1.0.1"
username="${PHYSIONET_USER:-sooka}"
default_destination="/mnt/w/Papers/Medical-VLM/physionet.org/files/chexstruct-cxreasonbench/1.0.1"
destination="${1:-$default_destination}"
expected_bytes=3252323711

mkdir -p "$destination"
url_file="$(mktemp)"
checksum_file="$(mktemp)"
trap 'rm -f "$url_file" "$checksum_file"' EXIT

files=(
  "CXReasonBench/dx_by_dicoms.json"
  "CXReasonBench/pnt_on_cxr.zip"
  "CXReasonBench/qa.zip"
  "CXReasonBench/segmask_bodypart.zip"
  "CheXStruct/CheXStruct_column_descriptions.json"
  "CheXStruct/diagnostic_tasks/aortic_knob_enlargement.csv"
  "CheXStruct/diagnostic_tasks/ascending_aorta_enlargement.csv"
  "CheXStruct/diagnostic_tasks/cardiomegaly.csv"
  "CheXStruct/diagnostic_tasks/carina_angle.csv"
  "CheXStruct/diagnostic_tasks/descending_aorta_enlargement.csv"
  "CheXStruct/diagnostic_tasks/descending_aorta_tortuous.csv"
  "CheXStruct/diagnostic_tasks/inclusion.csv"
  "CheXStruct/diagnostic_tasks/inspiration.csv"
  "CheXStruct/diagnostic_tasks/mediastinal_widening.csv"
  "CheXStruct/diagnostic_tasks/projection.csv"
  "CheXStruct/diagnostic_tasks/rotation.csv"
  "CheXStruct/diagnostic_tasks/trachea_deviation.csv"
  "CheXStruct/global/abdominal_xray.csv"
  "CheXStruct/global/mask_number.csv"
  "CheXStruct/global/window.csv"
  "LICENSE.txt"
  "README.md"
  "SHA256SUMS.txt"
)

for file in "${files[@]}"; do
  printf '%s/%s\n' "$base_url" "$file" >> "$url_file"
done

downloaded_bytes=0
for file in "${files[@]}"; do
  [[ "$file" == "SHA256SUMS.txt" ]] && continue
  path="$destination/$file"
  if [[ -f "$path" ]]; then
    size="$(stat --format='%s' "$path")"
    downloaded_bytes=$((downloaded_bytes + size))
  fi
done

if (( downloaded_bytes > expected_bytes )); then
  downloaded_bytes=$expected_bytes
fi
remaining_bytes=$((expected_bytes - downloaded_bytes))

human_bytes() {
  awk -v bytes="$1" 'BEGIN {
    split("B KiB MiB GiB TiB", unit, " ")
    i = 1
    while (bytes >= 1024 && i < 5) { bytes /= 1024; i++ }
    printf "%.2f %s", bytes, unit[i]
  }'
}

human_duration() {
  local seconds="$1"
  printf '%dh %02dm %02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

percent="$(awk -v have="$downloaded_bytes" -v total="$expected_bytes" 'BEGIN { printf "%.2f", 100 * have / total }')"
echo "Package size: $(human_bytes "$expected_bytes")"
echo "Already present: $(human_bytes "$downloaded_bytes") ($percent%)"
echo "Estimated remaining transfer: $(human_bytes "$remaining_bytes")"
if (( remaining_bytes > 0 )); then
  echo "Approximate transfer times (excluding server latency):"
  for mbps in 10 25 50 100; do
    seconds=$(( (remaining_bytes * 8 + mbps * 1000000 - 1) / (mbps * 1000000) ))
    printf '  %3d Mbps: %s\n' "$mbps" "$(human_duration "$seconds")"
  done
fi

echo "Downloading the compact CXReasonBench distribution to: $destination"
echo "PhysioNet user: $username"
echo "Your password will be requested once and will not be saved."

wget \
  --continue \
  --timestamping \
  --progress=bar:force:noscroll \
  --show-progress \
  --user "$username" \
  --ask-password \
  --input-file "$url_file" \
  --directory-prefix "$destination" \
  --force-directories \
  --no-host-directories \
  --cut-dirs=3

manifest="$destination/SHA256SUMS.txt"
if [[ ! -f "$manifest" ]]; then
  echo "Error: checksum manifest was not downloaded: $manifest" >&2
  exit 1
fi

for file in "${files[@]}"; do
  [[ "$file" == "SHA256SUMS.txt" ]] && continue
  awk -v wanted="$file" '$0 ~ "[ *]" wanted "$" { print }' "$manifest" >> "$checksum_file"
done

expected=$((${#files[@]} - 1))
found="$(wc -l < "$checksum_file")"
if [[ "$found" -ne "$expected" ]]; then
  echo "Error: expected $expected checksum entries, found $found." >&2
  exit 1
fi

echo "Verifying $expected packaged files against PhysioNet SHA-256 checksums..."
(
  cd "$destination"
  sha256sum --check --strict "$checksum_file"
)

echo "CXReasonBench compact download is complete and verified."
