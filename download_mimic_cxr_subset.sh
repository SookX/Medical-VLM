#!/usr/bin/env bash
set -Eeuo pipefail

username="${PHYSIONET_USER:-sooka}"
workspace="${1:-/mnt/w/Papers/Medical-VLM}"
qa_root="$workspace/physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench/qa"
download_root="$workspace/physionet.org/files"
target_root="$download_root/mimic-cxr-jpg/2.1.0/files"
base_url="https://physionet.org/files/mimic-cxr-jpg/2.1.0/"

if [[ ! -d "$qa_root" ]]; then
  echo "Error: CXReasonBench QA directory not found: $qa_root" >&2
  exit 1
fi

mkdir -p "$download_root"
files_list="$(mktemp)"
relative_list="$(mktemp)"
trap 'rm -f "$files_list" "$relative_list"' EXIT

# Extract only original MIMIC-CXR-JPG paths. Benchmark overlay PNG paths do
# not match this pattern and are therefore excluded.
find "$qa_root" \
  -path '*/path1/init/basic/*.json' \
  -type f \
  -print0 \
  | xargs -0 grep \
      --no-filename \
      --only-matching \
      --extended-regexp \
      'p[0-9]+/p[0-9]+/s[0-9]+/[0-9a-f-]+\.jpg' \
  | sort -u \
  > "$relative_list"

file_count="$(wc -l < "$relative_list")"
if [[ "$file_count" -ne 504 ]]; then
  echo "Error: expected 504 unique MIMIC JPEG paths, found $file_count." >&2
  exit 1
fi

sed 's|^|files/|' "$relative_list" > "$files_list"

present_before=0
while IFS= read -r relative_path; do
  if [[ -s "$target_root/$relative_path" ]]; then
    present_before=$((present_before + 1))
  fi
done < "$relative_list"

echo "MIMIC-CXR-JPG subset: $file_count unique JPEGs"
echo "Already present: $present_before/$file_count"
echo "Remaining file count: $((file_count - present_before))"
echo "Destination: $target_root"
echo "PhysioNet user: $username"
echo "Checking access with the first image before starting the batch."
echo "Your password is not saved; wget may request it again for the batch."

first_relative_path="$(head -n 1 "$relative_list")"
if ! wget \
  --spider \
  --user "$username" \
  --ask-password \
  "$base_url/files/$first_relative_path"; then
  echo "Error: PhysioNet denied access to MIMIC-CXR-JPG." >&2
  echo "Confirm that this account has signed the separate MIMIC-CXR-JPG data-use agreement, then rerun this script." >&2
  exit 1
fi

echo "Access check passed. Starting the 504-file batch."

cd "$download_root"
wget \
  --recursive \
  --timestamping \
  --continue \
  --no-parent \
  --no-host-directories \
  --cut-dirs=1 \
  --user "$username" \
  --ask-password \
  --input-file "$files_list" \
  --base "$base_url" \
  --progress=bar:force:noscroll \
  --show-progress

present_after=0
missing=0
while IFS= read -r relative_path; do
  if [[ -s "$target_root/$relative_path" ]]; then
    present_after=$((present_after + 1))
  else
    missing=$((missing + 1))
    echo "Missing or empty: $relative_path" >&2
  fi
done < "$relative_list"

echo "Downloaded and validated: $present_after/$file_count"
if [[ "$missing" -ne 0 ]]; then
  echo "Error: $missing MIMIC JPEGs are missing or empty." >&2
  exit 1
fi

echo "The CXReasonBench MIMIC-CXR-JPG subset is complete."
