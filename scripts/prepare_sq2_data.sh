#!/usr/bin/env bash
# Run from a Linux checkout with curl, unzip, sha256sum and uv installed.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data outputs/predictions outputs/checkpoints
stage=$(mktemp -d data/.sq2-downloads.XXXXXX)
trap 'rm -rf -- "$stage"' EXIT

matches() {
    [[ -f "$1" ]] && printf '%s  %s\n' "$2" "$1" | sha256sum --check --status
}

fetch() {
    local path=$1 expected=$2 url=$3
    local tmp="$stage/${path##*/}"
    if matches "$path" "$expected"; then
        printf 'Verified %s\n' "$path"
        return
    fi
    if [[ "$url" == https://drive.google.com/* ]]; then
        uvx gdown "$url" -O "$tmp"
    else
        curl --fail --location --retry 3 --output "$tmp" "$url"
    fi
    if ! matches "$tmp" "$expected"; then
        printf 'SHA-256 mismatch for %s. Check the source file before retrying.\n' "$path" >&2
        return 1
    fi
    mv -- "$tmp" "$path"
}

image_url=${SQ2_IMAGE_URL:-'https://www.dropbox.com/s/njehqednn8mycze/img_data.zip?dl=1'}
fetch data/img_data.zip 407ffea5497ae07dcf7a9850fc68bf6be5386f62787d8825458ac74885cff967 "$image_url"
unzip -oq data/img_data.zip 'monthly_20d/*' -d data

fetch outputs/predictions/baseline_seed42.parquet c83fa58832b023c97d0d0d1a0fab791e022f9aaff0e715d3583cbab693b86995 \
    'https://drive.google.com/uc?id=17nZVaJBSbmqKXcJaXVoq879Q6cqUvbxI'
fetch outputs/checkpoints/baseline_seed42.pt 72e19bc4576930e95160225666215dfed8ce9e289d38e9bb1a96232c192b295b \
    'https://drive.google.com/uc?id=1NEsfEFgBCw-glcg6PQe5hYxvNowdiI3N'
fetch outputs/predictions/baseline_ensemble5.parquet 96e742547f04972713f592899b63f60cffa18aea340d01c15557b6701b9a2d6e \
    'https://drive.google.com/uc?id=1kdFwu0YGGXf6T3wsT7ln-evbhW2Hnp_w'
