#!/bin/bash

set -euo pipefail

DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME}
SUPPORT_LIST=("maniskill" "openpi")
GITHUB_PREFIX=${GITHUB_PREFIX:-""}
USE_MIRRORS=${USE_MIRRORS:-0}
ASSETS=()

print_help() {
	cat <<EOF
Usage: bash download_assets.sh [--dir DIR] [--assets NAMES] [--use-mirror]

Options:
  --dir DIR         Root directory to store all downloaded assets.
					Default: \$DOWNLOAD_DIR or \$HOME.

  --assets NAMES    Comma-separated list of assets to download.

  --use-mirror      Use mirrors (HuggingFace / GitHub) for faster downloads.
					Mirrors are also picked up automatically when HF_ENDPOINT /
					GITHUB_PREFIX are already exported (e.g. by install.sh).

Examples:
  bash requirements/embodied/download_assets.sh --assets maniskill
  bash requirements/embodied/download_assets.sh --dir /opt/.assets --assets maniskill,openpi
  bash requirements/embodied/download_assets.sh --use-mirror --assets maniskill,openpi
EOF
}

# Configure HuggingFace / GitHub mirrors when requested. This is needed when the
# script is run on its own (e.g. a standalone Docker RUN) and does not inherit the
# mirror env vars that install.sh's setup_mirror exports. Values mirror install.sh.
setup_mirror() {
	if [ "$USE_MIRRORS" -eq 1 ]; then
		export UV_DEFAULT_INDEX=https://mirrors.ivolces.com/pypi/simple/
		export PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple/
		export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
		export GITHUB_PREFIX=${GITHUB_PREFIX:-https://ghfast.top/}
	fi
}

download_bridge_v2_real2sim() (
	local target_parent="$MS_ASSET_DIR/data/tasks"
	local target_dir="$target_parent/bridge_v2_real2sim_dataset"
	local sentinel="$target_dir/stages/bridge_table_1_v1.glb"

	if [ -f "$sentinel" ]; then
		echo "[download_assets] Bridge v2 Real2Sim assets already exist at $target_dir, skipping download."
		return
	fi
	if ! command -v oniond &> /dev/null; then
		echo "oniond is required to download bridge_v2_real2sim with --use-mirror." >&2
		return 1
	fi

	local staging_dir archive
	staging_dir=$(mktemp -d)
	trap 'rm -rf -- "$staging_dir"' EXIT

	(
		cd "$staging_dir"
		BUCKET=ai-infra oniond download dataset ManiSkill_bridge_v2_real2sim \
			--include bridge_v2_real2sim_dataset.zip \
			--dir "$staging_dir"
	)
	archive="$staging_dir/ManiSkill_bridge_v2_real2sim/bridge_v2_real2sim_dataset.zip"
	if [ ! -f "$archive" ]; then
		echo "oniond did not produce the expected archive: $archive" >&2
		return 1
	fi
	if ! echo "618512a205b4528cafecdad14b1788ed1130879f3064deb406516ed5b9c5ba92  $archive" \
		| sha256sum --check --status; then
		echo "Bridge v2 Real2Sim archive checksum verification failed." >&2
		return 1
	fi

	mkdir -p "$target_parent"
	rm -rf -- "$target_dir"
	unzip -q "$archive" -d "$target_parent"
	if [ ! -f "$sentinel" ]; then
		echo "Bridge v2 Real2Sim assets were not extracted to $target_dir." >&2
		return 1
	fi
	echo "[download_assets] Bridge v2 Real2Sim assets installed at $target_dir."
)

download_widowx250s() {
	local target_dir="$MS_ASSET_DIR/data/robots/widowx"
	local sentinel="$target_dir/wx250s.urdf"

	if [ -f "$sentinel" ]; then
		echo "[download_assets] WidowX250S assets already exist at $target_dir, skipping download."
		return
	fi
	if [ "$USE_MIRRORS" -eq 1 ]; then
		# ManiSkill downloads this GitHub archive with urllib, which ignores
		# git's insteadOf config. Rewrite the URL in memory for this asset.
		python - widowx250s <<'PYEOF'
import os
import sys

from mani_skill.utils.assets import data as ds
from mani_skill.utils.download_asset import main, parse_args

source = ds.DATA_SOURCES[sys.argv[1]]
github_prefix = os.environ.get("GITHUB_PREFIX", "")
if github_prefix and source.url.startswith("https://github.com"):
    source.url = github_prefix + source.url
main(parse_args([sys.argv[1], "-y"]))
PYEOF
	else
		python -m mani_skill.utils.download_asset widowx250s -y
	fi
	if [ ! -f "$sentinel" ]; then
		echo "WidowX250S assets were not installed at $target_dir." >&2
		return 1
	fi
}

download_maniskill_assets() {
	local root_dir=$1

	# ManiSkill assets
	export MS_ASSET_DIR="${root_dir}/.maniskill"
	mkdir -p "$MS_ASSET_DIR"
	# Ensure mani_skill is installed
	if ! python -c "import mani_skill" &> /dev/null; then
		echo "mani_skill is not installed. Please install it first." >&2
		exit 1
	fi
	if [ "$USE_MIRRORS" -eq 1 ]; then
		download_bridge_v2_real2sim
	else
		if [ -f "$MS_ASSET_DIR/data/tasks/bridge_v2_real2sim_dataset/stages/bridge_table_1_v1.glb" ]; then
			echo "[download_assets] Bridge v2 Real2Sim assets already exist, skipping download."
		else
			python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
		fi
	fi
	download_widowx250s

	# SAPIEN assets (PhysX)
	export PHYSX_VERSION=105.1-physx-5.3.1.patch0
	export PHYSX_DIR="${root_dir}/.sapien/physx/${PHYSX_VERSION}"
	if [ -f "$PHYSX_DIR/linux-so.zip" ] || [ -d "$PHYSX_DIR" ] && compgen -G "$PHYSX_DIR/*" > /dev/null; then
		echo "[download_assets] SAPIEN PhysX assets already exist at $PHYSX_DIR, skipping download."
	else
		mkdir -p "$PHYSX_DIR"
		wget -O "$PHYSX_DIR/linux-so.zip" "${GITHUB_PREFIX}https://github.com/sapien-sim/physx-precompiled/releases/download/${PHYSX_VERSION}/linux-so.zip"
		unzip "$PHYSX_DIR/linux-so.zip" -d "$PHYSX_DIR" && rm "$PHYSX_DIR/linux-so.zip"
	fi
}

download_openpi_assets() {
	local root_dir=$1

	export TOKENIZER_DIR="${root_dir}/.cache/openpi/"

	if [ -f "$TOKENIZER_DIR/paligemma_tokenizer.model" ]; then
		echo "[download_assets] OpenPI tokenizer already exists at $TOKENIZER_DIR, skipping download."
	else
		mkdir -p "$TOKENIZER_DIR"
		hf download RLinf/openpi_tokenizer --local-dir "$TOKENIZER_DIR"
	fi
}

parse_args() {
	while [ "$#" -gt 0 ]; do
		case "$1" in
			-h|--help)
				print_help
				exit 0
				;;
			--dir)
				if [ -z "${2:-}" ]; then
					echo "--dir requires a directory argument." >&2
					exit 1
				fi
				DOWNLOAD_DIR="$2"
				shift 2
				;;
			--assets)
				if [ -z "${2:-}" ]; then
					echo "--assets requires a comma-separated list of asset names." >&2
					exit 1
				fi
				IFS=',' read -r -a ASSETS <<<"$2"
				shift 2
				;;
			--use-mirror)
				USE_MIRRORS=1
				shift
				;;
			--*)
				echo "Unknown option: $1" >&2
				echo "Use --help to see available options." >&2
				exit 1
				;;
			*)
				echo "Unexpected positional argument: $1" >&2
				echo "Use --help to see usage." >&2
				exit 1
				;;
		esac
	done
}

main() {
	parse_args "$@"

	if [ ${#ASSETS[@]} -eq 0 ]; then
		echo "No assets specified. See --help for usage." >&2
		exit 1
	fi

	setup_mirror

	mkdir -p "$DOWNLOAD_DIR"

	for asset in "${ASSETS[@]}"; do
		case "$asset" in
			maniskill)
				download_maniskill_assets "$DOWNLOAD_DIR"
				;;
			openpi)
				download_openpi_assets "$DOWNLOAD_DIR"
				;;
			*)
				echo "Unknown asset group: $asset. Supported: ${SUPPORT_LIST[*]}" >&2
				exit 1
				;;
		esac
	done
}

main "$@"
