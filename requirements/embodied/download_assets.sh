#!/bin/bash

set -euo pipefail

DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME}
SUPPORT_LIST=("maniskill" "openpi" "libero")
GITHUB_PREFIX=${GITHUB_PREFIX:-""}
USE_MIRRORS=${USE_MIRRORS:-0}
ONIOND_BUCKET=${ONIOND_BUCKET:-ai-infra}
ASSETS=()

print_help() {
	cat <<EOF
Usage: bash download_assets.sh [--dir DIR] [--assets NAMES] [--use-mirror]

Options:
  --dir DIR         Root directory to store all downloaded assets.
					Default: \$DOWNLOAD_DIR or \$HOME.

  --assets NAMES    Comma-separated list of assets to download.
					Supported: ${SUPPORT_LIST[*]}.

  --use-mirror      Use mirrors for faster downloads. On this network the
					HuggingFace mirror is unreliable, so assets that have a copy
					in the Volcengine object store are pulled with oniond
					instead; the rest fall back to HF_ENDPOINT / GITHUB_PREFIX.
					Mirrors are also picked up automatically when HF_ENDPOINT /
					GITHUB_PREFIX are already exported (e.g. by install.sh).

Environment:
  ONIOND_BUCKET     Object-store bucket oniond reads from. Default: ai-infra.

Examples:
  bash requirements/embodied/download_assets.sh --assets maniskill
  bash requirements/embodied/download_assets.sh --dir /opt/.assets --assets maniskill,openpi
  bash requirements/embodied/download_assets.sh --use-mirror --assets maniskill,openpi,libero
EOF
}

# Configure HuggingFace / GitHub mirrors when requested. This is needed when the
# script is run on its own (e.g. a standalone Docker RUN) and does not inherit the
# mirror env vars that install.sh's setup_mirror exports. Values mirror install.sh.
setup_mirror() {
	if [ "$USE_MIRRORS" -eq 1 ]; then
		export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}
		export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
		if [ -z "${GITHUB_PREFIX:-}" ]; then
			# Prefer a prefix already resolved for this build (the Dockerfile
			# writes one), then the shared resolver next to this script, and only
			# then the historical default. This script is also copied to
			# /usr/local/bin in the image, where the sibling path does not exist.
			_mirror_helper="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/github_mirror.sh"
			if [ -n "${GITHUB_PREFIX_FILE:-}" ] && [ -f "$GITHUB_PREFIX_FILE" ]; then
				GITHUB_PREFIX="$(tr -d '[:space:]' < "$GITHUB_PREFIX_FILE")"
			elif [ -f "$_mirror_helper" ]; then
				# shellcheck source=requirements/github_mirror.sh
				source "$_mirror_helper"
				GITHUB_PREFIX="$(resolve_github_prefix)"
			else
				GITHUB_PREFIX="https://gh-proxy.org/"
			fi
		fi
		export GITHUB_PREFIX
	fi
}

# Ride out transient network / HF Hub errors (e.g. HTTP 429 rate limits during
# parallel Docker builds) with exponential backoff.
retry_cmd() {
	local max=5 delay=15 attempt=1
	until "$@"; do
		if [ "$attempt" -ge "$max" ]; then
			echo "[download_assets] '$*' failed after ${max} attempts" >&2
			return 1
		fi
		local wait=$((delay + RANDOM % 10))
		echo "[download_assets] '$*' failed (attempt ${attempt}/${max}); retrying in ${wait}s" >&2
		sleep "$wait"
		attempt=$((attempt + 1))
		delay=$((delay * 2))
	done
}

# Whether the Volcengine object store is usable for asset downloads. It only
# exists on the internal network, so callers outside it fall back to HuggingFace.
oniond_available() {
	[ "$USE_MIRRORS" -eq 1 ] && command -v oniond &> /dev/null
}

# Fetch one repo from the object store into DEST_PARENT/<name>.
#
# Usage: oniond_fetch {model|dataset} NAME DEST_PARENT [ONIOND_ARGS...]
#
# oniond is resumable and always writes to <dir>/<name>, so the caller gets the
# same layout as the HuggingFace repo it mirrors.
oniond_fetch() {
	local kind="$1" name="$2" dest_parent="$3"
	shift 3
	mkdir -p "$dest_parent"
	BUCKET="$ONIOND_BUCKET" oniond download "$kind" "$name" --dir "$dest_parent" "$@"
}

# Fetch the Bridge v2 Real2Sim dataset from the Volcengine object store via
# oniond. Its upstream home is a HuggingFace archive that the mirror network
# cannot reach, so on --use-mirror we pull the internal copy and verify it
# against a pinned checksum before unpacking.
#
# Runs in a subshell (`(` … `)`) so the staging-dir EXIT trap cannot clobber the
# caller's traps.
download_bridge_v2_real2sim_oniond() (
	local target_parent="$MS_ASSET_DIR/data/tasks"
	local target_dir="$target_parent/bridge_v2_real2sim_dataset"

	if ! command -v oniond &> /dev/null; then
		echo "oniond is required to download bridge_v2_real2sim with --use-mirror." >&2
		return 1
	fi

	local staging_dir archive
	staging_dir=$(mktemp -d)
	trap 'rm -rf -- "$staging_dir"' EXIT

	(
		cd "$staging_dir"
		BUCKET="$ONIOND_BUCKET" oniond download dataset ManiSkill_bridge_v2_real2sim \
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
)

download_bridge_v2_real2sim() {
	local sentinel="$MS_ASSET_DIR/data/tasks/bridge_v2_real2sim_dataset/stages/bridge_table_1_v1.glb"

	if [ -f "$sentinel" ]; then
		echo "[download_assets] Bridge v2 Real2Sim assets already exist, skipping download."
		return
	fi
	if [ "$USE_MIRRORS" -eq 1 ]; then
		retry_cmd download_bridge_v2_real2sim_oniond
	else
		retry_cmd python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
	fi
	if [ ! -f "$sentinel" ]; then
		echo "Bridge v2 Real2Sim assets were not installed at $(dirname "$(dirname "$sentinel")")." >&2
		return 1
	fi
	echo "[download_assets] Bridge v2 Real2Sim assets installed."
}

# ManiSkill fetches this GitHub archive with urllib, which ignores git's
# insteadOf config, so rewrite the URL in memory before downloading. Kept in a
# function (rather than an inline heredoc) so retry_cmd can re-run it — a retried
# heredoc would feed python an already-consumed stdin.
download_widowx250s_mirrored() {
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
}

download_widowx250s() {
	local target_dir="$MS_ASSET_DIR/data/robots/widowx"
	local sentinel="$target_dir/wx250s.urdf"

	if [ -f "$sentinel" ]; then
		echo "[download_assets] WidowX250S assets already exist at $target_dir, skipping download."
		return
	fi
	if [ "$USE_MIRRORS" -eq 1 ]; then
		retry_cmd download_widowx250s_mirrored
	else
		retry_cmd python -m mani_skill.utils.download_asset widowx250s -y
	fi
	if [ ! -f "$sentinel" ]; then
		echo "WidowX250S assets were not installed at $target_dir." >&2
		return 1
	fi
}

download_maniskill_assets() {
	local root_dir=$1

	# ManiSkill assets. Each asset checks its own sentinel file rather than
	# short-circuiting on the whole directory, so a run interrupted partway
	# through resumes instead of reporting everything as present.
	export MS_ASSET_DIR="${root_dir}/.maniskill"
	mkdir -p "$MS_ASSET_DIR"
	# Ensure mani_skill is installed
	if ! python -c "import mani_skill" &> /dev/null; then
		echo "mani_skill is not installed. Please install it first." >&2
		exit 1
	fi
	download_bridge_v2_real2sim
	download_widowx250s

	# SAPIEN assets (PhysX)
	export PHYSX_VERSION=105.1-physx-5.3.1.patch0
	export PHYSX_DIR="${root_dir}/.sapien/physx/${PHYSX_VERSION}"
	if [ -f "$PHYSX_DIR/linux-so.zip" ] || [ -d "$PHYSX_DIR" ] && compgen -G "$PHYSX_DIR/*" > /dev/null; then
		echo "[download_assets] SAPIEN PhysX assets already exist at $PHYSX_DIR, skipping download."
	else
		mkdir -p "$PHYSX_DIR"
		retry_cmd wget -O "$PHYSX_DIR/linux-so.zip" "${GITHUB_PREFIX}https://github.com/sapien-sim/physx-precompiled/releases/download/${PHYSX_VERSION}/linux-so.zip"
		unzip "$PHYSX_DIR/linux-so.zip" -d "$PHYSX_DIR" && rm "$PHYSX_DIR/linux-so.zip"
	fi
}

download_openpi_assets() {
	local root_dir=$1

	export TOKENIZER_DIR="${root_dir}/.cache/openpi/"
	# The repo stores the tokenizer under big_vision/, so that is where both the
	# HuggingFace and oniond paths land it.
	local sentinel="$TOKENIZER_DIR/big_vision/paligemma_tokenizer.model"

	if [ -f "$sentinel" ]; then
		echo "[download_assets] OpenPI tokenizer already exists at $TOKENIZER_DIR, skipping download."
		return
	fi

	mkdir -p "$TOKENIZER_DIR"
	if oniond_available; then
		# oniond writes to <dir>/<name>, so stage into a temp parent and move the
		# repo contents into TOKENIZER_DIR to match the HuggingFace layout.
		local staging
		staging=$(mktemp -d)
		if retry_cmd oniond_fetch model openpi_tokenizer "$staging"; then
			cp -a "$staging/openpi_tokenizer/." "$TOKENIZER_DIR/"
			rm -rf -- "$staging"
		else
			rm -rf -- "$staging"
			echo "[download_assets] oniond could not fetch the OpenPI tokenizer." >&2
			return 1
		fi
	else
		retry_cmd hf download RLinf/openpi_tokenizer --local-dir "$TOKENIZER_DIR"
	fi

	if [ ! -f "$sentinel" ]; then
		echo "[download_assets] OpenPI tokenizer was not installed at $TOKENIZER_DIR." >&2
		return 1
	fi
	echo "[download_assets] OpenPI tokenizer installed at $TOKENIZER_DIR."
}

# Fetch the LIBERO simulation assets (~286MB of meshes/scenes/textures). The
# rlinf-libero wheel does not ship them; its libero-download-assets command
# normally pulls them from HuggingFace, but it also accepts a pre-existing tree
# via LIBERO_ASSET_PATH and just symlinks to it. Staging one shared copy here
# means the per-venv installs symlink instead of each downloading 286MB.
download_libero_assets() {
	local root_dir=$1

	export LIBERO_ASSETS_DIR="${root_dir}/.libero_assets/LIBERO-assets"
	# assets_are_present() in rlinf-libero keys off the scenes/ subdirectory.
	local sentinel="$LIBERO_ASSETS_DIR/scenes"

	if [ -d "$sentinel" ]; then
		echo "[download_assets] LIBERO assets already exist at $LIBERO_ASSETS_DIR, skipping download."
		return
	fi
	if ! oniond_available; then
		echo "[download_assets] LIBERO assets need oniond (--use-mirror); libero-download-assets will fetch them from HuggingFace instead."
		return
	fi

	retry_cmd oniond_fetch dataset LIBERO-assets "${root_dir}/.libero_assets"
	if [ ! -d "$sentinel" ]; then
		echo "[download_assets] LIBERO assets were not installed at $LIBERO_ASSETS_DIR." >&2
		return 1
	fi
	echo "[download_assets] LIBERO assets installed at $LIBERO_ASSETS_DIR."
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
			libero)
				download_libero_assets "$DOWNLOAD_DIR"
				;;
			*)
				echo "Unknown asset group: $asset. Supported: ${SUPPORT_LIST[*]}" >&2
				exit 1
				;;
		esac
	done
}

main "$@"
