#!/bin/bash
#
# Resolves the GitHub accelerator prefix used when mirrors are on.
#
# Source it to get resolve_github_prefix, or run it to print the resolved
# prefix on stdout (diagnostics go to stderr):
#
#     GITHUB_PREFIX="$(bash requirements/github_mirror.sh)"
#
# Resolution order, so that one decision can be shared by everything in a
# build rather than made again per caller:
#
#   1. $GITHUB_PREFIX, when already set.
#   2. $GITHUB_PREFIX_FILE, when it exists. Written once by the Dockerfile;
#      an empty file means "go straight to github.com".
#   3. A live probe of the candidates below.
#
# Prints an empty string when GitHub should be reached directly. The trailing
# slash is part of the value: callers concatenate it with the full URL, as in
# "${GITHUB_PREFIX}https://github.com/org/repo.git".

# Which one is fastest moves around between networks and over time, and a
# mirror that has gone slow is the usual reason an install or a CI job stalls,
# so the prefix is measured rather than hardcoded.
GITHUB_PREFIX_CANDIDATES=(
    "https://gh-proxy.com/"
    "https://gh-proxy.org/"
    "https://ghfast.top/"
    "https://ghproxy.net/"
)
# Throughput, not latency, is what this ranks. What install.sh pulls from GitHub
# are hundred-megabyte wheels and release tarballs, and mirrors were measured
# holding a steady sub-second TTFB while their transfer rate swung between 0.5
# and 16 MB/s — so a latency probe reports a mirror as healthy right up to the
# point it stalls a download for minutes. Hence a real release artifact, and a
# range request large enough that transfer time dominates the measurement.
GITHUB_PREFIX_PROBE_URL="https://github.com/RLinf/apex/releases/download/25.09/apex-0.1+torch2.6-cp311-cp311-linux_x86_64.whl"
GITHUB_PREFIX_PROBE_BYTES=16777216      # 16 MiB
GITHUB_PREFIX_PROBE_TIMEOUT=10
GITHUB_PREFIX_PROBE_MIN_BYTES=1048576   # 1 MiB
GITHUB_PREFIX_FALLBACK="https://gh-proxy.org/"

# Echoes the mirror with the highest measured transfer rate. Candidates are
# probed in parallel, so the check costs one timeout at worst rather than one per
# candidate, and a healthy mirror finishes its range in about a second. Running
# them together does depress the absolute numbers as they share the link, but it
# leaves the ranking intact, which is all this needs.
# Falls back to a known-good mirror if none answers, so an offline or firewalled
# host behaves exactly as it did before.
pick_fastest_github_prefix() {
    if ! command -v curl &>/dev/null; then
        echo "$GITHUB_PREFIX_FALLBACK"
        return
    fi

    local results mirror fastest
    results="$(mktemp)"

    for mirror in "${GITHUB_PREFIX_CANDIDATES[@]}"; do
        (
            # -L is required: some mirrors answer with a 302 and an empty body,
            # so without it a working mirror looks like a failed one.
            # No `|| exit` on failure: a mirror too slow to finish inside the
            # timeout still reports what it managed to transfer, and being slow
            # is exactly what this is trying to measure.
            probe="$(curl -fsSL -r "0-$((GITHUB_PREFIX_PROBE_BYTES - 1))" \
                --max-time "$GITHUB_PREFIX_PROBE_TIMEOUT" -o /dev/null \
                -w '%{size_download} %{time_total} %{time_starttransfer}' \
                "${mirror}${GITHUB_PREFIX_PROBE_URL}" 2>/dev/null || true)"
            [ -n "$probe" ] || exit 0
            echo "$probe" | awk -v m="$mirror" -v min="$GITHUB_PREFIX_PROBE_MIN_BYTES" '
                {
                    # Subtract connect + TTFB so this is transfer rate rather
                    # than a number a low-latency, low-bandwidth mirror can win.
                    transfer = $2 - $3
                    if (transfer <= 0) transfer = $2
                    # Too little data to rate, or an error page served as 200.
                    if ($1 < min || transfer <= 0) exit
                    printf "%.0f %s\n", $1 / transfer, m
                }' >> "$results"
        ) &
    done
    wait || true

    if [ -s "$results" ]; then
        echo "GitHub mirror probe (fastest first):" >&2
        sort -rn "$results" | awk '{printf "  %-24s %.1f MB/s\n", $2, $1/1048576}' >&2
    else
        echo "No GitHub mirror answered the probe; falling back to $GITHUB_PREFIX_FALLBACK" >&2
    fi

    fastest="$(sort -rn "$results" 2>/dev/null | head -n 1 | awk '{print $2}')"
    rm -f "$results"
    echo "${fastest:-$GITHUB_PREFIX_FALLBACK}"
}

# Applies the resolution order documented at the top of this file.
resolve_github_prefix() {
    if [ -n "${GITHUB_PREFIX:-}" ]; then
        echo "$GITHUB_PREFIX"
        return
    fi

    # Existence, not content, is what makes the file authoritative: a build with
    # mirrors off writes an empty one to say "go direct", and that answer has to
    # survive a caller that would otherwise start probing.
    if [ -n "${GITHUB_PREFIX_FILE:-}" ] && [ -f "$GITHUB_PREFIX_FILE" ]; then
        tr -d '[:space:]' < "$GITHUB_PREFIX_FILE"
        echo
        return
    fi

    pick_fastest_github_prefix
}

# Only when run, not when sourced.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    resolve_github_prefix
fi
