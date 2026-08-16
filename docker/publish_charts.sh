#!/usr/bin/env bash
# Package the RLinf Helm chart and push it to the Volcengine OCI registry.
#
#   bash docker/publish_charts.sh            # package, log in, push
#   DRY_RUN=1 bash docker/publish_charts.sh  # package only
#
# Credentials come from the environment, never from this file:
#   HELM_REGISTRY_USERNAME  registry robot account
#   HELM_REGISTRY_PASSWORD  its password
#
# Optional:
#   HELM_VERSION            helm to fetch when the runner has none (default: latest)
#
# Nothing here needs root or a package manager. The CI image runs as nobody on a
# release old enough that its package sources are gone, so the only things
# assumed present are bash, curl and tar.

set -euo pipefail

registry="ai-containers-cn-beijing.cr.volces.com"
namespace="physicalai"

# Resolve from the script's own location so the working directory does not matter.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
chart_dir="docker/charts"

echo "=== environment ==="
echo "user: $(id -un 2>/dev/null || echo unknown) (uid $(id -u))"
echo "os:   $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}" || uname -s) $(uname -m)"
echo "helm: $(command -v helm >/dev/null 2>&1 && helm version --short 2>&1 || echo MISSING)"
echo "curl: $(command -v curl || echo MISSING)"
echo "tar:  $(command -v tar || echo MISSING)"
echo "==================="

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# helm ships a statically linked binary, so unpacking it into the work directory
# needs no privileges — the only option available here, since the runner is
# nobody, sudo cannot elevate, and there is no usable package source.
if ! command -v helm >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64 | amd64) helm_arch=amd64 ;;
    aarch64 | arm64) helm_arch=arm64 ;;
    *) echo "error: unsupported architecture $(uname -m)" >&2; exit 1 ;;
  esac
  helm_os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  helm_version="${HELM_VERSION:-$(curl -fsSL --max-time 30 https://get.helm.sh/helm-latest-version 2>/dev/null || echo v4.2.4)}"
  helm_url="https://get.helm.sh/helm-${helm_version}-${helm_os}-${helm_arch}.tar.gz"

  echo "helm missing, fetching ${helm_url}"
  mkdir -p "${work}/helm"
  curl -fsSL --max-time 300 "${helm_url}" | tar -xz -C "${work}/helm" ||
    { echo "error: could not download helm from ${helm_url}" >&2; exit 1; }
  export PATH="${work}/helm/${helm_os}-${helm_arch}:${PATH}"

  command -v helm >/dev/null 2>&1 ||
    { echo "error: helm still not on PATH after unpacking" >&2; exit 1; }
  echo "helm ready: $(helm version --short 2>&1)"
fi

# Name and version come from Chart.yaml, which is also where helm reads them, so
# the package is simply whatever lands in this otherwise empty directory. That
# keeps the chart's identity in one place and this script out of the business of
# parsing YAML.
mkdir -p "${work}/pkg"
helm lint "${chart_dir}" --set image.tag=ci-lint --set persistence.storageClass=ci-lint
helm package "${chart_dir}" --destination "${work}/pkg"
package="$(echo "${work}/pkg"/*.tgz)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry run: would push ${package##*/} to oci://${registry}/${namespace}"
  exit 0
fi

: "${HELM_REGISTRY_USERNAME:?Set HELM_REGISTRY_USERNAME (registry robot account).}"
: "${HELM_REGISTRY_PASSWORD:?Set HELM_REGISTRY_PASSWORD.}"

# Keep the login in a throwaway config: the default ~/.config/helm would leave
# the robot credentials on a reused CI runner for the next job to find.
config="${work}/registry.json"
printf '%s' "${HELM_REGISTRY_PASSWORD}" |
  helm registry login "${registry}" --username "${HELM_REGISTRY_USERNAME}" \
    --password-stdin --registry-config "${config}"

# helm push reports the pushed reference and its digest on success.
helm push "${package}" "oci://${registry}/${namespace}" --registry-config "${config}"
