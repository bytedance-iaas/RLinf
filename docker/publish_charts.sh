#!/usr/bin/env bash
# Package the RLinf Helm chart and push it to the Volcengine OCI registry.
#
#   bash docker/publish_charts.sh            # package, log in, push
#   DRY_RUN=1 bash docker/publish_charts.sh  # package only
#
# Credentials come from the environment, never from this file:
#   HELM_REGISTRY_USERNAME  registry robot account
#   HELM_REGISTRY_PASSWORD  its password

set -euo pipefail

registry="ai-containers-cn-beijing.cr.volces.com"
namespace="physicalai"

# Resolve from the script's own location so the working directory does not matter.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
chart_dir="docker/charts"

# Standard library only: pyyaml is not a dependency of this repo, and adding one
# just to read two scalars would make CI installs a prerequisite for publishing.
read -r name version <<<"$(python3 - "${chart_dir}/Chart.yaml" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
def field(key):
    m = re.search(rf'^{key}:\s*"?([^"#\s]+)"?', text, re.M)
    if not m:
        sys.exit(f"error: missing {key} in {sys.argv[1]}")
    return m.group(1)
print(field("name"), field("version"))
PY
)"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
package="${work}/${name}-${version}.tgz"

helm lint "${chart_dir}" --set image.tag=ci-lint --set persistence.storageClass=ci-lint
helm package "${chart_dir}" --version "${version}" --destination "${work}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry run: would push ${name}-${version}.tgz to oci://${registry}/${namespace}"
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

helm push "${package}" "oci://${registry}/${namespace}" --registry-config "${config}"
echo "pushed oci://${registry}/${namespace}/${name}:${version}"
