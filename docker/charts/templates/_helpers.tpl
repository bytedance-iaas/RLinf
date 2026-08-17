{{/*
Resource name. Comes from the release name so that two installs never collide;
nameOverride pins it only when you deliberately want a fixed name.
*/}}
{{- define "rlinf.fullname" -}}
{{- default .Release.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "rlinf.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Selector labels. Deliberately just `app`: this lands in the StatefulSet's IMMUTABLE
.spec.selector, and it is what the pre-Helm manifest used, so an existing deployment can be
adopted without deleting it first.
*/}}
{{- define "rlinf.selectorLabels" -}}
app: {{ include "rlinf.fullname" . }}
{{- end }}

{{- define "rlinf.labels" -}}
{{ include "rlinf.selectorLabels" . }}
app.kubernetes.io/name: {{ include "rlinf.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ include "rlinf.chart" . }}
{{- end }}

{{/*
The chart ships no default image tag, so catch it here: an empty tag would otherwise render as
"repo:" and surface much later as a confusing ImagePullBackOff.
*/}}
{{- define "rlinf.image.validate" -}}
{{- if not .Values.image.tag }}
{{- fail "image.tag is required: the chart does not track the image version. Pass --set image.tag=<tag> (and dashboard.image.tag if the dashboard is built separately)." }}
{{- end }}
{{- if not .Values.image.repository }}
{{- fail "image.repository is required." }}
{{- end }}
{{- if not .Values.persistence.storageClass }}
{{- fail "persistence.storageClass is required: an empty storageClassName means 'disable dynamic provisioning' to Kubernetes, so the claim would never bind. Pass --set persistence.storageClass=<class> (kubectl get storageclass)." }}
{{- end }}
{{- end }}

{{/*
Kubernetes Secret that supplies the dashboard's static HTTP Basic credentials.
*/}}
{{- define "rlinf.dashboard.authSecretName" -}}
{{- default (printf "%s-dashboard-auth" (include "rlinf.fullname" .)) .Values.dashboard.auth.existingSecret }}
{{- end }}

{{/*
Fail during rendering rather than starting an accidentally unauthenticated public dashboard.
*/}}
{{- define "rlinf.dashboard.auth.validate" -}}
{{- if .Values.dashboard.auth.enabled }}
{{- if not .Values.dashboard.enabled }}
{{- fail "dashboard.auth.enabled requires dashboard.enabled=true." }}
{{- end }}
{{- if not .Values.dashboard.auth.usernameKey }}
{{- fail "dashboard.auth.usernameKey is required when dashboard auth is enabled." }}
{{- end }}
{{- if not .Values.dashboard.auth.passwordKey }}
{{- fail "dashboard.auth.passwordKey is required when dashboard auth is enabled." }}
{{- end }}
{{- if not (regexMatch "^[A-Za-z0-9._-]+$" .Values.dashboard.auth.usernameKey) }}
{{- fail "dashboard.auth.usernameKey must be a valid Kubernetes Secret data key." }}
{{- end }}
{{- if not (regexMatch "^[A-Za-z0-9._-]+$" .Values.dashboard.auth.passwordKey) }}
{{- fail "dashboard.auth.passwordKey must be a valid Kubernetes Secret data key." }}
{{- end }}
{{- if eq .Values.dashboard.auth.usernameKey .Values.dashboard.auth.passwordKey }}
{{- fail "dashboard.auth.usernameKey and dashboard.auth.passwordKey must be different." }}
{{- end }}
{{- if .Values.dashboard.auth.existingSecret }}
{{- if or .Values.dashboard.auth.username .Values.dashboard.auth.password }}
{{- fail "Set dashboard.auth.existingSecret or inline username/password, not both." }}
{{- end }}
{{- else }}
{{- if not (trim .Values.dashboard.auth.username) }}
{{- fail "dashboard.auth.username is required when auth is enabled without existingSecret." }}
{{- end }}
{{- if contains ":" .Values.dashboard.auth.username }}
{{- fail "dashboard.auth.username must not contain ':'." }}
{{- end }}
{{- if not (trim .Values.dashboard.auth.password) }}
{{- fail "dashboard.auth.password is required when auth is enabled without existingSecret." }}
{{- end }}
{{- end }}
{{- else if or .Values.dashboard.auth.existingSecret .Values.dashboard.auth.username .Values.dashboard.auth.password }}
{{- fail "Set dashboard.auth.enabled=true when providing dashboard auth credentials." }}
{{- end }}
{{- end }}

{{/*
Name of the APIGInstance object the Ingress binds to.
  create=true  -> the CR this chart renders, <release>-apig
  create=false -> the CR the platform already made for the adopted gateway, which it names
                  <instance-id>-apig-instance
*/}}
{{- define "rlinf.apig.instanceObjectName" -}}
{{- if .Values.apig.create }}
{{- printf "%s-apig" (include "rlinf.fullname" .) }}
{{- else }}
{{- printf "%s-apig-instance" .Values.apig.existingId }}
{{- end }}
{{- end }}

{{- define "rlinf.apig.ingressName" -}}
{{- default (printf "%s-apig" (include "rlinf.fullname" .)) .Values.apig.ingressName }}
{{- end }}

{{/*
Ingress class. create=false has no safe default — it must match the class the adopted gateway
declares, so an empty value is a hard error rather than a silently unclaimed Ingress.
*/}}
{{- define "rlinf.apig.ingressClassName" -}}
{{- if .Values.apig.ingressClassName }}
{{- .Values.apig.ingressClassName }}
{{- else if .Values.apig.create }}
{{- printf "%s-apig" (include "rlinf.fullname" .) }}
{{- else }}
{{- fail "apig.ingressClassName is required when apig.create=false: it must match the ingress class the adopted gateway declares, and there is no default that could be correct." }}
{{- end }}
{{- end }}

{{- define "rlinf.apig.host" -}}
{{- default (printf "%s.apig.local" (include "rlinf.fullname" .)) .Values.apig.host }}
{{- end }}

{{/*
Binding annotations, derived from existingId. Empty until the gateway has an id — which for
create=true is only true after the two-step bootstrap, and until then the Ingress gets no address.
*/}}
{{- define "rlinf.apig.annotations" -}}
{{- with .Values.apig.annotations }}
{{- toYaml . }}
{{- end }}
{{- with .Values.apig.existingId }}
ingress.vke.volcengine.com/apig-instance-name: {{ include "rlinf.apig.instanceObjectName" $ | quote }}
ingress.vke.volcengine.com/loadbalancer-id: {{ . | quote }}
{{- end }}
{{- end }}

{{/*
Fail fast on the combinations that cannot work, so the error names the missing value instead of
surfacing later as an Ingress that never gets an address.
*/}}
{{- define "rlinf.apig.validate" -}}
{{- if .Values.apig.enabled }}
{{- if and .Values.dashboard.enabled (not .Values.dashboard.auth.enabled) }}
{{- fail "dashboard.auth.enabled=true is required when exposing the RLinf Dashboard through APIG." }}
{{- end }}
{{- if .Values.apig.create }}
{{- if not .Values.apig.subnetIds }}
{{- fail "apig.subnetIds is required when apig.create=true: the new gateway needs a subnet in this cluster's VPC." }}
{{- end }}
{{- if .Values.apig.existingId }}
{{- fail "apig.existingId must be empty when apig.create=true. The provisioned gateway's id is reported in the APIGInstance's status.id and the Ingress binds by ingress class, so nothing needs it back. Setting it writes spec.id, which is immutable — the admission webhook then rejects every upgrade with 'spec.id: Forbidden: forbidden to update'. Use existingId only with apig.create=false." }}
{{- end }}
{{- else }}
{{- if not .Values.apig.existingId }}
{{- fail "apig.existingId is required when apig.create=false: set it to the gateway's instance id from the APIG console, or set apig.create=true to provision a new gateway." }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}
