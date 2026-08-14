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
{{- fail "image.tag is required: the chart does not track the image version. Pass --set image.tag=<tag>." }}
{{- end }}
{{- if not .Values.image.repository }}
{{- fail "image.repository is required." }}
{{- end }}
{{- if not .Values.persistence.storageClass }}
{{- fail "persistence.storageClass is required: an empty storageClassName means 'disable dynamic provisioning' to Kubernetes, so the claim would never bind. Pass --set persistence.storageClass=<class> (kubectl get storageclass)." }}
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

{{- define "rlinf.apig.rayIngressName" -}}
{{- default (printf "%s-ray" (include "rlinf.apig.ingressName" .)) .Values.apig.rayIngressName }}
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
Ray needs a host of its own: one managed domain is issued per host, so sharing would leave the
two dashboards with no way to be told apart.
*/}}
{{- define "rlinf.apig.rayHost" -}}
{{- default (printf "ray.%s" (include "rlinf.apig.host" .)) .Values.apig.rayHost }}
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
{{- if .Values.apig.create }}
{{- if not .Values.apig.subnetIds }}
{{- fail "apig.subnetIds is required when apig.create=true: the new gateway needs a subnet in this cluster's VPC." }}
{{- end }}
{{- else }}
{{- if not .Values.apig.existingId }}
{{- fail "apig.existingId is required when apig.create=false: set it to the gateway's instance id from the APIG console, or set apig.create=true to provision a new gateway." }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}
