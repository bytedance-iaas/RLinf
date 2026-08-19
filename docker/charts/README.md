# rlinf chart

Helm packaging of a long-running RLinf training workload: a StatefulSet holding the GPU training
container plus a separately restartable Dashboard sidecar, a headless Service giving Ray stable
per-pod DNS, and a persistent `/workspace` for code, checkpoints and logs.

The pod idles on `sleep infinity` — you exec in and launch training by hand, the same way you
would on a bare node. The chart's job is to make the surrounding pieces (GPUs, disk, shared
memory, the dashboard, optional public routing) reproducible.

The public entry point is **optional and Volcengine-specific**: it renders APIG Ingresses and,
if asked, an `APIGInstance` CRD. On any other cluster leave `apig.enabled=false` and everything
else still applies.

## Install

```bash
helm install rlinf ./docker/charts/rlinf -n rlinf --create-namespace \
  --set image.tag=<tag> \
  --set persistence.storageClass=<block-storage-class>
```

That gives you the pod, the disk and both dashboards reachable in-cluster. APIG is off by default.
`image.tag` and `persistence.storageClass` have no defaults the chart could guess, so it fails at
render time rather than deploying something broken.

Keep the release name short and stable: object names, the PVC (`workspace-<release>-0`) and Ray's
per-pod DNS (`<release>-0.<release>.<ns>.svc`) are all derived from it. Renaming an existing
release strands its disk and starts on an empty one.

## Dashboard authentication

The dashboard serves HTTP Basic authentication from a Kubernetes Secret that you create before
installing. The chart never builds that Secret from values: Helm keeps values verbatim in the
release history, so a password passed that way stays readable to anyone who can run
`helm get values`.

```bash
kubectl create secret generic physical-ai-auth -n <namespace> \
  --from-literal=username=<user> \
  --from-literal=password=<password>
```

```yaml
dashboard:
  auth:
    enabled: true
    existingSecret: physical-ai-auth
```

The default keys are `username` and `password`; point `dashboard.auth.usernameKey` /
`dashboard.auth.passwordKey` elsewhere if the Secret uses different ones.

**The Secret has to be in the release's own namespace.** Kubernetes does not let a pod reference a
Secret from another namespace, so one shared credential across several components means creating
the same Secret name in each namespace that needs it — the name is what is common, not the object.
`physical-ai-auth` is the convention used here for that reason.

Credentials are read when the dashboard container starts. To rotate them, update the data in the
same Secret, keeping the name and keys, then restart only that sidecar so the training process
keeps running:

```bash
kubectl exec -n <namespace> <release>-0 -c dashboard -- sh -c 'kill -TERM 1'
```

Enabling or disabling auth, renaming the Secret or its keys, or changing either image tag all
change the pod template and therefore recreate the whole pod. Do those while no training job is
running. The `/healthz` endpoint stays unauthenticated and reports process liveness only;
`/api/health`, the UI, the API, SSE streams, media and the OpenAPI docs all require credentials.

Auth needs a dashboard image built from a commit that supports `RLINF_DASHBOARD_AUTH_MODE`. When
auth is on, the startup probe asserts that the protected endpoint actually answers 401, so an
older image that ignores the Secret never becomes Ready instead of quietly serving an
unauthenticated dashboard.

## Defaults worth knowing

| Value | Default | Notes |
|---|---|---|
| `dashboard.port` | `8420` | Sidecar HTTP port; also the Service port and APIG backend |
| `dashboard.auth.enabled` | `false` | Enables static Basic Auth from a Kubernetes Secret |
| `dshmSize` | `256Gi` | `/dev/shm`; the 64Mi container default causes "Bus error" |
| `persistence.size` | `500Gi` | block storage, mounted at `/workspace` |
| `resources` | 16C/128Gi → 60C/512Gi, 4 GPU | Measured for a 4-GPU pi0.5 run |
| `nodeSelector` | `{}` | No node pin — the GPU request is what schedules the pod |
| `apig.enabled` | `false` | No public entry point until you turn it on |

`dshmSize` is memory-backed and counts against `resources.limits.memory`, so keep it well under
that. Changing `dashboard.port` moves the container port, the Service port and the APIG backend
together.

## Public entry point (APIG) — two ways

APIG is off by default. Turning it on requires different values depending on how you get a
gateway:

| Value | New gateway | Existing gateway | Notes |
|---|---|---|---|
| `dashboard.auth.enabled` | `true` | `true` | Required whenever APIG exposes the RLinf Dashboard |
| `dashboard.auth.existingSecret` | **required** | **required** | Pre-created Secret with the Basic credentials |
| `apig.enabled` | `true` | `true` | Off means no public entry point |
| `apig.create` | `true` | `false` | Picks which mode |
| `apig.subnetIds` | **required** | — | A subnet in this cluster's VPC |
| `apig.existingId` | **must stay empty** | **required** | Gateway instance id, from the APIG console |
| `apig.ingressClassName` | optional | **required** | Must match the class that gateway declares, or it never claims the Ingress |
| `apig.host` | recommended | recommended | Internal placeholder host, unique per gateway. Defaults to `<release>.apig.local` |

Missing values fail at render time with a message naming the value, not later as an Ingress that
silently never gets an address.

### A. Provision a new gateway

```yaml
dashboard:
  auth:
    enabled: true
    existingSecret: physical-ai-auth

apig:
  enabled: true
  create: true
  subnetIds:
    - subnet-xxxxxxxxxxxxxxxxxxxxx       # must be in this cluster's VPC
  host: rlinf.apig.test                  # internal placeholder, unique per gateway
```

One install is enough — there is nothing to feed back. Provisioning takes a few minutes; watch it
with:

```bash
kubectl get apiginstance rlinf-apig -n rlinf
```

Once it reports `Running`, its id appears in `status.id` and the Ingress picks the gateway up by
ingress class, without that id ever being restated in the values.

⚠️ **Do not copy that id into `apig.existingId`.** `existingId` writes `spec.id`, which the CRD
treats as immutable, and the admission webhook then rejects every subsequent upgrade:

```text
spec.id: Forbidden: forbidden to update, old: , new: <id>
```

The release stays `failed` until the value is removed again. The chart now refuses to render this
combination up front. `existingId` belongs to `create: false` only.

Gateway sizing (`instanceSpecCode`, `clbSpecCode`, `replicas`, `publicNetworkBillingType`,
`publicNetworkBandwidth`) is all optional — empty means the platform picks. If you do want to
control it, `1c2g` (IPv4) with `traffic` billing and a 200 Mbps cap is plenty: the gateway proxies
UI traffic only, never training data.

⚠️ A gateway provisioned this way is **deleted by `helm uninstall`**, taking its
`*.volceapi.com` domain with it. A reinstall comes back under a different name.

### B. Adopt an existing gateway

```yaml
dashboard:
  auth:
    enabled: true
    existingSecret: physical-ai-auth

apig:
  enabled: true
  create: false
  existingId: gd9xxxxxxxxxxxxxxxxxx      # instance id from the APIG console
  ingressClassName: apig                 # must match what that gateway declares
  host: rlinf.apig.test                  # must not collide with a host already on it
```

One step, and `helm uninstall` leaves the gateway alone. Use this for anything whose URL people
have bookmarked.

⚠️ The gateway's name in the APIG console is frequently **not** the name Kubernetes shows. The
in-cluster object is `<instance-id>-apig-instance`, while the console lists whatever the gateway
was originally named — often after whichever service created it. Match on the instance id, not the
name.

## Adopting objects created outside Helm

If the workload already exists from a raw `kubectl apply`, a plain install fails with "resource
already exists". Helm can take the objects over in place, which does **not** restart the pod as
long as the rendered spec matches what is running:

```bash
helm install rlinf ./docker/charts/rlinf -n rlinf --take-ownership -f my-values.yaml
```

Two things make the difference between a silent adoption and a surprise:

- Reproduce the existing **Ingress name and host** (`apig.ingressName`, `apig.host`). Letting them
  default renames the object, which deletes and recreates the route — and a recreated route can
  come back under a new domain.
- Keep `.spec.selector` identical. The chart selects on `app: <release>`, matching the convention
  a hand-written manifest usually uses; if yours differs, the StatefulSet cannot be adopted at all
  because that field is immutable.

Confirm there is no real change before committing to it — render and diff against the cluster:

```bash
helm template rlinf ./docker/charts/rlinf -n rlinf -f my-values.yaml | kubectl diff -f -
```

Letting the Ingress names default instead would delete and recreate them, and a recreated route
can come back under a new domain.

The Namespace is deliberately not in the chart (`--create-namespace` handles it), so
`helm uninstall` can never take the whole namespace with it.

## Things that will bite you

**PVC is immutable.** `persistence.size` and `storageClass` live in a `volumeClaimTemplate`, which
Kubernetes forbids changing after creation — an upgrade that touches them is rejected. Resize by
expanding the PVC directly (if the class supports online expansion) and updating the value to match, or
`kubectl delete sts --cascade=orphan` (keeps pods and PVCs) before reinstalling.

**PVC survives uninstall.** `workspace-<release>-0` is left behind with all checkpoints, and a
reinstall rebinds it. Delete it by hand to reclaim the disk.

**APIG hosts are routing keys, not DNS.** The real URLs are auto-assigned `*.volceapi.com`
domains, visible only at <https://console.volcengine.com/veapig> → instance → Service list.
`kubectl` cannot read them. Never put an assigned `*.volceapi.com` name into `apig.host`.

**Do not verify APIG by curling the CLB IP with a Host header.** It answers 401 with error code
010002 for every host, working ones included. Test the assigned domain instead.

**Only the dashboard is published.** Nothing else in the pod gets a route; reach anything else
with `kubectl port-forward`.
