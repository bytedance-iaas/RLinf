# rlinf chart

Helm packaging of a long-running RLinf training workload: a StatefulSet holding the GPU training
container plus a separately restartable Dashboard sidecar, a headless Service giving Ray stable
per-pod DNS, and a persistent `/workspace` for code, checkpoints and logs.

The pod idles on `sleep infinity` — you exec in and launch training by hand, the same way you
would on a bare node. The chart's job is to make the surrounding pieces (GPUs, disk, shared
memory, dashboards, optional public routing) reproducible.

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

The RLinf Dashboard supports static HTTP Basic authentication. For production,
put the credentials in an existing Kubernetes Secret and let Helm inject only
Secret references into the dashboard sidecar:

```bash
chmod 600 /secure/path/rlinf-dashboard-auth.env
kubectl create secret generic rlinf-dashboard-auth -n rlinf \
  --from-env-file=/secure/path/rlinf-dashboard-auth.env

helm upgrade --install rlinf ./docker/charts/rlinf -n rlinf \
  --set dashboard.auth.enabled=true \
  --set dashboard.auth.existingSecret=rlinf-dashboard-auth
```

The protected env file uses the Secret's key names:

```text
username=operator
password=replace-with-a-secret
```

The default Secret keys are `username` and `password`; set
`dashboard.auth.usernameKey` or `dashboard.auth.passwordKey` when an existing
Secret uses different keys. Basic Auth must stay behind the chart's HTTPS APIG
route (or another TLS-terminating ingress).

For a private test deployment, Helm can create the Secret from a protected
values file:

```yaml
dashboard:
  auth:
    enabled: true
    username: operator
    password: replace-me
```

```bash
chmod 600 private-values.yaml
helm upgrade --install rlinf ./docker/charts/rlinf -n rlinf -f private-values.yaml
```

Inline credentials are stored in Helm's release history, so an existing Secret
is preferred. For production, use an existing Secret from the first deployment;
do not migrate a Helm-created inline Secret to an `existingSecret` with the same
name, because Helm may delete the release-owned Secret during that upgrade.

Secret values are read when the dashboard container starts. A safe in-place
rotation means updating the data in the same Secret, with the same name and key
names, then restarting only that sidecar (not the StatefulSet pod and its
training process):

```bash
kubectl exec -n rlinf rlinf-0 -c dashboard -- sh -c 'kill -TERM 1'
```

Enabling or disabling authentication, changing the Secret name/key names, or
changing either image tag changes the StatefulSet pod template and therefore
recreates the whole pod. Make those changes only when no training job is
running. The unauthenticated `/healthz` endpoint contains only process liveness;
`/api/health`, the UI, API, SSE streams, media, and OpenAPI docs all require
credentials.

Authentication requires a Dashboard image built from a commit that supports
`RLINF_DASHBOARD_AUTH_MODE`. When auth is enabled, the startup probe deliberately
rejects older images that ignore the Secret instead of exposing an apparently
healthy but unauthenticated service. Build and publish a new Dashboard image,
then set `dashboard.image.tag` to that tag before enabling auth.

## Defaults worth knowing

| Value | Default | Notes |
|---|---|---|
| `dashboard.port` | `8420` | Sidecar HTTP port; also the Service port and APIG backend |
| `dashboard.auth.enabled` | `false` | Enables static Basic Auth from a Kubernetes Secret |
| `rayDashboardPort` | `8265` | Ray's own default; nothing listens until `ray start` |
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
| `dashboard.auth.existingSecret` | recommended | recommended | Secret holding the Basic username and password |
| `apig.enabled` | `true` | `true` | Off means no public entry point |
| `apig.create` | `true` | `false` | Picks which mode |
| `apig.subnetIds` | **required** | — | A subnet in this cluster's VPC |
| `apig.existingId` | filled in at step 2 | **required** | Gateway instance id, from the APIG console |
| `apig.ingressClassName` | optional | **required** | Must match the class that gateway declares, or it never claims the Ingress |
| `apig.host` | recommended | recommended | Internal placeholder host, unique per gateway. Defaults to `<release>.apig.local` |

Missing values fail at render time with a message naming the value, not later as an Ingress that
silently never gets an address.

### A. Provision a new gateway

```yaml
dashboard:
  auth:
    enabled: true
    existingSecret: rlinf-dashboard-auth

apig:
  enabled: true
  create: true
  subnetIds:
    - subnet-xxxxxxxxxxxxxxxxxxxxx       # must be in this cluster's VPC
  host: rlinf.apig.test                  # internal placeholder, unique per gateway
```

**This is a two-step bootstrap.** The gateway's id does not exist until it has been provisioned,
and the Ingress needs that id to bind. After the first install:

1. Wait for the gateway to report Running (a few minutes).
2. Read its id:
   ```bash
   kubectl get apiginstance rlinf-apig -n rlinf -o jsonpath='{.status.id}'
   ```
3. Put it in `apig.existingId` (leave `create: true`) and upgrade.

Skip step 3 and the Ingress gets no address and APIG lists no service or domain.

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
    existingSecret: rlinf-dashboard-auth

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

- Reproduce the existing **Ingress names and hosts** (`apig.ingressName`, `apig.rayIngressName`,
  `apig.host`, `apig.rayHost`). Letting them default renames the objects, which deletes and
  recreates the routes — and a recreated route can come back under a new domain.
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

**The Ray route 503s until `ray start`** runs inside the pod. Expected.

**RLinf Dashboard auth does not protect Ray.** The Ray dashboard route remains unauthenticated
and accepts arbitrary job submissions — treat that URL as a remote shell on a GPU box.
