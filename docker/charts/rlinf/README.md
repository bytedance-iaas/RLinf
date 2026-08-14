# rlinf chart

Helm packaging for a long-running RLinf training workload: one GPU StatefulSet,
stable per-pod DNS for Ray, a persistent `/workspace`, and optional Volcengine
APIG access to Ray's dashboard.

```bash
helm install rlinf ./docker/charts/rlinf -n rlinf --create-namespace \
  --set image.tag=<tag> \
  --set persistence.storageClass=<block-storage-class>
```

The pod idles on `sleep infinity`; exec into it and launch training by hand.
APIG is disabled by default, so the chart renders on non-Volcengine clusters.
Keep the release name stable because it determines the StatefulSet, PVC, and
Ray DNS names.

The workspace claim is retained after uninstall. Its storage class and size are
part of the StatefulSet's immutable volume claim template; resize the PVC in
place or orphan the StatefulSet before changing them.

When APIG is enabled, the Ray dashboard route is unauthenticated and can submit
jobs. Treat its public URL as a remote shell on the GPU workload.
