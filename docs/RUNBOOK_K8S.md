# Kubernetes Runbook — CryptoScreener-X

> DEC-031: Kubernetes Manifests MVP

## Prerequisites

- `kubectl` configured with cluster access
- Container image pushed to a registry accessible by the cluster
- Secrets created out-of-band (never committed to git)

## Quick Start

```bash
# 1. Create secrets (one-time)
kubectl create secret generic cryptoscreener-secrets \
  --from-literal=BINANCE_API_KEY=<key> \
  --from-literal=BINANCE_SECRET_KEY=<secret> \
  --from-literal=ANTHROPIC_API_KEY=<key>

# 2. Update image in deployment.yaml or use kustomize overlay
# Edit k8s/deployment.yaml: image: your-registry/cryptoscreener:v1.0.0

# 3. Apply manifests
kubectl apply -k k8s/

# 4. Verify
kubectl rollout status deployment/cryptoscreener
kubectl get pods -l app=cryptoscreener
```

## Manifest Overview

| File | Purpose |
|---|---|
| `deployment.yaml` | Pod spec with probes, resources, security context |
| `service.yaml` | ClusterIP service exposing metrics port 9090 |
| `configmap.yaml` | Non-secret pipeline configuration |
| `secret.yaml` | **Template only** — create via `kubectl create secret` |
| `kustomization.yaml` | Kustomize entrypoint |

## Probe Semantics

| Probe | Endpoint | Meaning |
|---|---|---|
| Liveness | `GET /healthz` | Process alive. Failure → pod restart. |
| Readiness | `GET /readyz` | Pipeline ready (WS connected + fresh events). Failure → removed from Service endpoints. |

**Liveness** starts checking at 5s, every 30s. 3 consecutive failures trigger restart.

**Readiness** starts checking at 10s, every 10s. 3 consecutive failures remove the pod from traffic. The pod is NOT restarted — it stays running and can recover when WS reconnects.

## Configuration

Modify `configmap.yaml` to change pipeline settings. After editing:

```bash
kubectl apply -k k8s/
kubectl rollout restart deployment/cryptoscreener
```

## Troubleshooting

### Pod stuck in CrashLoopBackOff

```bash
kubectl logs deployment/cryptoscreener --previous
kubectl describe pod -l app=cryptoscreener
```

Common causes:
- Missing secrets (`cryptoscreener-secrets` not created)
- Invalid config values (check `__post_init__` validation errors in logs)

### Readiness probe failing (0/1 READY)

```bash
kubectl exec deployment/cryptoscreener -- \
  python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:9090/readyz').read())"
```

Common causes:
- WS not connected yet (wait for warmup)
- Event staleness exceeded `READINESS_STALENESS_S` (check WS connectivity)
- See [RUNBOOK_DEPLOYMENT.md](RUNBOOK_DEPLOYMENT.md) "Readiness stuck 503" section

### Checking metrics

```bash
kubectl port-forward svc/cryptoscreener 9090:9090
curl http://localhost:9090/metrics
curl http://localhost:9090/healthz
curl http://localhost:9090/readyz
```

### Resource pressure

If OOMKilled, increase `resources.limits.memory` in `deployment.yaml`.
If CPU-throttled, increase `resources.limits.cpu`.

Monitor via:
```bash
kubectl top pod -l app=cryptoscreener
```

## Security Notes

- Pod runs as non-root (UID 65534)
- Read-only root filesystem
- All capabilities dropped
- Secrets are never in configmap or committed to git
- `secret.yaml` is a template with empty values
