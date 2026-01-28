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
| `servicemonitor.yaml` | Prometheus Operator auto-discovery (DEC-033) |
| `prometheusrule.yaml` | Alert rules packaged as PrometheusRule CRD (DEC-033) |

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

## Prometheus Operator Integration (DEC-033)

### Prerequisites

- Prometheus Operator installed (CRDs `ServiceMonitor` and `PrometheusRule` must exist)
- Verify: `kubectl get crd servicemonitors.monitoring.coreos.com prometheusrules.monitoring.coreos.com`

### Setup

The ServiceMonitor and PrometheusRule are included in `kustomization.yaml` by default. After `kubectl apply -k k8s/`, verify:

```bash
# Confirm ServiceMonitor created
kubectl get servicemonitor -A | grep cryptoscreener

# Confirm PrometheusRule created
kubectl get prometheusrule -A | grep cryptoscreener

# Check ServiceMonitor details
kubectl describe servicemonitor cryptoscreener
```

### Verifying Prometheus Discovers the Target

```bash
# Port-forward to Prometheus
kubectl port-forward svc/prometheus-operated 9090:9090 -n monitoring

# Check targets (browser or curl)
# http://localhost:9090/targets — look for "cryptoscreener" target
```

### Troubleshooting: No Target in Prometheus

1. **Label mismatch**: ServiceMonitor selects by `app.kubernetes.io/name: cryptoscreener` + `app.kubernetes.io/part-of: cryptoscreener-x`. Verify the Service has these labels:
   ```bash
   kubectl get svc cryptoscreener -o jsonpath='{.metadata.labels}'
   ```

2. **Namespace selector**: If Prometheus is configured with `serviceMonitorNamespaceSelector`, ensure the cryptoscreener namespace is included.

3. **RBAC**: Prometheus ServiceAccount needs `get`/`list`/`watch` on `endpoints` and `services` in the cryptoscreener namespace.

4. **Operator not reconciling**: Check operator logs:
   ```bash
   kubectl logs -l app.kubernetes.io/name=prometheus-operator -n monitoring
   ```

### Clusters Without Prometheus Operator

If the cluster does not have Prometheus Operator, comment out the CRD resources in `kustomization.yaml`:

```yaml
# - servicemonitor.yaml
# - prometheusrule.yaml
```

The Deployment already has scrape annotations for annotation-based Prometheus discovery:
```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "9090"
prometheus.io/path: "/metrics"
```

For alert rules without Prometheus Operator, load `monitoring/alert_rules.yml` directly into your Prometheus config.

## Security Notes

- Pod runs as non-root (UID 65534)
- Read-only root filesystem
- All capabilities dropped
- Secrets are never in configmap or committed to git
- `secret.yaml` is a template with empty values
