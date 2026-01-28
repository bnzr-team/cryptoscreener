# Runbook: Secrets Management (DEC-034)

## Overview

CryptoScreener requires three secrets at runtime:

| Env Var | Purpose |
|---|---|
| `BINANCE_API_KEY` | Binance REST/WS authentication |
| `BINANCE_SECRET_KEY` | Binance request signing |
| `ANTHROPIC_API_KEY` | LLM explainer service |

These are mounted into the pod via the Kubernetes Secret `cryptoscreener-secrets`.

## Secret Provisioning Options

### Option A: External Secrets Operator (Recommended for production)

Uses ESO to sync secrets from a cloud backend into K8s automatically.

#### Prerequisites

1. Install External Secrets Operator: https://external-secrets.io/latest/introduction/getting-started/
2. Verify CRDs exist:
   ```bash
   kubectl get crd externalsecrets.external-secrets.io secretstores.external-secrets.io
   ```

#### Setup

1. **Configure your backend** in `k8s/secretstore.yaml`:
   - AWS Secrets Manager: uncomment the `aws:` block, set region + credentials
   - HashiCorp Vault: uncomment the `vault:` block, set server URL + auth
   - For dev/test: the Kubernetes backend is pre-configured (reads from a regular K8s Secret)

2. **Store secrets in your backend** using the keys:
   - `cryptoscreener/binance-api-key`
   - `cryptoscreener/binance-secret-key`
   - `cryptoscreener/anthropic-api-key`

3. **Enable ESO resources** in `k8s/kustomization.yaml`:
   ```yaml
   - secretstore.yaml
   - externalsecret.yaml
   ```

4. **Apply**:
   ```bash
   kubectl apply -k k8s/
   ```

5. **Verify**:
   ```bash
   # Check ExternalSecret status
   kubectl get externalsecret cryptoscreener-secrets -o wide

   # Should show STATUS=SecretSynced
   kubectl describe externalsecret cryptoscreener-secrets
   ```

#### Troubleshooting: ExternalSecret not syncing

1. **SecretStore unreachable**: Check provider credentials and network access:
   ```bash
   kubectl describe secretstore cryptoscreener-store
   ```

2. **Key not found**: Verify the remote key paths match what's in your backend:
   ```bash
   kubectl get externalsecret cryptoscreener-secrets -o jsonpath='{.status.conditions}'
   ```

3. **RBAC**: For Kubernetes backend, the `default` ServiceAccount needs read access to the source Secret.

### Option B: Manual kubectl (Simple / dev)

Create the secret directly:

```bash
kubectl create secret generic cryptoscreener-secrets \
  --from-literal=BINANCE_API_KEY=<your-key> \
  --from-literal=BINANCE_SECRET_KEY=<your-secret> \
  --from-literal=ANTHROPIC_API_KEY=<your-key>
```

### Option C: Sealed Secrets

If using Bitnami Sealed Secrets, encrypt with `kubeseal` and commit the SealedSecret manifest.

## Secret Hygiene

### CI Guard

The `secret_guard.yml` CI workflow scans every PR for leaked secrets. It detects:
- AWS access key IDs (`AKIA...`)
- Long hex strings (>= 40 chars)
- Env var assignments with base64-encoded values

If the guard fails, check the CI output for the file and line number, then remove or redact the value.

### Runtime Redaction

`scripts/run_live.py` defines `REDACTED_ENV_VARS` (line 70) which prevents logging of:
- `BINANCE_API_KEY`
- `BINANCE_SECRET_KEY`
- `ANTHROPIC_API_KEY`

### Rules

1. **Never commit real secret values** to git (even in `secret.yaml` — it's a template with `REPLACE_ME` placeholders)
2. **Never log secrets** — `REDACTED_ENV_VARS` enforces this at runtime
3. **Rotate secrets** via your backend; ESO's `refreshInterval: 1h` picks up changes automatically
4. **Audit access** via your cloud provider's secret access logs

## Manifest Reference

| File | Purpose |
|---|---|
| `k8s/secret.yaml` | Template only — never apply directly with real values |
| `k8s/secretstore.yaml` | ESO SecretStore (configure your backend provider) |
| `k8s/externalsecret.yaml` | ESO ExternalSecret (syncs 3 keys into `cryptoscreener-secrets`) |
