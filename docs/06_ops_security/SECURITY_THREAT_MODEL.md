# Security Threat Model

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Assets
- Exchange API keys (even read-only)
- Trading behavior data
- Model artifacts

## Threats
- Key leakage (logs, repo, screenshots)
- Supply chain (deps)
- Remote UI exposure

## Controls
- Secrets only via env/secret manager; never in repo
- Log redaction
- Dependency pinning + audits
- Optional network isolation
