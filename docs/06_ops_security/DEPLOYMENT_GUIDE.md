# Deployment Guide

**Project:** In‑Play Predictor (CryptoScreener‑X) — ML + LLM  
**Date:** 2026-01-24

---


## Environments
- dev: local
- staging: replay + limited live
- prod: live

## Steps
1. Configure `.env` with keys (read-only).
2. Configure `/configs/default.yaml`.
3. Run smoke: `python -m scripts.run_live --dry-run`
4. Start service manager (systemd/docker).
5. Verify dashboards and alert channels.

## Rollback
- Switch model_version to previous
- Disable LLM
- Reduce universe size
