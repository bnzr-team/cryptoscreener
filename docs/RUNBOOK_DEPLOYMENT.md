# Deployment Runbook — CryptoScreener-X

DEC-029: Deployment Readiness MVP
DEC-030: Production Readiness v1.5

---

## Prerequisites

- Docker 24+ and docker compose v2
- Outbound internet (Binance public WS/REST)

## Build

```bash
docker build -t cryptoscreener .
```

## Run

```bash
# Full stack (app + Prometheus)
docker compose up -d

# App only
docker run -d -p 9090:9090 cryptoscreener --top 50 --metrics-port 9090
```

## Verify

```bash
# Health (process alive) — always 200 unless dead
curl http://localhost:9090/healthz
# → {"status":"ok","uptime_s":12.3,"ws_connected":true,"last_event_ts":1706400000}

# Readiness (pipeline ready to serve) — 200 when ready, 503 when not
curl http://localhost:9090/readyz
# → {"ready":true,"ws_connected":true,"last_event_age_s":2.1}

# Prometheus metrics
curl http://localhost:9090/metrics | head -20

# Prometheus UI (if using compose)
open http://localhost:9091
```

## Configuration

| Flag | Default | Description |
|---|---|---|
| `--top N` | 50 | Top N symbols by 24h volume |
| `--symbols X,Y` | — | Explicit symbol list |
| `--cadence MS` | 1000 | Snapshot cadence (ms) |
| `--duration-s S` | — | Auto-stop after S seconds |
| `--metrics-port P` | 9090 | Prometheus /metrics port (0=disabled) |
| `--llm` | off | Enable LLM explanations (needs ANTHROPIC_API_KEY) |
| `--dry-run` | off | Validate config + start metrics server, then exit |
| `--graceful-timeout-s S` | 10 | Seconds to wait for in-flight work on shutdown |

## Key Metrics

| Metric | Type | Alert threshold |
|---|---|---|
| `cryptoscreener_pipeline_event_queue_depth` | Gauge | >8000 |
| `cryptoscreener_pipeline_events_dropped_total` | Counter | >0 |
| `cryptoscreener_pipeline_tick_drift_ms` | Gauge | >2000 |
| `cryptoscreener_pipeline_rss_mb` | Gauge | >512 |
| `cryptoscreener_cb_transitions_closed_to_open_total` | Counter | >0 |
| `cryptoscreener_ws_total_disconnects_total` | Counter | rate>0.1/s |

## Fault Flags (Test Only)

These are **disabled by default** and require `ALLOW_FAULTS=1` or `ENV=dev` to enable:

| Flag | Effect |
|---|---|
| `--fault-drop-ws-every-s N` | Force WS disconnect every N seconds |
| `--fault-slow-consumer-ms N` | Add N ms delay per event |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/healthz` returns `{"status":"stopped"}` | Pipeline crashed or shutting down | Check container logs |
| `ws_connected: false` | Binance WS unreachable | Check network, circuit breaker state |
| High `event_queue_depth` | Consumer too slow | Check tick drift, RSS |
| `events_dropped > 0` | Sustained overload | Reduce symbol count or increase cadence |

### Readiness stuck 503

1. `curl /readyz` — check `reason` field
2. If `"no WS shards connected"`: check `ws_total_disconnects`, circuit breaker state, network
3. If `"no events received yet"`: pipeline just started, wait for warmup (~15s)
4. If `"stale"`: events stopped flowing. Check WS reconnects, Binance status page, circuit breaker
5. Verify `curl /healthz` returns `"status":"ok"` (process is alive)

### Reconnect storm

1. Check `cryptoscreener_ws_total_disconnects_total` rate — should be <0.1/s
2. Check `cryptoscreener_ws_total_reconnect_attempts_total` rate
3. If circuit breaker open: `cryptoscreener_cb_transitions_closed_to_open_total` incrementing
4. Check Binance status: https://www.binance.com/en/support/announcement
5. Consider reducing symbol count to reduce shard pressure

### Backpressure drops increasing

1. Check `cryptoscreener_pipeline_events_dropped_total` rate
2. Check `cryptoscreener_pipeline_event_queue_depth` — sustained near 10,000 = overload
3. Check `cryptoscreener_pipeline_tick_drift_ms` — >2s indicates consumer can't keep up
4. Check `cryptoscreener_pipeline_rss_mb` — >512MB may indicate memory pressure
5. **Fix**: Reduce `--top N` (fewer symbols) or increase `--cadence` (less frequent snapshots)

## Shutdown

```bash
docker compose down
# or
docker stop <container>
```
