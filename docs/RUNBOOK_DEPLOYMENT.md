# Deployment Runbook — CryptoScreener-X

DEC-029: Deployment Readiness MVP

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
# Health endpoint
curl http://localhost:9090/healthz
# → {"status":"ok","uptime_s":12.3,"ws_connected":true,"last_event_ts":1706400000}

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

These are **disabled by default** and exist for soak/chaos testing:

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

## Shutdown

```bash
docker compose down
# or
docker stop <container>
```
