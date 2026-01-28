# Grafana Dashboards — Runbook (DEC-036)

## Overview

CryptoScreener ships two Grafana dashboards as JSON files:

| Dashboard | File | UID | Purpose |
|---|---|---|---|
| Overview | `monitoring/grafana/dashboards/cryptoscreener-overview.json` | `cryptoscreener-overview` | WS health, circuit breaker, REST governor |
| Backpressure | `monitoring/grafana/dashboards/cryptoscreener-backpressure.json` | `cryptoscreener-backpressure` | Queue depths, drops, tick drift, RSS |

Both dashboards use template variables (`$namespace`, `$pod`, `$job`) and a `$datasource` selector for multi-cluster support.

## Prerequisites

- Grafana 9+ (schemaVersion 39)
- A Prometheus datasource scraping the `cryptoscreener` service (see DEC-033/DEC-035 for setup)
- All 18 `cryptoscreener_*` metrics exposed on `GET /metrics` (DEC-024/DEC-028)

## Import Steps

### Option A: Manual Import (UI)

1. Open Grafana → **Dashboards** → **New** → **Import**
2. Click **Upload dashboard JSON file**
3. Select `monitoring/grafana/dashboards/cryptoscreener-overview.json`
4. Choose your Prometheus datasource
5. Click **Import**
6. Repeat for `cryptoscreener-backpressure.json`

### Option B: Provisioning (File-based)

Add to your Grafana provisioning config (e.g. `/etc/grafana/provisioning/dashboards/cryptoscreener.yaml`):

```yaml
apiVersion: 1
providers:
  - name: cryptoscreener
    orgId: 1
    folder: CryptoScreener
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /path/to/monitoring/grafana/dashboards
      foldersFromFilesStructure: false
```

Restart Grafana or wait for the provisioning scan interval.

### Option C: docker-compose (Development)

If using the existing `docker-compose.yml`, mount the dashboards directory:

```yaml
services:
  grafana:
    image: grafana/grafana:11-oss
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Template Variables

| Variable | Type | Description |
|---|---|---|
| `$datasource` | Datasource | Prometheus datasource selector |
| `$namespace` | Query | Kubernetes namespace (from metric labels) |
| `$pod` | Query | Pod name (filtered by `$namespace`) |
| `$job` | Query | Prometheus job name (filtered by `$namespace`) |

All variables default to "All" and support regex matching.

## Dashboard Panels

### Overview Dashboard

| Row | Panel | Metric(s) | Type |
|---|---|---|---|
| WS Health | Reconnects & Disconnects Rate | `ws_total_reconnect_attempts`, `ws_total_disconnects` | timeseries |
| WS Health | Ping Timeouts & Subscribe Delays | `ws_total_ping_timeouts`, `ws_total_subscribe_delayed` | timeseries |
| WS Health | 4× stat panels | 5m increase of each WS counter | stat |
| Circuit Breaker | CB Transitions | `cb_transitions_closed_to_open` | timeseries |
| Circuit Breaker | CB Last OPEN Duration | `cb_last_open_duration_ms` | timeseries |
| REST Governor | Queue Depth | `gov_current_queue_depth`, `gov_max_queue_depth` | timeseries |
| REST Governor | Concurrency | `gov_current_concurrent`, `gov_max_concurrent_requests` | timeseries |
| REST Governor | Request Rate | `gov_requests_allowed`, `gov_requests_dropped` | timeseries |
| REST Governor | Queue Saturation | depth / max_depth | gauge |

### Backpressure Dashboard

| Row | Panel | Metric(s) | Type |
|---|---|---|---|
| Queue Depths | Event Queue Depth | `pipeline_event_queue_depth` | timeseries |
| Queue Depths | Snapshot Queue Depth | `pipeline_snapshot_queue_depth` | timeseries |
| Drops | Drop Rate | `pipeline_events_dropped`, `pipeline_snapshots_dropped` | timeseries |
| Drops | Events Dropped (5m) | `pipeline_events_dropped` increase | stat |
| Drops | Snapshots Dropped (5m) | `pipeline_snapshots_dropped` increase | stat |
| Process Health | Tick Drift | `pipeline_tick_drift_ms` | timeseries |
| Process Health | Process RSS | `pipeline_rss_mb` | timeseries |

## Troubleshooting

### No data on dashboards

1. Verify Prometheus is scraping the target:
   ```
   curl http://<prometheus>:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="cryptoscreener")'
   ```
2. Check the metric exists:
   ```
   curl http://<cryptoscreener>:9090/metrics | grep cryptoscreener_ws
   ```
3. Verify template variable values match your labels (namespace, pod, job)

### "No data" on specific panels

- Counter metrics require traffic to increment. If the service just started, counters may be 0.
- `rate()` and `increase()` need at least 2 scrape intervals of data.
- Check that the `$datasource` variable points to the correct Prometheus instance.

### Dashboard shows stale data

- Check Grafana time range (top-right). Default is "Last 1 hour".
- Verify Prometheus scrape interval (15s recommended, see DEC-033/DEC-035).
- Check for clock skew between Prometheus and the target pod.

### Panels show "N/A" or "No data" for gauge/stat

- Gauge metrics (`gov_current_queue_depth`, `pipeline_event_queue_depth`, etc.) require the pipeline to be running.
- If using `--dry-run`, these metrics won't be populated.
