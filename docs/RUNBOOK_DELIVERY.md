# Runbook: RankEvent Delivery (DEC-039)

## Overview

The delivery module routes RankEvents to operators via configurable sinks (Telegram, Slack, webhook). It includes anti-spam controls, deterministic formatting, and dry-run mode for testing.

## Sinks

| Sink | Env Vars Required | Use Case |
|---|---|---|
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Personal/team notifications |
| Slack | `SLACK_WEBHOOK_URL` | Channel notifications |
| Webhook | `DELIVERY_WEBHOOK_URL` | Custom integrations, automation |

## Quick Start

### Enable Telegram Delivery

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather) and get the token.
2. Get your chat ID (message the bot, then check `https://api.telegram.org/bot<TOKEN>/getUpdates`).
3. Set environment variables:
   ```bash
   export TELEGRAM_BOT_TOKEN="your-bot-token"
   export TELEGRAM_CHAT_ID="your-chat-id"
   ```
4. Run with delivery enabled:
   ```bash
   python scripts/run_live.py --delivery-telegram
   ```

### Enable Slack Delivery

1. Create an Incoming Webhook in Slack: https://api.slack.com/messaging/webhooks
2. Set environment variable:
   ```bash
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
   ```
3. Run with delivery enabled:
   ```bash
   python scripts/run_live.py --delivery-slack
   ```

### Enable Webhook Delivery

1. Set your webhook URL:
   ```bash
   export DELIVERY_WEBHOOK_URL="https://your-endpoint.com/alerts"
   ```
2. Run with delivery enabled:
   ```bash
   python scripts/run_live.py --delivery-webhook
   ```

### Multiple Sinks

Enable multiple sinks by combining flags:
```bash
python scripts/run_live.py --delivery-telegram --delivery-slack
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--delivery-telegram` | off | Enable Telegram sink |
| `--delivery-slack` | off | Enable Slack sink |
| `--delivery-webhook` | off | Enable generic webhook sink |
| `--delivery-dry-run` | off | Log messages without sending |
| `--delivery-cooldown-s` | 120 | Per-symbol cooldown in seconds |

## Anti-Spam Controls

The delivery module includes three anti-spam mechanisms:

### 1. Per-Symbol Cooldown

Same symbol + event type won't repeat within the cooldown window (default: 120 seconds).

Configurable via `--delivery-cooldown-s`:
```bash
python scripts/run_live.py --delivery-telegram --delivery-cooldown-s 300  # 5 min cooldown
```

### 2. Global Rate Limit

Maximum 30 deliveries per minute across all symbols. Prevents notification storms.

### 3. Status Transition Only

By default, only delivers when status changes (e.g., RISKY -> TRADEABLE). Same status repeats are suppressed.

## Message Format

Messages are formatted deterministically using a template:

```
[ALERT_TRADABLE] BTCUSDT

Status: TRADEABLE
Rank: #1 | Score: 0.92
Time: 2024-01-28 12:34:56 UTC
```

For events with LLM text, it's appended:
```
---
Analysis:
The price action shows bullish momentum...
```

### Grafana Links

To include Grafana dashboard links:
```python
# In config
DeliveryConfig(
    include_grafana_link=True,
    grafana_base_url="https://grafana.example.com"
)
```

## Dry Run Mode

Test your configuration without sending actual messages:
```bash
python scripts/run_live.py --delivery-telegram --delivery-dry-run
```

Dry run logs the formatted message but doesn't call sink APIs.

## Kubernetes Deployment

### Add Delivery Secrets

If using External Secrets Operator, add to your secret backend:
- `cryptoscreener/telegram-bot-token`
- `cryptoscreener/telegram-chat-id`
- `cryptoscreener/slack-webhook-url`
- `cryptoscreener/delivery-webhook-url`

Then update `k8s/externalsecret.yaml`:
```yaml
data:
  - secretKey: TELEGRAM_BOT_TOKEN
    remoteRef:
      key: cryptoscreener/telegram-bot-token
  - secretKey: TELEGRAM_CHAT_ID
    remoteRef:
      key: cryptoscreener/telegram-chat-id
```

Or create directly:
```bash
kubectl create secret generic cryptoscreener-delivery-secrets \
  --from-literal=TELEGRAM_BOT_TOKEN=<token> \
  --from-literal=TELEGRAM_CHAT_ID=<chat-id>
```

### Update Deployment

Add delivery flags to the deployment:
```yaml
containers:
  - name: cryptoscreener
    args:
      - "--delivery-telegram"
    envFrom:
      - secretRef:
          name: cryptoscreener-delivery-secrets
```

## Metrics

Delivery metrics are available via the `/metrics` endpoint:

| Metric | Description |
|---|---|
| `delivery_total_received` | Total events received for delivery |
| `delivery_total_delivered` | Successfully delivered batches |
| `delivery_total_failed` | Failed delivery attempts |
| `delivery_total_suppressed` | Events blocked by anti-spam |
| `dedupe_suppressed_cooldown` | Blocked by per-symbol cooldown |
| `dedupe_suppressed_rate_limit` | Blocked by global rate limit |
| `dedupe_suppressed_duplicate` | Blocked by status transition filter |

## Troubleshooting

### No messages received

1. **Check enabled**: Verify sink is enabled in logs:
   ```
   INFO Telegram sink enabled
   ```

2. **Check credentials**: Verify env vars are set:
   ```bash
   echo $TELEGRAM_BOT_TOKEN | head -c 10  # Should show partial token
   ```

3. **Check anti-spam**: Events may be suppressed. Check dedupe metrics:
   ```bash
   curl localhost:9090/metrics | grep dedupe
   ```

4. **Use dry-run**: Test formatting without sending:
   ```bash
   python scripts/run_live.py --delivery-telegram --delivery-dry-run
   ```

### Rate limited (429)

Telegram and Slack have rate limits. The sinks handle 429 responses with automatic retry (up to 2 retries with exponential backoff).

If rate limiting persists:
- Increase `--delivery-cooldown-s` to reduce frequency
- Reduce number of monitored symbols

### Connection errors

Check network access to:
- Telegram: `api.telegram.org:443`
- Slack: `hooks.slack.com:443`

For K8s, ensure egress is allowed to these endpoints.

### Message not formatted correctly

The formatter uses a deterministic template. If LLM text appears malformed:
- Check `include_llm_text` setting
- Verify the RankEvent payload contains valid `llm_text`

## Webhook Payload Format

The webhook sink sends JSON with all format variants:

```json
{
  "text": "[ALERT_TRADABLE] BTCUSDT\n\nStatus: TRADEABLE\n...",
  "html": "<b>[ALERT_TRADABLE]</b> BTCUSDT\n\n...",
  "markdown": "*[ALERT_TRADABLE]* BTCUSDT\n\n..."
}
```

### Custom Headers

Configure webhook headers in code:
```python
WebhookSinkConfig(
    enabled=True,
    url="https://...",
    headers={"Authorization": "Bearer token", "X-Custom": "value"}
)
```

## Secret Hygiene

Delivery secrets are included in `DELIVERY_REDACTED_ENV_VARS`:
- `TELEGRAM_BOT_TOKEN`
- `SLACK_WEBHOOK_URL`
- `DELIVERY_WEBHOOK_URL`

These are never logged. See [RUNBOOK_SECRETS.md](RUNBOOK_SECRETS.md) for general secret management.
