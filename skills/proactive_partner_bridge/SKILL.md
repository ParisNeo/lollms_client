---
name: Proactive Partner Bridge Server
description: Architecture and usage guide for the Couples Counselor Bridge Server — multi-platform messaging that routes Discord/Slack/Telegram/WhatsApp channels to the couple's vault tenants.
author: LoLLMS
version: 1.0.0
category: lollms_client/connections
---

# Couples Counselor Bridge Server

## Overview

The Bridge Server (`examples_perso/proactive_partner/bridge_server.py`) connects
the couples counselor to external messaging platforms (Discord, Slack, Telegram,
WhatsApp) via webhook polling. Each partner gets their own platform channel, and
the counselor auto-replies to the correct partner's channel.

## Architecture

```
Slack webhook   ──poll──┐
Discord webhook ──poll──┤──→ CouplesBridgeServer ──→ CouplesCounselor.chat()
Telegram API   ──poll──┘        │                          │
                                │         ├── vault tenant routing
                                │         └── session state tracking
                                └── auto-reply ←───────────┘
                                     (via client.send_connection_message)
```

## Channel → Tenant Mapping

The bridge maps each connection profile alias to a vault tenant using
naming conventions:

| Alias contains | Maps to | Example |
|---|---|---|
| `alice`, `partner_a`, `_a_` | `private_a` | Alice's Slack DM |
| `bob`, `partner_b`, `_b_`  | `private_b` | Bob's Telegram |
| `shared`, `general`, `counsel`, `couple`, `both` | `shared` | Couple group chat |
| anything else | `shared` (safe default) | Unknown channel |

## Setup

1. Configure connection profiles in `~/.lollms_client/config.yaml`:

```yaml
connection:
  bindings:
    my_slack:
      binding_name: generic_webhook
      binding_config:
        service_key: "https://hooks.slack.com/services/T.../B.../XXX"
    my_telegram:
      binding_name: generic_webhook
      binding_config:
        service_key: "https://api.telegram.org/botTOKEN/sendMessage"
  profiles:
    alice_slack:
      binding_profile_name: my_slack
      model_name: "#alice-dm"
    bob_telegram:
      binding_profile_name: my_telegram
      model_name: "@bob_chat_id"
    shared_counsel:
      binding_profile_name: my_slack
      model_name: "#couples-general"
```

2. Run the bridge standalone:
```bash
python examples_perso/proactive_partner/bridge_server.py
```

3. Or start it from the proactive partner session:
```
/bridge          # Foreground or background
/bridge-status   # Check which threads are alive
/bridge-stop     # Gracefully shut down all channels
```

## How It Works

### Incoming Message Flow
1. Bridge polls each channel concurrently (`_poll_channel`)
2. New message → dedupe by message ID → `_process_incoming_message()`
3. Channel alias maps to tenant (`alice_slack` → `private_a`)
4. Call `counselor.start_individual_session("partner_a")` → vault context switches
5. `counselor.chat(message)` → safety gate + assessment + mechanism selection + vault hydration
6. Response text → `_send_reply()` → `client.switch_connection(alias)` → `conn.send_message()`

### Vault Isolation Guarantee
- `private_a` vault items are NEVER visible in the shared context.
- `private_b` vault items are NEVER visible in the shared context.
- A private session only hydrates `shared` + the private tenant's own items.
- The counselor's safety gate ALWAYS runs before any private content is stored.

## Extending to More Channels

To add WhatsApp, Signal, or any new platform:
1. Create a new binding in `src/lollms_client/connection_bindings/<name>/`
2. Implement `LollmsConnectionBinding` with `connect()`, `send_message()`, `list_channels()`
3. Set `BindingName = "<ClassName>"` in `__init__.py`
4. Add a `description.yaml` with `global_input_parameters`
5. The bridge auto-discovers it via `get_available_bindings()`

The wizard (`/config` or `python -m lollms_client.lollms_config_cli_env`) will list it under CONNECTION bindings with auto-prompting from `description.yaml`.