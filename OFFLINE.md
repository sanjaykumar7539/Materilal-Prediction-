# Offline guarantee strategy

The code contains **no required external model or API call**.

## In-process guard

```python
from ownllm.offline import enforce_offline
enforce_offline()
```

Or set `OWNLLM_OFFLINE=1` before starting the API service.

This:

1. Sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `NO_PROXY`.
2. Patches `socket.socket.connect` to allow only `127.0.0.1` / `localhost` / `::1`.

## Hard offline (recommended for air-gap)

1. Pre-build a Docker image with all Python wheels installed (no runtime `pip install` from the internet).
2. Disable container/host egress via firewall / network policy.
3. Mount models, tokenizers, and data from local volumes only.
4. Localhost services (Redis) remain allowed.

Validate by running training, retrieval, inference, and UI with the network physically disabled.
