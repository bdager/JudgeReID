import os

from omegaconf import OmegaConf


def resolve_vlm_params(cfg) -> dict:
    """Resolve the VLM's endpoint and generation parameters.

    Precedence (highest first):
      1. ``cfg.vlm[name]``            — experiment-specific inline overrides
      2. ``configs/vlm/{name}.yaml``  — provider defaults
      3. ``cfg.vlm``                  — global fallback for keys neither of the above sets
                                        (temperature, max_tokens, top_p, max_retries, ...)

    Provider-specific keys (``top_k``, ``presence_penalty``, ``enable_thinking``, ...)
    are forwarded as they are, so new sampling options need no code change here.
    The returned dict includes ``model`` and ``base_url``.
    """
    name = cfg.vlm.name

    configs_dir = cfg.vlm.get("configs_dir", "configs/vlm")
    provider_cfg_path = os.path.join(configs_dir, f"{name}.yaml")
    if os.path.exists(provider_cfg_path):
        provider_cfg = OmegaConf.load(provider_cfg_path)
    else:
        provider_cfg = OmegaConf.create({})

    inline_overrides = cfg.vlm.get(name, OmegaConf.create({}))
    merged_model_cfg = OmegaConf.merge(provider_cfg, inline_overrides)

    def _get(key, default=None):
        """Per-model value → global vlm default → hard-coded default."""
        val = merged_model_cfg.get(key, None)
        if val is None:
            val = cfg.vlm.get(key, default)
        return val

    params = {
        "model": merged_model_cfg.get("model"),
        "base_url": merged_model_cfg.get("base_url", None),
        "temperature": _get("temperature", 0),
        "max_tokens": _get("max_tokens", 500),
        "top_p": _get("top_p", 0.0001),
    }

    _handled = {"model", "base_url", "temperature", "max_tokens", "top_p"}
    for key, value in merged_model_cfg.items():
        if key not in _handled and key not in params:
            params[key] = value

    # Global vlm.* scalars not covered above (e.g. max_retries, concurrency);
    # structural keys and per-model blocks are skipped
    _structural = {"name", "configs_dir"}
    for key, value in cfg.vlm.items():
        if key in _structural or key in _handled or key in params:
            continue
        if OmegaConf.is_config(value):
            continue
        params[key] = value

    return params
