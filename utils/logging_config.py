import importlib.util
import os
import yaml
import logging.config


def setup_logging(
    config_path="configs/logging_config.yaml", log_level="WARNING"
):
    """
    Loads logging config from YAML and applies the verbose flag.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Override Root level based on log_level argument
        config["loggers"]["root"]["level"] = log_level

        # colorlog is optional: without it dictConfig raises on the 'colored'
        # formatter and takes the whole script down over log styling alone.
        if importlib.util.find_spec("colorlog") is None:
            for handler in config.get("handlers", {}).values():
                if handler.get("formatter") == "colored":
                    handler["formatter"] = "standard"
            config.get("formatters", {}).pop("colored", None)

        logging.config.dictConfig(config)

    else:
        # Fallback if file is missing
        logging.basicConfig(level=logging.INFO)
        print(f"Warning: Logging config file not found at {config_path}")
