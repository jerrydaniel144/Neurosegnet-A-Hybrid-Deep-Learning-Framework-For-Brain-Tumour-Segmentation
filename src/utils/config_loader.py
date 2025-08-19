import yaml
from pathlib import Path
from typing import Union


def load_config(config_path: Union[str, Path]) -> dict:
    """
    Load a YAML configuration file into a Python dictionary.

    Args:
        config_path (str or Path): Path to the YAML configuration file.

    Returns:
        dict: Configuration as a dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: dict, required_keys: list):
    """
    Check if required keys exist in the configuration dictionary.

    Args:
        config (dict): Loaded configuration.
        required_keys (list): List of expected keys.

    Raises:
        KeyError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing config keys: {', '.join(missing_keys)}")
    print("Config validated successfully.")


def print_config_summary(config: dict, title="Configuration Summary"):
    """
    Nicely format and print a summary of the configuration dictionary.

    Args:
        config (dict): Loaded configuration.
        title (str): Header title for the summary.
    """
    print(f"\n{title}")
    print("=" * len(title))
    for key, value in config.items():
        print(f"{key:>20} : {value}")
    print("=" * len(title) + "\n")