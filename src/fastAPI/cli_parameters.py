import os
import argparse
from pathlib import Path
from typing import NamedTuple


class CacheConfig(NamedTuple):
    """Configuration container for cache paths."""
    model_cache: Path
    data_cache: Path


def parse_cli_parameters(args: list = None) -> CacheConfig:
    """
    Parse command-line arguments and environment variables for cache configuration.
    
    Environment variables (take precedence over defaults):
    - MODEL_CACHE: Path to the cache directory where the trained model is stored
    - DATA_CACHE: Path to the cache directory where preprocessed data is stored
    
    CLI arguments (override environment variables):
    - --model-cache: Path to the model cache directory
    - --data-cache: Path to the data cache directory
    
    Args:
        args: List of arguments to parse. If None, uses sys.argv. If empty list [], only environment variables are used.
              This is useful when called from within FastAPI startup where sys.argv contains uvicorn arguments.
    
    Returns:
        CacheConfig: Named tuple containing model_cache and data_cache paths
        
    Example:
        >>> config = parse_cli_parameters()  # From CLI
        >>> print(config.model_cache)
        >>> config = parse_cli_parameters([])  # From app startup (env vars only)
        >>> print(config.data_cache)
    """
    # Get environment variables with defaults
    model_cache_env = os.getenv("MODEL_CACHE")
    data_cache_env = os.getenv("DATA_CACHE")
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Configure cache directories for the ML API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  MODEL_CACHE    Path to the cache where the trained model is stored
  DATA_CACHE     Path to the cache where preprocessed data is stored

Examples:
  python -m src.fastAPI.app
  python -m src.fastAPI.app --model-cache /path/to/models --data-cache /path/to/data
  MODEL_CACHE=/custom/models DATA_CACHE=/custom/data python -m src.fastAPI.app
        """
    )
    
    parser.add_argument(
        "--model-cache",
        type=str,
        default=model_cache_env,
        help=f"Path to the cache directory where the trained model is stored (env: MODEL_CACHE)"
    )
    
    parser.add_argument(
        "--data-cache",
        type=str,
        default=data_cache_env,
        help=f"Path to the cache directory where preprocessed data is stored (env: DATA_CACHE)"
    )
    
    parsed_args = parser.parse_args(args)
    
    # Convert to Path objects and validate
    model_cache = Path(parsed_args.model_cache)
    data_cache = Path(parsed_args.data_cache)

    # Log configuration
    print(f"Cache Configuration:")
    print(f"  Model Cache: {model_cache.resolve()}")
    print(f"  Data Cache:  {data_cache.resolve()}")

    if not model_cache.exists():
        print(f"Warning: Model cache directory does not exist at {model_cache.resolve()}. Ensure the model is trained and cached correctly.")
    if not data_cache.exists():
        print(f"Warning: Data cache directory does not exist at {data_cache.resolve()}. Ensure the data is preprocessed and cached correctly.")
    
    return CacheConfig(model_cache=model_cache, data_cache=data_cache)

if __name__ == "__main__":
    config = parse_cli_parameters()
    print(f"\nCache configuration loaded successfully!")
    print(f"Model path: {config.model_cache}")
    print(f"Data path: {config.data_cache}")
