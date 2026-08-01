"""
Shared logger setup.

Plain stdlib logging, formatted so GitHub Actions run logs are readable:
timestamp, level, module name, message.
"""

import logging
import sys

_CONFIGURED = False

def _configure_root():
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring the root handler once."""
    _configure_root()
    return logging.getLogger(name)
