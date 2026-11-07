"""Flask web front end for pyGrafLab.

The server module imports Flask, which is an optional dependency, so it is not
imported here.  Launch it either way::

    python -m pygraflab.web
    pygraflab-web --port 5002
"""

__all__ = ["main"]


def main(argv=None):
    """Entry point proxy, so ``pygraflab.web.main`` always resolves."""
    from .server import main as _main

    return _main(argv)
