"""Allow ``python -m pygraflab.web`` to launch the web server."""
import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
