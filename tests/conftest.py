"""Keep the standard logging console quiet during tests; .entries capture is level-independent."""

import os

os.environ.setdefault("KINOFORGE_LOG_LEVEL", "CRITICAL")
