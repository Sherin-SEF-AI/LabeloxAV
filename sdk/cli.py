"""Moved to sdk/labelox/cli.py, the installable client package (labelox-client).

This path stays so existing imports keep working: the module object here IS sdk.labelox.cli, not a copy,
so everything it defines, private names included, resolves exactly as before.
"""

import sys

from sdk.labelox import cli as _moved

if __name__ == "__main__":
    # `python -m sdk.cli ...` from a checkout: run the moved CLI rather than replacing __main__.
    _moved.main()
else:
    sys.modules[__name__] = _moved
