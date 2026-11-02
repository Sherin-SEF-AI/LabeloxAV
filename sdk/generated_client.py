"""Moved to sdk/labelox/generated.py, the installable client package (labelox-client).

This path stays so existing imports keep working: the module object here IS sdk.labelox.generated, not a copy,
so everything it defines, private names included, resolves exactly as before.
"""

import sys

from sdk.labelox import generated as _moved

sys.modules[__name__] = _moved
