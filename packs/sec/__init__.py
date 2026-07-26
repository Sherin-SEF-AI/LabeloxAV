"""LabeloxSec: the India CCTV / security-footage domain pack.

The registry reads `PACK` from here. The static-camera scene model, ingestion adapter, and CCTV quality
profile landed in SEC-M2; the ontology and the assembled pack in SEC-M3. The engine never imports this
package (see .importlinter); it reaches it only through packs.registry.
"""

from packs.sec.pack import PACK

__all__ = ["PACK"]
