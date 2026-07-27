"""Static-scene kernels: domain-neutral image-sequence math a fixed camera needs (a background prior, and
motion against it). The AV pack does not use these; a static-camera pack builds its scene model on top of
them. Kept in core so the kernels stay shared and pack-agnostic (see docs/PACK_INTERFACE.md)."""
