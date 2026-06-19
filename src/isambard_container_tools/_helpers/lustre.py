"""Workarounds for running Python on Lustre filesystems.

Slurm redirects stdout/stderr to files on Lustre, where ``flush()``
can intermittently fail with ``ESTALE`` (errno 116).  Python's logging
module flushes after every message, producing thousands of noisy
``--- Logging error ---`` tracebacks.  :func:`patch_lustre_streams`
suppresses the harmless error at two levels:

1. Wraps ``sys.stdout`` / ``sys.stderr`` so that direct ``print()``
   calls and new logging handlers are protected.
2. Monkey-patches ``logging.StreamHandler.flush`` so that handlers
   that already captured a reference to the *original* stream before
   the patch was applied are also protected.
"""

import errno
import logging

_patched = False


def _make_safe_flush():
    """Return a patched ``StreamHandler.flush`` that suppresses ESTALE."""
    _original_flush = logging.StreamHandler.flush

    def _safe_flush(self):
        try:
            _original_flush(self)
        except OSError as exc:
            if exc.errno != errno.ESTALE:
                raise

    return _safe_flush


def patch_lustre_streams() -> None:
    """Suppress ``ESTALE`` errors when flushing stdout/stderr on Lustre.

    Safe to call multiple times — subsequent calls are no-ops.
    Call this early in your script::

        from isambard_container_tools._helpers.lustre import patch_lustre_streams
        patch_lustre_streams()
    """
    global _patched  # noqa: PLW0603
    if _patched:
        return
    _patched = True

    # Patch logging.StreamHandler.flush so that *all* handlers —
    # including those created before this call that hold a direct
    # reference to the original sys.stderr — are protected.
    logging.StreamHandler.flush = _make_safe_flush()
