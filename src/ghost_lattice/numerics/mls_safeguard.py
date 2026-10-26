"""Explicit dimension-appropriate extension of existing 3D nearest-support safeguard.

Original source files stay pinned. Ordinary well-conditioned MLS calls return the
original function's exact result. Installation is scoped to a context manager.
"""

from contextlib import contextmanager
from pathlib import Path
import hashlib
import numpy as np
from . import plane_lpm as gbp2d
from . import rotated_plane as compact_rotated


class MLSGuard:

    def __init__(self):
        self.original = gbp2d.mls_weights
        self.calls = 0
        self.events = []
        self.max_regular_moment_error = 0.0

    def __call__(self, Y, h=1.0):
        self.calls += 1
        Y = np.asarray(Y)
        d = np.linalg.norm(Y, axis=1)
        if len(Y) == 0:
            raise ValueError("MLS guard requires a nonempty support set")
        rho = np.exp(-((d / h) ** 2))
        P = np.column_stack([np.ones(len(Y)), Y])
        A = P.T @ (rho[:, None] * P)
        condition = float(np.linalg.cond(A))
        reason = None
        if (
            len(Y) < 3
            or not np.all(np.isfinite(A))
            or (not np.isfinite(condition))
            or (condition > 100000000.0)
        ):
            reason = "insufficient_or_ill_conditioned"
        if reason is None:
            try:
                w = self.original(Y, h)
                if not np.all(np.isfinite(w)):
                    reason = "nonfinite_weights"
            except np.linalg.LinAlgError:
                reason = "singular_solve"
        if reason is not None:
            w = np.zeros(len(Y))
            nearest = int(np.argmin(d))
            w[nearest] = 1.0
            self.events.append(
                {
                    "call": self.calls,
                    "reason": reason,
                    "support_count": len(Y),
                    "condition": condition if np.isfinite(condition) else None,
                    "nearest_offset": Y[nearest].tolist(),
                    "nearest_distance": float(d[nearest]),
                    "offsets": Y.tolist(),
                }
            )
        else:
            self.max_regular_moment_error = max(
                self.max_regular_moment_error,
                float(np.max(np.abs(P.T @ w - np.array([1.0, 0.0, 0.0])))),
            )
        return w

    def report(self):
        return {
            "calls": self.calls,
            "fallback_count": len(self.events),
            "fallback_fraction": len(self.events) / max(self.calls, 1),
            "minimum_supports": 3,
            "condition_threshold": 100000000.0,
            "regular_MLS_max_moment_error": self.max_regular_moment_error,
            "fallback_trigger": "Insufficient support, moment condition number above1e8, singular solve or nonfinite weights; ordinary well-conditioned original 2D MLS calls kept exactly",
            "fallback_rule": "Nearest existing support with unit weight, extending the existing 3D nearest-support treatment",
            "fallback_precision": "Partition of unity retained; first moment not imposed on fallback stencils",
            "events": self.events,
        }

    @contextmanager
    def installed(self):
        a, b = (gbp2d.mls_weights, compact_rotated.mls_weights)
        gbp2d.mls_weights = compact_rotated.mls_weights = self
        try:
            yield self
        finally:
            gbp2d.mls_weights, compact_rotated.mls_weights = (a, b)


def module_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
