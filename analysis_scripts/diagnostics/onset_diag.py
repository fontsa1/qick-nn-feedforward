"""Decode NN_axi's onset diagnostics, across the v1.38 format change.

v1.38 (2026-09-21) packs TWO fields into the single `drain_count_last`
AXI-lite register rather than adding a new port (a new port means a BD and
wrapper change plus a full re-integration, for a diagnostic):

    idle_flush = drain_count_last >> 16     words IDLE_FLUSH_L discarded
                                            since the previous shot
    drain      = drain_count_last & 0xFFFF  unchanged meaning

Pre-v1.38 bitstreams leave the upper half at 0, so this decoder is correct for
both and callers do not need to know which build is loaded. Both fields are
bounded well under 16 bits by construction (drain <= MAX_DRAIN_WORDS = 1152;
the idle flush count is saturated at 0xFFFF in firmware).

Why it matters: `drain` pinned at 1 on every shot was the signature of the
one-shot lag -- the input queue held a full stale shot and DRAIN_L, being
level-triggered, latched its first sample every time. After v1.38's
IDLE_FLUSH_L holds the queue empty between shots, `drain` should STOP being
pinned at 1 and `idle_flush` should settle near 0 in steady state. Reading the
raw register without masking would show ~5e7 and look like a fault.
"""


def decode_drain(raw):
    """(idle_flush, drain) from a raw drain_count_last value."""
    raw = int(raw) & 0xFFFFFFFF
    return raw >> 16, raw & 0xFFFF


def read_onset_diag(soc):
    """soc.nn_onset_diagnostics() with drain_count_last decoded.

    Returns the original keys plus `drain` and `idle_flush`, so existing code
    that reads `drain_count_last` still sees the raw register and new code can
    use the decoded fields.
    """
    d = dict(soc.nn_onset_diagnostics())
    idle_flush, drain = decode_drain(d.get("drain_count_last", 0))
    d["idle_flush"] = idle_flush
    d["drain"] = drain
    return d
