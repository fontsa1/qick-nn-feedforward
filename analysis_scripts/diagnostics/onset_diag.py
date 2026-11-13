"""Decode NN_axi's onset diagnostics register.

`drain_count_last` packs two fields:

    idle_flush = drain_count_last >> 16     words IDLE_FLUSH_L discarded since the previous shot
    drain      = drain_count_last & 0xFFFF  words read before the window

Builds without the idle flush leave the upper half at 0, so the decoder works for both. Both fields stay well under
16 bits (the idle flush count saturates at 0xFFFF in firmware). The raw register value (~5e7) is not a fault.
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
