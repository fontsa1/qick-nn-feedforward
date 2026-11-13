# v1.44 loop latency: 10 reloads and 4 cold starts

The protocol logs and JSONs here flag `CHECKS FAILED` / `"passed": false` because `path_minus_adv` was checked
against v1.43's centre. All other closure checks passed on all 14 loads.

- **What the check does.** It subtracts the configured NN_live advance (55 tProc ticks, 111.9 ns) from the boxcar path
  that `box_probe.py` measures on the same load. It catches a wrong advance setting (0.1078 us reads about
  7 ns).
- **Centre.** 3.2 ns comes from v1.43's boxcar path (115.1 ns). On v1.44 the path is 113.3 ns,
  one 552.96 MHz sample shorter, as on v1.42 (113.24 ns). The path moves between place-and-route runs. All 14 v1.44
  loads read 1.33 to 1.54 ns.
- **What it affects.** Not L_50 or its segments: those are measured from the window's last sample to the returning
  pulse and do not use the path term. The path term enters only L_buf, the buffer-based cross-check, which tracked it on
  every load (`lbuf_minus_l50` passed).

**Centre for v1.44:** 1.4 ns, same +/- 1.0 ns window. The JSONs are as recorded;
`campaign_v144/final/analysis/loop_groups_data_v144.py` applies the v1.44 centre and requires every other check to pass.
`campaign_v144/run_loads.sh` passes `latency_protocol.py --expect-path-adv 1.4`.
