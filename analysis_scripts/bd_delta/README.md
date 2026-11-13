# Block-design delta: QICK's design to ours

`vivado_scripts/v1.43/from_qick/nnqick_bd_delta.tcl` turns QICK's `qick_tprocv2_4x2_standard` block design (commit
`779dffa`) into our v1.43 block design. It is generated here; don't edit it by hand.

## Files

| File | What it is |
|---|---|
| `dump_bd.tcl` | `dump_bd <file>` writes the open block design as tab-separated lines: cells with all CONFIG values, ports, nets and interface nets (as sets of endpoints), address assignments |
| `dump_both.tcl` | Builds QICK's project in a scratch folder, dumps it, then opens our v1.43 project read-only and dumps that |
| `qick_779dffa_d_1.tsv`, `v143_OG_build.tsv` | The two dumps the current delta was generated from |
| `gen_delta.py` | Writes the delta. Cells and their settings come from the two `write_bd_tcl` scripts (QICK's `bd_2023-1.tcl` and our `write_bd_tcl` export of v1.43, `OG_build_v1.43_bd.tcl`); connectivity comes from the dumps; the address map comes from our export |
| `diff_bd.py` | Compares two dumps |
| `check_from_qick.tsv`, `check_from_qick_vs_v143.txt` | The check: QICK's project plus the delta, made with `nnqick_from_qick.tcl`, dumped and compared with v1.43 |

## Regenerating

```
vivado -mode batch -source dump_both.tcl          # edit the two paths at its top first
python gen_delta.py <QICK>/firmware/projects/qick_tprocv2_4x2_standard/bd_2023-1.tcl \
       <our write_bd_tcl export>/OG_build_v1.43_bd.tcl qick_779dffa_d_1.tsv v143_OG_build.tsv \
       ../../vivado_scripts/v1.43/from_qick/nnqick_bd_delta.tcl
```

## Checking

Run `nnqick_from_qick.tcl` on a fresh QICK project, dump the result with `dump_bd.tcl`, and run
`python diff_bd.py <that dump> v143_OG_build.tsv`. The check (`check_from_qick_vs_v143.txt`) leaves 88 lines:
- 85 are instance names (`d_1_...` against `OG_build_...`).
- One is an unconnected one-pin net stub in our design (`qick_processor_0/trig_1_o`).
- Two are address exclusions on converter registers inside QICK's AXI interconnect (`s00_couplers/auto_ds`, `auto_pc`).

None of them changes the hardware. Every cell, every CONFIG value, all connectivity and every address assignment
match.

## Notes

- Removing nets in the delta disconnects every endpoint of a QICK net that changes and reconnects the new sets. Vivado
  prints "NET ... has no source" warnings in between; they don't describe the final design, which the check covers.
- Vivado 2023.1 can fail inside its own scripts ("couldn't read file" for files that exist) when the block design is
  opened late in a session, so `nnqick_from_qick.tcl` opens it right after saving the project.
