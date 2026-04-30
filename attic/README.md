# Shard — attic

Parked profiles. The Shard core container (header + index + entries +
checksums) does not depend on these. They are not built or tested by default.

The role byte values they declared remain reserved — readers that don't
recognize a role still decode the file as a generic named-entry archive.

## profiles/

- `mosh/` — ModelShard (role 0x01). Model weights keyed by layer name.
- `columnshard/` — ColumnShard (role 0x08). Columnar tabular data with row
  group statistics. Travels with `column_encoding.go`.
- `sampleshard/` — SampleShard (role 0x02). Pre-tokenized training samples
  with PyTorch DataLoader integration. Python + TypeScript packages.

To revive a profile, restore the files and re-decouple from `attic/`. Git
history of the cut commit shows the original integration points in
`shard/go/shard/`.
