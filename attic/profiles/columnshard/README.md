# ColumnShard — Parked Profile

Role `0x08`. Columnar tabular data with row-group statistics (`.cshard` files).

**Status: parked.** This profile is not built or tested by default. The code
(`columnshard_reader.go`, `columnshard_writer.go`, `columnshard_schema.go`,
`column_encoding.go`) is preserved here for reference. The role constant
`ShardRoleColumn = 0x08` is still defined in `shard.go`.

To see the original integration see git history at the cut commit
(`git log --diff-filter=D -- go/shard/columnshard_reader.go`).
