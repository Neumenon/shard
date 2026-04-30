# ModelShard (MoSH) — Parked Profile

Role `0x01`. Stores model weight tensors keyed by layer name (`.mosh` files).

**Status: parked.** This profile is not built or tested by default. The code
is preserved here for reference; `LlamaSchema` and the MoSH-specific wrapper
were removed from the active Shard core at the same time. The role constant
`ShardRoleMoSH = 0x01` is still defined in `shard.go` because WShard chunking
references it, but no MoSH profile logic runs at build time.

To see the original integration see git history at the cut commit
(`git log --diff-filter=D -- go/shard/mosh.go`).
