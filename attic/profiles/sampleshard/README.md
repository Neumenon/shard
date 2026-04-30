# SampleShard — Parked Profile

Role `0x02`. Pre-tokenized training samples with PyTorch DataLoader integration
(`.smpl` files).

**Status: parked.** This profile is not built or tested by default. The Python
package (`py/`) and TypeScript package (`js/`) are preserved here for
reference. The role constant `ShardRoleSample = 0x02` is still defined in
`shard.go`. The `ProfileMeta json.RawMessage` field in `ShardMetadata` carries
any profile-specific JSON; the earlier typed `*SampleProfile` struct was
removed from Shard core when the profile was parked.

To see the original integration see git history at the cut commit
(`git log --diff-filter=D -- go/shard/sampleshard*`).

See `py/README.md` for the Python API reference (archived, not maintained).
