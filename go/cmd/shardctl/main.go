package main

import (
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"hash/crc32"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/phenomenon0/Agent-GO/cowrie/ucodec"
	"github.com/phenomenon0/Agent-GO/cowrie/ucodec/hfimport"
)

var jsonOutput bool
var listPrefix string
var verboseOutput bool

func main() {
	flag.BoolVar(&jsonOutput, "json", false, "output in JSON format")
	flag.BoolVar(&verboseOutput, "v", false, "verbose output")
	flag.StringVar(&listPrefix, "prefix", "", "filter entries by hierarchical prefix (for list command)")
	flag.StringVar(&outputDir, "o", ".", "output directory for downloads")
	flag.IntVar(&quantBits, "quant", 0, "quantization bits for convert (0=none, 4=Q4_K, 5=Q5_K, 8=INT8)")
	flag.Parse()

	args := flag.Args()
	if len(args) < 1 {
		printUsage()
		os.Exit(1)
	}

	cmd := args[0]
	switch cmd {
	case "info":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shardctl info <file.shard>")
			os.Exit(1)
		}
		cmdInfo(args[1])
	case "list":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shardctl list [-prefix <prefix>] <file.shard>")
			os.Exit(1)
		}
		cmdList(args[1], listPrefix)
	case "cat":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: shardctl cat <file.shard> <entry_name>")
			os.Exit(1)
		}
		cmdCat(args[1], args[2])
	case "stats":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shardctl stats <file.shard>")
			os.Exit(1)
		}
		cmdStats(args[1])
	case "verify":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shardctl verify <file.shard>")
			os.Exit(1)
		}
		os.Exit(cmdVerify(args[1]))
	case "diff":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: shardctl diff <file1.shard> <file2.shard>")
			os.Exit(1)
		}
		os.Exit(cmdDiff(args[1], args[2]))
	case "audit":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shardctl audit <file.shard>")
			os.Exit(1)
		}
		os.Exit(cmdAudit(args[1]))
	case "download":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shardctl download [-o <output_dir>] <repo_id> [file_pattern]")
			fmt.Fprintln(os.Stderr, "  repo_id: HuggingFace repo (e.g., 'meta-llama/Llama-2-7b')")
			fmt.Fprintln(os.Stderr, "  file_pattern: optional glob pattern (e.g., '*.safetensors')")
			os.Exit(1)
		}
		pattern := ""
		if len(args) > 2 {
			pattern = args[2]
		}
		os.Exit(cmdDownload(args[1], pattern))
	case "convert":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shardctl convert [-o <output.mosh>] <model_dir>")
			fmt.Fprintln(os.Stderr, "  model_dir: directory with safetensors and config.json")
			os.Exit(1)
		}
		os.Exit(cmdConvert(args[1]))
	case "inspect":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: shardctl inspect <file.safetensors>")
			os.Exit(1)
		}
		os.Exit(cmdInspect(args[1]))
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

var outputDir string
var quantBits int

func printUsage() {
	fmt.Fprintln(os.Stderr, `shardctl - inspect and verify shard v2 files

Usage: shardctl [flags] <command> [args...]

Commands:
  info     <file.shard>                  Show header info
  list     <file.shard>                  List all entries with sizes
  cat      <file.shard> <entry>          Dump entry data
  stats    <file.shard>                  Show compression stats
  verify   <file.shard>                  Verify all checksums (exit 0 if OK)
  diff     <file1.shard> <file2.shard>   Compare two shards
  audit    <file.shard>                  Comprehensive integrity audit
  download <repo_id> [file_pattern]      Download from HuggingFace Hub
  convert  <model_dir>                   Convert HF safetensors to MoSH
  inspect  <file.safetensors>            Inspect safetensors file

Flags:
  -json            Output in JSON format
  -v               Verbose output
  -prefix <path>   Filter entries by hierarchical prefix (for list command)
  -o <dir>         Output directory/file for downloads/convert (default: current dir)
  -quant <bits>    Quantization bits for convert (0=none, 4=Q4_K, 5=Q5_K, 8=INT8)`)
}

func cmdInfo(path string) {
	reader, err := ucodec.OpenShardV2(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening shard: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	h := reader.Header()

	info, err := os.Stat(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error stating file: %v\n", err)
		os.Exit(1)
	}

	hasContentTypes := h.Flags&ucodec.ShardV2FlagHasContentTypes != 0

	if jsonOutput {
		out := map[string]any{
			"version":           h.Version,
			"role":              roleString(h.Role),
			"role_id":           h.Role,
			"entry_count":       h.EntryCount,
			"compression":       compressionString(h.CompressionDefault),
			"alignment":         h.Alignment,
			"file_size":         info.Size(),
			"flags":             h.Flags,
			"has_content_types": hasContentTypes,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
		return
	}

	fmt.Printf("Shard V2 Info: %s\n", path)
	fmt.Printf("  Version:       %d\n", h.Version)
	fmt.Printf("  Role:          %s (%d)\n", roleString(h.Role), h.Role)
	fmt.Printf("  Entry Count:   %d\n", h.EntryCount)
	fmt.Printf("  Compression:   %s\n", compressionString(h.CompressionDefault))
	fmt.Printf("  Alignment:     %d bytes\n", h.Alignment)
	fmt.Printf("  File Size:     %s (%d bytes)\n", humanSize(info.Size()), info.Size())
	fmt.Printf("  Flags:         0x%04x\n", h.Flags)
	fmt.Printf("  Content Types: %v\n", hasContentTypes)
}

func cmdList(path string, prefix string) {
	reader, err := ucodec.OpenShardV2(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening shard: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	// Get entries to display (filtered by prefix if specified)
	var indices []int
	if prefix != "" {
		names := reader.ListPrefix(prefix)
		for _, name := range names {
			idx := reader.Lookup(name)
			if idx >= 0 {
				indices = append(indices, idx)
			}
		}
	} else {
		count := reader.EntryCount()
		indices = make([]int, count)
		for i := 0; i < count; i++ {
			indices[i] = i
		}
	}

	hasContentTypes := reader.Header().Flags&ucodec.ShardV2FlagHasContentTypes != 0

	if jsonOutput {
		entries := make([]map[string]any, len(indices))
		for j, i := range indices {
			info := reader.GetEntryInfo(i)
			entry := map[string]any{
				"name":       reader.EntryName(i),
				"disk_size":  info.DiskSize,
				"orig_size":  info.OrigSize,
				"compressed": info.IsCompressed(),
			}
			if hasContentTypes {
				entry["content_type"] = ucodec.ContentTypeName(info.ContentType())
			}
			entries[j] = entry
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(entries)
		return
	}

	if hasContentTypes {
		fmt.Printf("%-40s %12s %12s %-10s %s\n", "NAME", "DISK SIZE", "ORIG SIZE", "TYPE", "COMPRESSED")
		fmt.Printf("%s\n", strings.Repeat("-", 95))
	} else {
		fmt.Printf("%-40s %12s %12s %s\n", "NAME", "DISK SIZE", "ORIG SIZE", "COMPRESSED")
		fmt.Printf("%s\n", strings.Repeat("-", 80))
	}
	for _, i := range indices {
		name := reader.EntryName(i)
		info := reader.GetEntryInfo(i)
		comp := ""
		if info.IsCompressed() {
			comp = compressionTypeString(info.CompressionType())
		}
		if hasContentTypes {
			fmt.Printf("%-40s %12s %12s %-10s %s\n",
				truncate(name, 40),
				humanSize(int64(info.DiskSize)),
				humanSize(int64(info.OrigSize)),
				ucodec.ContentTypeName(info.ContentType()),
				comp)
		} else {
			fmt.Printf("%-40s %12s %12s %s\n",
				truncate(name, 40),
				humanSize(int64(info.DiskSize)),
				humanSize(int64(info.OrigSize)),
				comp)
		}
	}
	if prefix != "" {
		fmt.Printf("\nTotal: %d entries (prefix: %s)\n", len(indices), prefix)
	} else {
		fmt.Printf("\nTotal: %d entries\n", len(indices))
	}
}

func cmdCat(path, entryName string) {
	reader, err := ucodec.OpenShardV2(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening shard: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	idx := reader.Lookup(entryName)
	if idx < 0 {
		fmt.Fprintf(os.Stderr, "entry not found: %s\n", entryName)
		os.Exit(1)
	}

	data, err := reader.ReadEntry(idx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading entry: %v\n", err)
		os.Exit(1)
	}

	if jsonOutput {
		var parsed any
		if err := json.Unmarshal(data, &parsed); err == nil {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(parsed)
			return
		}
		out := map[string]any{
			"name":   entryName,
			"size":   len(data),
			"hex":    hex.EncodeToString(data),
			"binary": true,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
		return
	}

	if isLikelyJSON(data) {
		var parsed any
		if err := json.Unmarshal(data, &parsed); err == nil {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(parsed)
			return
		}
	}

	if isTextData(data) {
		fmt.Print(string(data))
		if len(data) > 0 && data[len(data)-1] != '\n' {
			fmt.Println()
		}
		return
	}

	fmt.Printf("Binary data (%d bytes):\n", len(data))
	printHexDump(data)
}

func cmdStats(path string) {
	reader, err := ucodec.OpenShardV2(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening shard: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	h := reader.Header()
	count := reader.EntryCount()

	var totalDisk, totalOrig uint64
	var compressedCount, uncompressedCount int

	type entryStats struct {
		Name     string  `json:"name"`
		DiskSize uint64  `json:"disk_size"`
		OrigSize uint64  `json:"orig_size"`
		Ratio    float64 `json:"ratio"`
		Checksum string  `json:"checksum"`
	}

	entries := make([]entryStats, count)
	for i := 0; i < count; i++ {
		info := reader.GetEntryInfo(i)
		totalDisk += info.DiskSize
		totalOrig += info.OrigSize

		if info.IsCompressed() {
			compressedCount++
		} else {
			uncompressedCount++
		}

		ratio := float64(1)
		if info.OrigSize > 0 {
			ratio = float64(info.DiskSize) / float64(info.OrigSize)
		}

		entries[i] = entryStats{
			Name:     reader.EntryName(i),
			DiskSize: info.DiskSize,
			OrigSize: info.OrigSize,
			Ratio:    ratio,
			Checksum: fmt.Sprintf("0x%08x", info.Checksum),
		}
	}

	overallRatio := float64(1)
	if totalOrig > 0 {
		overallRatio = float64(totalDisk) / float64(totalOrig)
	}

	if jsonOutput {
		out := map[string]any{
			"total_disk_size":    totalDisk,
			"total_orig_size":    totalOrig,
			"compression_ratio":  overallRatio,
			"compressed_entries": compressedCount,
			"raw_entries":        uncompressedCount,
			"has_checksums":      h.Flags&ucodec.ShardV2FlagHasChecksums != 0,
			"entries":            entries,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
		return
	}

	fmt.Printf("Compression Statistics: %s\n\n", path)
	fmt.Printf("  Total Disk Size:    %s\n", humanSize(int64(totalDisk)))
	fmt.Printf("  Total Orig Size:    %s\n", humanSize(int64(totalOrig)))
	fmt.Printf("  Compression Ratio:  %.2f%% (%.3fx)\n", overallRatio*100, overallRatio)
	fmt.Printf("  Compressed Entries: %d\n", compressedCount)
	fmt.Printf("  Raw Entries:        %d\n", uncompressedCount)
	fmt.Printf("  Has Checksums:      %v\n", h.Flags&ucodec.ShardV2FlagHasChecksums != 0)

	fmt.Printf("\nPer-entry checksums:\n")
	fmt.Printf("%-40s %12s\n", "NAME", "CRC32C")
	fmt.Printf("%s\n", strings.Repeat("-", 54))
	for _, e := range entries {
		fmt.Printf("%-40s %12s\n", truncate(e.Name, 40), e.Checksum)
	}
}

func roleString(r ucodec.ShardRole) string {
	switch r {
	case ucodec.ShardRoleUnknown:
		return "Unknown"
	case ucodec.ShardRoleMoSH:
		return "MoSH"
	case ucodec.ShardRoleSample:
		return "Sample"
	case ucodec.ShardRoleGemmPanel:
		return "GemmPanel"
	case ucodec.ShardRoleManifest:
		return "Manifest"
	case ucodec.ShardRoleWShard:
		return "WShard"
	default:
		return fmt.Sprintf("Unknown(%d)", r)
	}
}

func compressionString(c uint8) string {
	switch c {
	case ucodec.CompressNone:
		return "none"
	case ucodec.CompressZstd:
		return "zstd"
	case ucodec.CompressLZ4:
		return "lz4"
	default:
		return fmt.Sprintf("unknown(%d)", c)
	}
}

func compressionTypeString(c uint8) string {
	switch c {
	case ucodec.CompressZstd:
		return "zstd"
	case ucodec.CompressLZ4:
		return "lz4"
	default:
		return "yes"
	}
}

func humanSize(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}

func isLikelyJSON(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	for _, b := range data {
		if b == ' ' || b == '\t' || b == '\n' || b == '\r' {
			continue
		}
		return b == '{' || b == '['
	}
	return false
}

func isTextData(data []byte) bool {
	if len(data) == 0 {
		return true
	}
	nonPrintable := 0
	for _, b := range data {
		if b < 32 && b != '\n' && b != '\r' && b != '\t' {
			nonPrintable++
		}
	}
	return float64(nonPrintable)/float64(len(data)) < 0.1
}

func printHexDump(data []byte) {
	const lineSize = 16
	maxLines := 32
	lines := (len(data) + lineSize - 1) / lineSize
	if lines > maxLines {
		lines = maxLines
	}

	for i := 0; i < lines; i++ {
		start := i * lineSize
		end := start + lineSize
		if end > len(data) {
			end = len(data)
		}

		fmt.Printf("%08x  ", start)

		for j := start; j < start+lineSize; j++ {
			if j < end {
				fmt.Printf("%02x ", data[j])
			} else {
				fmt.Print("   ")
			}
			if j == start+7 {
				fmt.Print(" ")
			}
		}

		fmt.Print(" |")
		for j := start; j < end; j++ {
			if data[j] >= 32 && data[j] < 127 {
				fmt.Printf("%c", data[j])
			} else {
				fmt.Print(".")
			}
		}
		fmt.Println("|")
	}

	if len(data) > maxLines*lineSize {
		fmt.Printf("... (%d more bytes)\n", len(data)-maxLines*lineSize)
	}
}

// cmdVerify verifies all checksums in the shard file.
// Returns 0 if all checksums pass, 1 if any fail.
func cmdVerify(path string) int {
	reader, err := ucodec.OpenShardV2(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening shard: %v\n", err)
		return 1
	}
	defer reader.Close()

	h := reader.Header()
	hasChecksums := h.Flags&ucodec.ShardV2FlagHasChecksums != 0

	if !hasChecksums {
		if jsonOutput {
			out := map[string]any{
				"status":  "no_checksums",
				"message": "file does not have checksums enabled",
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(out)
		} else {
			fmt.Println("Warning: file does not have checksums enabled")
		}
		return 0
	}

	count := reader.EntryCount()
	var passed, failed int
	var failures []map[string]any

	// Use CRC32C (Castagnoli) table
	crcTable := crc32.MakeTable(crc32.Castagnoli)

	for i := 0; i < count; i++ {
		name := reader.EntryName(i)
		info := reader.GetEntryInfo(i)

		// Read entry data (decompressed)
		data, err := reader.ReadEntry(i)
		if err != nil {
			failed++
			failures = append(failures, map[string]any{
				"name":  name,
				"error": err.Error(),
			})
			if verboseOutput && !jsonOutput {
				fmt.Printf("FAIL: %s - read error: %v\n", name, err)
			}
			continue
		}

		// Compute CRC32C
		computed := crc32.Checksum(data, crcTable)
		expected := info.Checksum

		if computed == expected {
			passed++
			if verboseOutput && !jsonOutput {
				fmt.Printf("OK:   %s (0x%08x)\n", name, computed)
			}
		} else {
			failed++
			failures = append(failures, map[string]any{
				"name":     name,
				"expected": fmt.Sprintf("0x%08x", expected),
				"computed": fmt.Sprintf("0x%08x", computed),
			})
			if !jsonOutput {
				fmt.Printf("FAIL: %s - expected 0x%08x, got 0x%08x\n", name, expected, computed)
			}
		}
	}

	if jsonOutput {
		out := map[string]any{
			"status":   "complete",
			"passed":   passed,
			"failed":   failed,
			"total":    count,
			"failures": failures,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
	} else if failed == 0 {
		fmt.Printf("Verified %d entries - all checksums OK\n", count)
	} else {
		fmt.Printf("\nVerification complete: %d passed, %d failed\n", passed, failed)
	}

	if failed > 0 {
		return 1
	}
	return 0
}

// cmdDiff compares two shard files and reports differences.
func cmdDiff(path1, path2 string) int {
	reader1, err := ucodec.OpenShardV2(path1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening %s: %v\n", path1, err)
		return 1
	}
	defer reader1.Close()

	reader2, err := ucodec.OpenShardV2(path2)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening %s: %v\n", path2, err)
		return 1
	}
	defer reader2.Close()

	// Build entry maps
	entries1 := make(map[string]int)
	entries2 := make(map[string]int)

	for i := 0; i < reader1.EntryCount(); i++ {
		entries1[reader1.EntryName(i)] = i
	}
	for i := 0; i < reader2.EntryCount(); i++ {
		entries2[reader2.EntryName(i)] = i
	}

	// Collect all unique names
	allNames := make(map[string]bool)
	for name := range entries1 {
		allNames[name] = true
	}
	for name := range entries2 {
		allNames[name] = true
	}

	// Sort names for consistent output
	names := make([]string, 0, len(allNames))
	for name := range allNames {
		names = append(names, name)
	}
	sort.Strings(names)

	var onlyIn1, onlyIn2, different, same []string
	var diffs []map[string]any

	for _, name := range names {
		idx1, in1 := entries1[name]
		idx2, in2 := entries2[name]

		if in1 && !in2 {
			onlyIn1 = append(onlyIn1, name)
			diffs = append(diffs, map[string]any{"name": name, "status": "only_in_first"})
		} else if !in1 && in2 {
			onlyIn2 = append(onlyIn2, name)
			diffs = append(diffs, map[string]any{"name": name, "status": "only_in_second"})
		} else {
			// Both have entry - compare checksums and sizes
			info1 := reader1.GetEntryInfo(idx1)
			info2 := reader2.GetEntryInfo(idx2)

			if info1.Checksum != info2.Checksum || info1.OrigSize != info2.OrigSize {
				different = append(different, name)
				diffs = append(diffs, map[string]any{
					"name":   name,
					"status": "modified",
					"file1":  map[string]any{"size": info1.OrigSize, "checksum": fmt.Sprintf("0x%08x", info1.Checksum)},
					"file2":  map[string]any{"size": info2.OrigSize, "checksum": fmt.Sprintf("0x%08x", info2.Checksum)},
				})
			} else {
				same = append(same, name)
			}
		}
	}

	if jsonOutput {
		out := map[string]any{
			"file1":          path1,
			"file2":          path2,
			"only_in_first":  onlyIn1,
			"only_in_second": onlyIn2,
			"modified":       different,
			"identical":      len(same),
			"differences":    diffs,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
	} else {
		if len(onlyIn1) > 0 {
			fmt.Printf("Only in %s:\n", path1)
			for _, name := range onlyIn1 {
				fmt.Printf("  - %s\n", name)
			}
		}
		if len(onlyIn2) > 0 {
			fmt.Printf("Only in %s:\n", path2)
			for _, name := range onlyIn2 {
				fmt.Printf("  + %s\n", name)
			}
		}
		if len(different) > 0 {
			fmt.Printf("Modified:\n")
			for _, name := range different {
				info1 := reader1.GetEntryInfo(entries1[name])
				info2 := reader2.GetEntryInfo(entries2[name])
				fmt.Printf("  ~ %s (%s -> %s)\n", name,
					humanSize(int64(info1.OrigSize)),
					humanSize(int64(info2.OrigSize)))
			}
		}

		fmt.Printf("\nSummary: %d identical, %d modified, %d added, %d removed\n",
			len(same), len(different), len(onlyIn2), len(onlyIn1))
	}

	if len(onlyIn1) > 0 || len(onlyIn2) > 0 || len(different) > 0 {
		return 1
	}
	return 0
}

// cmdAudit performs a comprehensive integrity audit of a shard file.
func cmdAudit(path string) int {
	info, err := os.Stat(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error accessing file: %v\n", err)
		return 1
	}

	reader, err := ucodec.OpenShardV2(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening shard: %v\n", err)
		return 1
	}
	defer reader.Close()

	h := reader.Header()
	count := reader.EntryCount()

	// Audit checks
	type auditIssue struct {
		Severity string `json:"severity"` // error, warning, info
		Code     string `json:"code"`
		Message  string `json:"message"`
		Entry    string `json:"entry,omitempty"`
	}

	var issues []auditIssue

	// Check 1: File size consistency
	if h.TotalFileSize != uint64(info.Size()) {
		issues = append(issues, auditIssue{
			Severity: "error",
			Code:     "E001",
			Message:  fmt.Sprintf("header file size (%d) != actual size (%d)", h.TotalFileSize, info.Size()),
		})
	}

	// Check 2: Checksums enabled
	hasChecksums := h.Flags&ucodec.ShardV2FlagHasChecksums != 0
	if !hasChecksums {
		issues = append(issues, auditIssue{
			Severity: "warning",
			Code:     "W001",
			Message:  "checksums not enabled - data integrity cannot be verified",
		})
	}

	// Check 3: Alignment
	if h.Alignment == 0 {
		issues = append(issues, auditIssue{
			Severity: "info",
			Code:     "I001",
			Message:  "no alignment set - may impact mmap performance",
		})
	}

	// Check 4: Verify all entries can be read and checksums match
	crcTable := crc32.MakeTable(crc32.Castagnoli)
	var checksumErrors, readErrors int
	var totalOrigSize, totalDiskSize uint64

	for i := 0; i < count; i++ {
		name := reader.EntryName(i)
		entryInfo := reader.GetEntryInfo(i)

		totalOrigSize += entryInfo.OrigSize
		totalDiskSize += entryInfo.DiskSize

		// Try to read entry
		data, err := reader.ReadEntry(i)
		if err != nil {
			readErrors++
			issues = append(issues, auditIssue{
				Severity: "error",
				Code:     "E002",
				Message:  fmt.Sprintf("read error: %v", err),
				Entry:    name,
			})
			continue
		}

		// Verify checksum if enabled
		if hasChecksums {
			computed := crc32.Checksum(data, crcTable)
			if computed != entryInfo.Checksum {
				checksumErrors++
				issues = append(issues, auditIssue{
					Severity: "error",
					Code:     "E003",
					Message:  fmt.Sprintf("checksum mismatch: expected 0x%08x, got 0x%08x", entryInfo.Checksum, computed),
					Entry:    name,
				})
			}
		}

		// Check for empty entries
		if entryInfo.OrigSize == 0 {
			issues = append(issues, auditIssue{
				Severity: "warning",
				Code:     "W002",
				Message:  "empty entry",
				Entry:    name,
			})
		}

		// Check for reserved entry names
		if len(name) >= 4 && strings.HasPrefix(name, "__") && strings.HasSuffix(name, "__") {
			// Reserved entry - check if valid
			validReserved := map[string]bool{
				"__metadata__": true,
				"__schema__":   true,
				"__manifest__": true,
			}
			if !validReserved[name] {
				issues = append(issues, auditIssue{
					Severity: "info",
					Code:     "I002",
					Message:  "non-standard reserved entry name",
					Entry:    name,
				})
			}
		}
	}

	// Check 5: Compression efficiency
	if totalOrigSize > 0 {
		ratio := float64(totalDiskSize) / float64(totalOrigSize)
		if ratio > 1.05 {
			issues = append(issues, auditIssue{
				Severity: "warning",
				Code:     "W003",
				Message:  fmt.Sprintf("compression is ineffective (%.1f%% of original size)", ratio*100),
			})
		}
	}

	// Determine overall status
	var errorCount, warningCount, infoCount int
	for _, issue := range issues {
		switch issue.Severity {
		case "error":
			errorCount++
		case "warning":
			warningCount++
		case "info":
			infoCount++
		}
	}

	status := "pass"
	exitCode := 0
	if errorCount > 0 {
		status = "fail"
		exitCode = 1
	} else if warningCount > 0 {
		status = "warn"
	}

	if jsonOutput {
		out := map[string]any{
			"file":          path,
			"status":        status,
			"entry_count":   count,
			"total_size":    totalOrigSize,
			"disk_size":     totalDiskSize,
			"has_checksums": hasChecksums,
			"error_count":   errorCount,
			"warning_count": warningCount,
			"info_count":    infoCount,
			"issues":        issues,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
	} else {
		fmt.Printf("Audit Report: %s\n", path)
		fmt.Printf("================================================================================\n")
		fmt.Printf("  Status:          %s\n", strings.ToUpper(status))
		fmt.Printf("  Entry Count:     %d\n", count)
		fmt.Printf("  Total Size:      %s (disk: %s)\n", humanSize(int64(totalOrigSize)), humanSize(int64(totalDiskSize)))
		fmt.Printf("  Has Checksums:   %v\n", hasChecksums)
		fmt.Printf("  Role:            %s\n", roleString(h.Role))
		fmt.Printf("  Compression:     %s\n", compressionString(h.CompressionDefault))
		fmt.Printf("  Alignment:       %d bytes\n", h.Alignment)
		fmt.Printf("\n")

		if len(issues) > 0 {
			fmt.Printf("Issues Found:\n")
			for _, issue := range issues {
				prefix := ""
				switch issue.Severity {
				case "error":
					prefix = "ERROR"
				case "warning":
					prefix = "WARN "
				case "info":
					prefix = "INFO "
				}
				if issue.Entry != "" {
					fmt.Printf("  [%s] %s: %s (%s)\n", prefix, issue.Code, issue.Message, issue.Entry)
				} else {
					fmt.Printf("  [%s] %s: %s\n", prefix, issue.Code, issue.Message)
				}
			}
			fmt.Printf("\n")
		}

		fmt.Printf("Summary: %d errors, %d warnings, %d info\n", errorCount, warningCount, infoCount)
	}

	return exitCode
}

// HuggingFace Hub API types
type hfRepoFile struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	OID  string `json:"oid"`
	LFS  *struct {
		SHA256      string `json:"sha256"`
		Size        int64  `json:"size"`
		PointerSize int    `json:"pointerSize"`
	} `json:"lfs,omitempty"`
}

// cmdDownload downloads files from HuggingFace Hub.
func cmdDownload(repoID, pattern string) int {
	// Default pattern for model weights
	if pattern == "" {
		pattern = "*.safetensors"
	}

	// List files in repository
	files, err := listHFRepoFiles(repoID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error listing repo files: %v\n", err)
		return 1
	}

	// Filter files by pattern
	var matched []hfRepoFile
	for _, f := range files {
		if f.Type != "file" {
			continue
		}
		match, err := filepath.Match(pattern, filepath.Base(f.Path))
		if err != nil {
			fmt.Fprintf(os.Stderr, "invalid pattern: %v\n", err)
			return 1
		}
		if match {
			matched = append(matched, f)
		}
	}

	if len(matched) == 0 {
		fmt.Fprintf(os.Stderr, "no files match pattern '%s' in %s\n", pattern, repoID)
		if jsonOutput {
			out := map[string]any{
				"status":  "no_matches",
				"repo_id": repoID,
				"pattern": pattern,
			}
			json.NewEncoder(os.Stdout).Encode(out)
		}
		return 1
	}

	// Show files to download
	var totalSize int64
	for _, f := range matched {
		totalSize += f.Size
	}

	if jsonOutput {
		fileList := make([]map[string]any, len(matched))
		for i, f := range matched {
			fileList[i] = map[string]any{
				"path": f.Path,
				"size": f.Size,
			}
		}
		out := map[string]any{
			"repo_id":    repoID,
			"pattern":    pattern,
			"file_count": len(matched),
			"total_size": totalSize,
			"files":      fileList,
			"output_dir": outputDir,
		}
		json.NewEncoder(os.Stdout).Encode(out)
	} else {
		fmt.Printf("Repository: %s\n", repoID)
		fmt.Printf("Pattern: %s\n", pattern)
		fmt.Printf("Files to download: %d (%s total)\n\n", len(matched), humanSize(totalSize))
		for _, f := range matched {
			fmt.Printf("  %s (%s)\n", f.Path, humanSize(f.Size))
		}
		fmt.Println()
	}

	// Ensure output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error creating output directory: %v\n", err)
		return 1
	}

	// Download each file
	var downloaded, failed int
	var downloadedBytes int64
	startTime := time.Now()

	for i, f := range matched {
		outPath := filepath.Join(outputDir, filepath.Base(f.Path))

		if !jsonOutput {
			fmt.Printf("[%d/%d] Downloading %s...\n", i+1, len(matched), f.Path)
		}

		n, err := downloadHFFile(repoID, f.Path, outPath)
		if err != nil {
			failed++
			if !jsonOutput {
				fmt.Fprintf(os.Stderr, "  ERROR: %v\n", err)
			}
			continue
		}

		downloaded++
		downloadedBytes += n

		if !jsonOutput {
			fmt.Printf("  Saved to %s (%s)\n", outPath, humanSize(n))
		}
	}

	elapsed := time.Since(startTime)
	speed := float64(downloadedBytes) / elapsed.Seconds() / 1024 / 1024 // MB/s

	if jsonOutput {
		out := map[string]any{
			"status":          "complete",
			"downloaded":      downloaded,
			"failed":          failed,
			"total_bytes":     downloadedBytes,
			"elapsed_seconds": elapsed.Seconds(),
			"speed_mbps":      speed,
		}
		json.NewEncoder(os.Stdout).Encode(out)
	} else {
		fmt.Printf("\nDownload complete: %d files (%s) in %.1fs (%.1f MB/s)\n",
			downloaded, humanSize(downloadedBytes), elapsed.Seconds(), speed)
		if failed > 0 {
			fmt.Printf("Failed: %d files\n", failed)
		}
	}

	if failed > 0 {
		return 1
	}
	return 0
}

// listHFRepoFiles lists all files in a HuggingFace repository.
func listHFRepoFiles(repoID string) ([]hfRepoFile, error) {
	url := fmt.Sprintf("https://huggingface.co/api/models/%s", repoID)

	client := &http.Client{Timeout: 30 * time.Second}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set User-Agent to avoid being blocked
	req.Header.Set("User-Agent", "shardctl/1.0")

	// Check for HF token in environment
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return nil, fmt.Errorf("repository not found: %s", repoID)
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	// Parse model info response
	var modelInfo struct {
		Siblings []struct {
			RFilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&modelInfo); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to hfRepoFile format
	files := make([]hfRepoFile, 0, len(modelInfo.Siblings))
	for _, s := range modelInfo.Siblings {
		files = append(files, hfRepoFile{
			Type: "file",
			Path: s.RFilename,
			Size: 0, // Size not available in this endpoint
		})
	}

	return files, nil
}

// downloadHFFile downloads a single file from HuggingFace Hub.
func downloadHFFile(repoID, filePath, outPath string) (int64, error) {
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, filePath)

	client := &http.Client{
		Timeout: 0, // No timeout for large files
	}

	// Check for HF token in environment
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return 0, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", "shardctl/1.0")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 401 {
		return 0, fmt.Errorf("authentication required - set HF_TOKEN environment variable")
	}
	if resp.StatusCode == 404 {
		return 0, fmt.Errorf("file not found: %s", filePath)
	}
	if resp.StatusCode != 200 {
		return 0, fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	// Create output file
	out, err := os.Create(outPath)
	if err != nil {
		return 0, fmt.Errorf("failed to create output file: %w", err)
	}
	defer out.Close()

	// Copy with progress (if verbose and not JSON)
	var written int64
	if verboseOutput && !jsonOutput {
		written, err = copyWithProgress(out, resp.Body, resp.ContentLength)
	} else {
		written, err = io.Copy(out, resp.Body)
	}

	if err != nil {
		os.Remove(outPath) // Clean up partial file
		return 0, fmt.Errorf("download failed: %w", err)
	}

	return written, nil
}

// copyWithProgress copies data while showing progress.
func copyWithProgress(dst io.Writer, src io.Reader, total int64) (int64, error) {
	buf := make([]byte, 32*1024) // 32KB buffer
	var written int64
	lastUpdate := time.Now()

	for {
		n, err := src.Read(buf)
		if n > 0 {
			nw, ew := dst.Write(buf[:n])
			if nw > 0 {
				written += int64(nw)
			}
			if ew != nil {
				return written, ew
			}
		}

		// Update progress every 100ms
		if time.Since(lastUpdate) > 100*time.Millisecond {
			if total > 0 {
				pct := float64(written) / float64(total) * 100
				fmt.Printf("\r  Progress: %.1f%% (%s / %s)", pct, humanSize(written), humanSize(total))
			} else {
				fmt.Printf("\r  Downloaded: %s", humanSize(written))
			}
			lastUpdate = time.Now()
		}

		if err == io.EOF {
			fmt.Printf("\r%80s\r", "") // Clear progress line
			return written, nil
		}
		if err != nil {
			return written, err
		}
	}
}

// cmdConvert converts HuggingFace safetensors to MoSH format.
func cmdConvert(modelDir string) int {
	// Determine output path
	outMoSH := outputDir
	if outMoSH == "." {
		// Default output name based on model dir
		baseName := filepath.Base(modelDir)
		outMoSH = baseName + ".mosh"
	}
	if !strings.HasSuffix(outMoSH, ".mosh") && !strings.HasSuffix(outMoSH, ".umsh") {
		outMoSH = outMoSH + ".mosh"
	}

	outMeta := strings.TrimSuffix(outMoSH, filepath.Ext(outMoSH)) + ".meta.json"

	cfg := hfimport.ConvertConfig{
		HFDir:     modelDir,
		OutUMSH:   outMoSH,
		OutMeta:   outMeta,
		QuantBits: quantBits,
	}

	if !jsonOutput {
		fmt.Printf("Converting %s -> %s\n", modelDir, outMoSH)
		if quantBits > 0 {
			fmt.Printf("Quantization: %d-bit\n", quantBits)
		}
		fmt.Println()
	}

	if err := hfimport.ConvertHFToUMSH(cfg); err != nil {
		if jsonOutput {
			out := map[string]any{
				"status": "error",
				"error":  err.Error(),
			}
			json.NewEncoder(os.Stdout).Encode(out)
		} else {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
		}
		return 1
	}

	// Get file sizes
	moshInfo, _ := os.Stat(outMoSH)
	metaInfo, _ := os.Stat(outMeta)

	if jsonOutput {
		out := map[string]any{
			"status":     "complete",
			"mosh_file":  outMoSH,
			"meta_file":  outMeta,
			"mosh_size":  moshInfo.Size(),
			"meta_size":  metaInfo.Size(),
			"quant_bits": quantBits,
		}
		json.NewEncoder(os.Stdout).Encode(out)
	} else {
		fmt.Printf("\nOutput:\n")
		fmt.Printf("  MoSH:     %s (%s)\n", outMoSH, humanSize(moshInfo.Size()))
		fmt.Printf("  Metadata: %s (%s)\n", outMeta, humanSize(metaInfo.Size()))
	}

	return 0
}

// cmdInspect shows contents of a SafeTensors file.
func cmdInspect(path string) int {
	stf, err := hfimport.LoadSafetensors(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading safetensors: %v\n", err)
		return 1
	}
	defer stf.Close()

	info, _ := os.Stat(path)

	if jsonOutput {
		tensors := make([]map[string]any, len(stf.Tensors))
		for i, t := range stf.Tensors {
			tensors[i] = map[string]any{
				"name":   t.Name,
				"dtype":  t.DType,
				"shape":  t.Shape,
				"offset": t.Offset,
				"length": t.Length,
			}
		}
		out := map[string]any{
			"file":        path,
			"file_size":   info.Size(),
			"data_offset": stf.DataOffset,
			"num_tensors": len(stf.Tensors),
			"tensors":     tensors,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
		return 0
	}

	fmt.Printf("SafeTensors File: %s\n", path)
	fmt.Printf("  File Size:   %s\n", humanSize(info.Size()))
	fmt.Printf("  Data Offset: %d bytes\n", stf.DataOffset)
	fmt.Printf("  Tensors:     %d\n\n", len(stf.Tensors))

	// Sort by offset for display
	tensors := make([]hfimport.TensorInfo, len(stf.Tensors))
	copy(tensors, stf.Tensors)
	sort.Slice(tensors, func(i, j int) bool {
		return tensors[i].Offset < tensors[j].Offset
	})

	fmt.Printf("%-50s %-6s %-20s %12s\n", "NAME", "DTYPE", "SHAPE", "SIZE")
	fmt.Printf("%s\n", strings.Repeat("-", 95))

	var totalBytes int64
	for _, t := range tensors {
		shapeStr := fmt.Sprintf("%v", t.Shape)
		if len(shapeStr) > 20 {
			shapeStr = shapeStr[:17] + "..."
		}
		fmt.Printf("%-50s %-6s %-20s %12s\n",
			truncate(t.Name, 50),
			t.DType,
			shapeStr,
			humanSize(t.Length))
		totalBytes += t.Length
	}

	fmt.Printf("\nTotal tensor data: %s\n", humanSize(totalBytes))
	return 0
}
