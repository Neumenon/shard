// export.go - HuggingFace to UMSH export command
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/phenomenon0/Agent-GO/cowrie/ucodec/hfimport"
)

func runExport(args []string) error {
	fs := flag.NewFlagSet("export", flag.ExitOnError)

	hfDir := fs.String("hf", "", "Path to HuggingFace model directory")
	modelPath := fs.String("model", "", "Alias for --hf")
	outUMSH := fs.String("out", "", "Output UMSH file path")
	outMeta := fs.String("meta", "", "Output metadata JSON path (default: <out>.meta.json)")
	quantBits := fs.Int("quant", 0, "Quantization bits (0=none, 8=int8)")
	pruneRatio := fs.Float64("prune", 0, "Prune ratio (0=none, 0.5=50% sparsity)")
	verbose := fs.Bool("verbose", false, "Show detailed progress")
	_ = verbose // Not used yet

	fs.Usage = func() {
		fmt.Print(`ucodec export - Convert HuggingFace model to UMSH format

Usage:
  ucodec export --hf <path> --out <output.umsh> [options]

Options:
  --hf, --model <path>   Path to HuggingFace model directory
  --out, -o <path>       Output UMSH file path
  --meta <path>          Output metadata JSON path (default: <out>.meta.json)
  --quant <bits>         Quantization bits (0=none, 8=int8)
  --prune <ratio>        Prune ratio (0=none, 0.5=50% sparsity)
  --verbose, -v          Show detailed progress

Example:
  ucodec export --hf /path/to/TinyLlama-1.1B-Chat-v1.0 --out tinyllama.umsh --quant 8
`)
	}

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Handle alias
	if *modelPath != "" && *hfDir == "" {
		*hfDir = *modelPath
	}

	if *hfDir == "" {
		return fmt.Errorf("--hf path is required")
	}
	if *outUMSH == "" {
		return fmt.Errorf("--out path is required")
	}

	// Default meta path
	if *outMeta == "" {
		ext := filepath.Ext(*outUMSH)
		base := (*outUMSH)[:len(*outUMSH)-len(ext)]
		*outMeta = base + ".meta.json"
	}

	cfg := hfimport.ConvertConfig{
		HFDir:      *hfDir,
		OutUMSH:    *outUMSH,
		OutMeta:    *outMeta,
		QuantBits:  *quantBits,
		PruneRatio: *pruneRatio,
	}

	fmt.Printf("Converting HuggingFace model: %s\n", *hfDir)
	fmt.Printf("  Output UMSH: %s\n", *outUMSH)
	fmt.Printf("  Output Meta: %s\n", *outMeta)
	if *quantBits > 0 {
		fmt.Printf("  Quantization: INT%d\n", *quantBits)
	}
	if *pruneRatio > 0 {
		fmt.Printf("  Pruning: %.0f%%\n", *pruneRatio*100)
	}
	fmt.Println()

	if err := hfimport.ConvertHFToUMSH(cfg); err != nil {
		return fmt.Errorf("export failed: %w", err)
	}

	// Copy tokenizer.json if present
	tokSrc := filepath.Join(*hfDir, "tokenizer.json")
	if _, err := os.Stat(tokSrc); err == nil {
		ext := filepath.Ext(*outUMSH)
		base := (*outUMSH)[:len(*outUMSH)-len(ext)]
		tokDst := base + ".tokenizer.json"

		if err := copyFile(tokSrc, tokDst); err != nil {
			fmt.Printf("  Warning: could not copy tokenizer.json: %v\n", err)
		} else {
			fmt.Printf("  Tokenizer copied to %s\n", tokDst)
		}
	}

	fmt.Printf("\nDone! Model exported to %s\n", *outUMSH)
	return nil
}

// copyFile copies a file from src to dst.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}
