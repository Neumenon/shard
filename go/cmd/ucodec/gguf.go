// gguf.go - GGUF to MoSH conversion command
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/phenomenon0/Agent-GO/cowrie/ucodec/hfimport"
)

func runGGUF(args []string) error {
	fs := flag.NewFlagSet("gguf", flag.ExitOnError)

	input := fs.String("input", "", "Input GGUF file path")
	inputShort := fs.String("i", "", "Input GGUF file path (short)")
	output := fs.String("out", "", "Output MoSH file path (.mosh)")
	outputShort := fs.String("o", "", "Output MoSH file path (short)")
	verbose := fs.Bool("verbose", false, "Show detailed progress")
	verboseShort := fs.Bool("v", false, "Show detailed progress (short)")

	fs.Usage = func() {
		fmt.Print(`ucodec gguf - Convert GGUF model to MoSH format (Model Shard)

Usage:
  ucodec gguf --input <gguf-path> --out <mosh-path> [options]

Options:
  --input, -i <path>   Input GGUF file path (e.g., Ollama blob)
  --out, -o <path>     Output MoSH file path (.mosh)
  --verbose, -v        Show detailed progress and tensor mappings

Example:
  # Convert Ollama's Qwen2.5:7b model
  ucodec gguf -i ~/.ollama/models/blobs/sha256-2bada... -o models/qwen2.5-7b.mosh

  # With verbose output
  ucodec gguf --input model.gguf --out model.mosh --verbose

Notes:
  - GGUF files from Ollama are stored in ~/.ollama/models/blobs/
  - Find the correct blob with: ollama show <model> --modelfile
  - Output format is MoSH (Model Shard) using Shard.1 spec
  - Unmapped tensors are logged to stderr for debugging
`)
	}

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Handle short flags
	if *inputShort != "" && *input == "" {
		*input = *inputShort
	}
	if *outputShort != "" && *output == "" {
		*output = *outputShort
	}
	if *verboseShort {
		*verbose = true
	}

	// Validate required arguments
	if *input == "" {
		return fmt.Errorf("--input is required")
	}
	if *output == "" {
		return fmt.Errorf("--out is required")
	}

	// Auto-add .mosh extension if missing
	if !strings.HasSuffix(*output, ".mosh") && !strings.HasSuffix(*output, ".shard") {
		*output = *output + ".mosh"
	}

	// Validate input file exists
	if _, err := os.Stat(*input); os.IsNotExist(err) {
		return fmt.Errorf("input file not found: %s", *input)
	}

	// Validate GGUF magic header
	if err := validateGGUFMagic(*input); err != nil {
		return fmt.Errorf("invalid GGUF file: %w", err)
	}

	if *verbose {
		fmt.Printf("Converting GGUF to MoSH (Model Shard)\n")
		fmt.Printf("  Input:  %s\n", *input)
		fmt.Printf("  Output: %s\n", *output)
		fmt.Println()
	}

	// Run conversion
	cfg := hfimport.GGUFConvertConfig{
		GGUFPath: *input,
		OutMoSH:  *output,
	}

	if err := hfimport.ConvertGGUFToMoSH(cfg); err != nil {
		return fmt.Errorf("conversion failed: %w", err)
	}

	fmt.Printf("\nSuccess! Model saved to %s\n", *output)
	fmt.Printf("Format: MoSH (Model Shard, Shard.1)\n")
	return nil
}

// validateGGUFMagic checks if the file has a valid GGUF magic header.
func validateGGUFMagic(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// GGUF magic is "GGUF" (0x46554747 little-endian)
	magic := make([]byte, 4)
	if _, err := f.Read(magic); err != nil {
		return fmt.Errorf("failed to read magic: %w", err)
	}

	if string(magic) != "GGUF" {
		return fmt.Errorf("not a GGUF file (magic: %x, expected: GGUF)", magic)
	}

	return nil
}
