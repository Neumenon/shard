// ucodec - Universal Codec CLI for model compression and inference
//
// A CLI toolchain for quantizing, pruning, and running inference on UMSH model files.
//
// Commands:
//
//	ucodec quantize  - Apply INT8 quantization to UMSH model
//	ucodec prune     - Apply 2:4 structured sparsity
//	ucodec export    - Convert HuggingFace model to UMSH
//	ucodec run       - Run inference on UMSH model
//	ucodec info      - Show info about UMSH file
//
// Example pipeline:
//
//	ucodec export --model meta-llama/Llama-2-7b-hf --out model.fp16.umsh
//	ucodec quantize --config compress.yaml --in model.fp16.umsh --out model.int8.umsh
//	ucodec prune --config compress.yaml --in model.int8.umsh --out model.sparse.umsh
//	ucodec run --umsh model.sparse.umsh --prompt "Hello, world"
package main

import (
	"fmt"
	"os"
)

const version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	var err error
	switch cmd {
	case "export", "convert", "e":
		err = runExport(args)
	case "gguf", "convert-gguf", "g":
		err = runGGUF(args)
	case "quantize", "quant", "q":
		err = runQuantize(args)
	case "prune", "sparse", "p":
		err = runPrune(args)
	case "run", "r":
		err = runRun(args)
	case "info", "inspect", "i":
		err = runInfo(args)
	case "generate-config", "gen-config":
		err = runGenerateConfig(args)
	case "version", "--version", "-v":
		fmt.Printf("ucodec version %s\n", version)
		return
	case "help", "--help", "-h":
		if len(args) > 0 {
			subcommandHelp(args[0])
		} else {
			usage()
		}
		return
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Print(`ucodec - Universal Codec CLI for model compression

Usage:
  ucodec <command> [options]

Commands:
  export        Convert HuggingFace model to UMSH format
  gguf          Convert GGUF model to UMSH format (e.g., Ollama models)
  quantize      Apply INT8 quantization to UMSH model
  prune         Apply structured sparsity (2:4 pruning)
  run           Run inference on UMSH model
  info          Show information about UMSH file
  gen-config    Generate a default compression config file

Shortcuts:
  quant, q      Same as quantize
  sparse, p     Same as prune
  r             Same as run
  i, inspect    Same as info

Options:
  --version     Show version
  --help        Show this help

Examples:
  # Generate a default compression config
  ucodec gen-config --out compress.yaml

  # Quantize a model
  ucodec quantize --config compress.yaml --in model.fp16.umsh --out model.int8.umsh

  # Apply 2:4 sparsity
  ucodec prune --config compress.yaml --in model.int8.umsh --out model.sparse.umsh

  # Show model info
  ucodec info model.sparse.umsh

For help on a specific command:
  ucodec help <command>
`)
}

func subcommandHelp(cmd string) {
	switch cmd {
	case "quantize", "quant", "q":
		fmt.Print(`ucodec quantize - Apply INT8 quantization to UMSH model

Usage:
  ucodec quantize [options]

Options:
  --config, -c <path>   Path to compression config YAML
  --in, -i <path>       Input UMSH file (overrides config)
  --out, -o <path>      Output UMSH file (overrides config)
  --scheme <name>       Quantization scheme (default: per_channel_symmetric_int8)
  --verbose, -v         Show detailed progress

Quantization Schemes:
  per_channel_symmetric_int8   Per-channel symmetric INT8 (default, best quality)
  per_tensor_symmetric_int8    Per-tensor symmetric INT8 (faster, less accurate)

Example:
  ucodec quantize --config compress.yaml --in model.fp16.umsh --out model.int8.umsh
`)

	case "prune", "sparse", "p":
		fmt.Print(`ucodec prune - Apply structured sparsity to UMSH model

Usage:
  ucodec prune [options]

Options:
  --config, -c <path>   Path to compression config YAML
  --in, -i <path>       Input UMSH file (overrides config)
  --out, -o <path>      Output UMSH file (overrides config)
  --pattern <N:M>       Sparsity pattern (default: 2:4)
  --verbose, -v         Show detailed progress

Sparsity Patterns:
  2:4   Keep 2 of every 4 elements (50% sparsity, NVIDIA Ampere+ optimized)
  1:4   Keep 1 of every 4 elements (75% sparsity)

Example:
  ucodec prune --config compress.yaml --in model.int8.umsh --out model.sparse.umsh
`)

	case "info", "inspect", "i":
		fmt.Print(`ucodec info - Show information about UMSH file

Usage:
  ucodec info [options] <umsh-file>

Options:
  --tensors, -t     List all tensor names and shapes
  --json            Output in JSON format
  --verbose, -v     Show detailed tensor info

Example:
  ucodec info model.umsh
  ucodec info --tensors model.umsh
`)

	case "run", "r":
		fmt.Print(`ucodec run - Run inference on UMSH model

Usage:
  ucodec run [options] <umsh-file>

Options:
  --umsh, -m <path>     Path to UMSH model file
  --prompt, -p <text>   Input prompt for generation
  --max-tokens <n>      Maximum tokens to generate (default: 100)
  --temperature <f>     Sampling temperature (default: 0.7)
  --interactive, -i     Interactive chat mode
  --serve               Start HTTP server mode
  --port <n>            Server port (default: 11434)
  --verbose, -v         Show detailed model info

Examples:
  # Single prompt
  ucodec run --umsh model.umsh --prompt "Hello, world"

  # Interactive mode
  ucodec run --umsh model.umsh --interactive

  # Server mode (Ollama-compatible)
  ucodec run --umsh model.umsh --serve --port 11434

Note: Current version uses a placeholder tokenizer. Full tokenizer support
will be added in a future release.
`)

	case "gen-config", "generate-config":
		fmt.Print(`ucodec gen-config - Generate a default compression config

Usage:
  ucodec gen-config [options]

Options:
  --out, -o <path>     Output config file (default: compress.yaml)
  --model <name>       Model name for config (e.g., meta-llama/Llama-2-7b-hf)
  --input <path>       Input UMSH path for config
  --output <path>      Output UMSH path for config

Example:
  ucodec gen-config --out llama2.compress.yaml --model meta-llama/Llama-2-7b-hf
`)

	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", cmd)
		usage()
	}
}
