package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/phenomenon0/Agent-GO/pkg/quant"
	"github.com/phenomenon0/Agent-GO/cowrie/ucodec"
)

func runInfo(args []string) error {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	showTensors := fs.Bool("tensors", false, "List all tensor names and shapes")
	fs.Bool("t", false, "List all tensor names and shapes (shorthand)")
	jsonOutput := fs.Bool("json", false, "Output in JSON format")
	verbose := fs.Bool("verbose", false, "Show detailed tensor info")
	fs.Bool("v", false, "Verbose (shorthand)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Handle shorthand flags
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "t" {
			*showTensors = true
		}
		if f.Name == "v" {
			*verbose = true
		}
	})

	if fs.NArg() < 1 {
		return fmt.Errorf("usage: ucodec info <umsh-file>")
	}

	umshPath := fs.Arg(0)

	// Open UMSH file
	reader, err := ucodec.OpenUMSH(umshPath)
	if err != nil {
		return fmt.Errorf("open UMSH: %w", err)
	}
	defer reader.Close()

	// Collect info
	info := UMSHInfo{
		Path:        umshPath,
		TensorCount: reader.TensorCount(),
		TensorNames: reader.TensorNames(),
	}

	// Get file size
	if fi, err := os.Stat(umshPath); err == nil {
		info.FileSize = fi.Size()
		info.FileSizeHuman = humanSize(fi.Size())
	}

	// Collect tensor details if requested
	if *showTensors || *verbose {
		info.Tensors = make([]TensorInfo, 0, len(info.TensorNames))
		for _, name := range info.TensorNames {
			tensor, err := reader.GetTensor(name)
			if err != nil {
				continue
			}
			ti := TensorInfo{
				Name:      name,
				DType:     ucodec.DTypeName(tensor.DType),
				Shape:     tensor.Shape(),
				Elements:  tensor.NumElements(),
				ByteSize:  tensor.ByteSize(),
				Sparsity:  quant.CountSparsity(tensor),
			}
			info.Tensors = append(info.Tensors, ti)
			info.TotalElements += ti.Elements
			info.TotalBytes += ti.ByteSize
		}
	}

	// Output
	if *jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(info)
	}

	// Human-readable output
	fmt.Printf("UMSH File: %s\n", info.Path)
	fmt.Printf("  File Size:    %s (%d bytes)\n", info.FileSizeHuman, info.FileSize)
	fmt.Printf("  Tensor Count: %d\n", info.TensorCount)

	if *showTensors || *verbose {
		fmt.Printf("  Total Elements: %d\n", info.TotalElements)
		fmt.Printf("  Total Bytes:    %s (%d bytes)\n", humanSize(info.TotalBytes), info.TotalBytes)
		fmt.Println()
		fmt.Println("Tensors:")
		for _, t := range info.Tensors {
			sparsityStr := ""
			if t.Sparsity > 0.01 {
				sparsityStr = fmt.Sprintf(" (%.1f%% sparse)", t.Sparsity*100)
			}
			if *verbose {
				fmt.Printf("  %-50s %s %v  %d elements  %s%s\n",
					t.Name, t.DType, t.Shape, t.Elements, humanSize(t.ByteSize), sparsityStr)
			} else {
				fmt.Printf("  %-50s %s %v%s\n", t.Name, t.DType, t.Shape, sparsityStr)
			}
		}
	}

	return nil
}

// UMSHInfo holds information about a UMSH file.
type UMSHInfo struct {
	Path          string       `json:"path"`
	FileSize      int64        `json:"file_size"`
	FileSizeHuman string       `json:"file_size_human"`
	TensorCount   int          `json:"tensor_count"`
	TensorNames   []string     `json:"tensor_names,omitempty"`
	TotalElements int64        `json:"total_elements,omitempty"`
	TotalBytes    int64        `json:"total_bytes,omitempty"`
	Tensors       []TensorInfo `json:"tensors,omitempty"`
}

// TensorInfo holds information about a single tensor.
type TensorInfo struct {
	Name     string   `json:"name"`
	DType    string   `json:"dtype"`
	Shape    []uint64 `json:"shape"`
	Elements int64    `json:"elements"`
	ByteSize int64    `json:"byte_size"`
	Sparsity float64  `json:"sparsity,omitempty"`
}

// humanSize returns a human-readable size string.
func humanSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func runGenerateConfig(args []string) error {
	fs := flag.NewFlagSet("gen-config", flag.ExitOnError)
	outPath := fs.String("out", "compress.yaml", "Output config file path")
	fs.String("o", "", "Output path (shorthand)")
	modelName := fs.String("model", "", "Model name for config")
	inputUMSH := fs.String("input", "", "Input UMSH path for config")
	outputUMSH := fs.String("output", "", "Output UMSH path for config")

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Handle shorthand
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "o" && *outPath == "compress.yaml" {
			if val := fs.Lookup("o").Value.String(); val != "" {
				*outPath = val
			}
		}
	})

	// Create default config
	cfg := quant.DefaultConfig()

	if *modelName != "" {
		cfg.ModelName = *modelName
	}
	if *inputUMSH != "" {
		cfg.InputUMSH = *inputUMSH
	}
	if *outputUMSH != "" {
		cfg.OutputUMSH = *outputUMSH
	}

	// Set default paths if not specified
	if cfg.InputUMSH == "" {
		cfg.InputUMSH = "model.fp16.umsh"
	}
	if cfg.OutputUMSH == "" {
		cfg.OutputUMSH = "model.int8_sparse.umsh"
	}

	// Save config
	if err := quant.SaveConfig(*outPath, cfg); err != nil {
		return fmt.Errorf("save config: %w", err)
	}

	fmt.Printf("Generated compression config: %s\n", *outPath)
	fmt.Println("\nEdit the config to customize:")
	fmt.Println("  - Quantization rules (which tensors to quantize)")
	fmt.Println("  - Sparsity rules (which tensors to prune)")
	fmt.Println("  - Input/output paths")

	return nil
}
