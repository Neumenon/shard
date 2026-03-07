package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/phenomenon0/Agent-GO/pkg/quant"
	"github.com/phenomenon0/Agent-GO/cowrie/ucodec"
)

func runQuantize(args []string) error {
	fs := flag.NewFlagSet("quantize", flag.ExitOnError)
	configPath := fs.String("config", "", "Path to compression config YAML")
	fs.String("c", "", "Config path (shorthand)")
	inputPath := fs.String("in", "", "Input UMSH file (overrides config)")
	fs.String("i", "", "Input path (shorthand)")
	outputPath := fs.String("out", "", "Output UMSH file (overrides config)")
	fs.String("o", "", "Output path (shorthand)")
	scheme := fs.String("scheme", "", "Quantization scheme (overrides config)")
	verbose := fs.Bool("verbose", false, "Show detailed progress")
	fs.Bool("v", false, "Verbose (shorthand)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Handle shorthand flags
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "c":
			if *configPath == "" {
				*configPath = fs.Lookup("c").Value.String()
			}
		case "i":
			if *inputPath == "" {
				*inputPath = fs.Lookup("i").Value.String()
			}
		case "o":
			if *outputPath == "" {
				*outputPath = fs.Lookup("o").Value.String()
			}
		case "v":
			*verbose = true
		}
	})

	// Load config if provided
	var cfg *quant.CompressionConfig
	if *configPath != "" {
		var err error
		cfg, err = quant.LoadConfig(*configPath)
		if err != nil {
			return fmt.Errorf("load config: %w", err)
		}
	} else {
		// Use defaults
		cfg = quant.DefaultConfig()
	}

	// Override with command-line options
	if *inputPath != "" {
		cfg.InputUMSH = *inputPath
	}
	if *outputPath != "" {
		cfg.OutputUMSH = *outputPath
	}
	if *scheme != "" {
		cfg.Quantization.Scheme = *scheme
	}

	// Validate paths
	if cfg.InputUMSH == "" {
		return fmt.Errorf("input UMSH path required (--in or in config)")
	}
	if cfg.OutputUMSH == "" {
		return fmt.Errorf("output UMSH path required (--out or in config)")
	}

	// Run quantization
	return quantizeUMSH(cfg, *verbose)
}

func quantizeUMSH(cfg *quant.CompressionConfig, verbose bool) error {
	start := time.Now()

	// Open input UMSH
	reader, err := ucodec.OpenUMSH(cfg.InputUMSH)
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer reader.Close()

	// Create output UMSH
	writer, err := ucodec.NewUMSHWriter(cfg.OutputUMSH)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}

	tensorNames := reader.TensorNames()
	fmt.Printf("Quantizing %d tensors from %s\n", len(tensorNames), cfg.InputUMSH)

	var stats QuantStats

	// Process each tensor
	for i, name := range tensorNames {
		tensor, err := reader.GetTensor(name)
		if err != nil {
			return fmt.Errorf("read tensor %q: %w", name, err)
		}

		// Check if tensor matches a quantization rule
		rule := cfg.FindQuantRule(name)
		shouldQuantize := cfg.Quantization.Enabled && rule != nil && !rule.Skip

		if verbose {
			fmt.Printf("[%d/%d] %s ", i+1, len(tensorNames), name)
		}

		if shouldQuantize {
			// Quantize this tensor
			qt, err := quantizeTensor(tensor, rule, cfg.Quantization.Scheme)
			if err != nil {
				if verbose {
					fmt.Printf("SKIP (error: %v)\n", err)
				}
				// Write original tensor on error
				if err := writer.AddTensor(name, tensor); err != nil {
					return fmt.Errorf("write tensor %q: %w", name, err)
				}
				stats.Skipped++
				continue
			}

			// Write quantized tensor
			if err := writer.AddTensor(name, qt.Quantized); err != nil {
				return fmt.Errorf("write quantized tensor %q: %w", name, err)
			}

			// Write scale tensor with ".scale" suffix
			scaleName := name + ".scale"
			if err := writer.AddTensor(scaleName, qt.Scales); err != nil {
				return fmt.Errorf("write scale tensor %q: %w", scaleName, err)
			}

			if verbose {
				ratio := float64(tensor.ByteSize()) / float64(qt.Quantized.ByteSize())
				fmt.Printf("QUANTIZED (%.1fx compression)\n", ratio)
			}

			stats.Quantized++
			stats.OriginalBytes += tensor.ByteSize()
			stats.QuantizedBytes += qt.Quantized.ByteSize() + qt.Scales.ByteSize()
		} else {
			// Copy tensor unchanged
			if err := writer.AddTensor(name, tensor); err != nil {
				return fmt.Errorf("write tensor %q: %w", name, err)
			}

			if verbose {
				skipReason := "no matching rule"
				if rule != nil && rule.Skip {
					skipReason = "marked skip"
				}
				fmt.Printf("COPY (%s)\n", skipReason)
			}

			stats.Copied++
			stats.OriginalBytes += tensor.ByteSize()
			stats.QuantizedBytes += tensor.ByteSize()
		}
	}

	// Close writer
	if err := writer.Close(); err != nil {
		return fmt.Errorf("close output: %w", err)
	}

	elapsed := time.Since(start)

	// Print summary
	fmt.Println()
	fmt.Printf("Quantization complete in %v\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  Tensors quantized: %d\n", stats.Quantized)
	fmt.Printf("  Tensors copied:    %d\n", stats.Copied)
	fmt.Printf("  Tensors skipped:   %d\n", stats.Skipped)

	if stats.OriginalBytes > 0 {
		ratio := float64(stats.OriginalBytes) / float64(stats.QuantizedBytes)
		fmt.Printf("  Compression:       %.2fx (%s -> %s)\n",
			ratio, humanSize(stats.OriginalBytes), humanSize(stats.QuantizedBytes))
	}

	fmt.Printf("  Output: %s\n", cfg.OutputUMSH)

	return nil
}

func quantizeTensor(tensor *ucodec.TensorV1, rule *quant.QuantRule, scheme string) (*quant.QuantizedTensor, error) {
	// Determine scheme to use
	targetScheme := scheme
	if targetScheme == "" {
		targetScheme = quant.SchemePerChannelSymmetricInt8
	}

	// Only quantize float tensors
	dtype := ucodec.DTypeName(tensor.DType)
	if !strings.HasPrefix(dtype, "float") && !strings.HasPrefix(dtype, "bfloat") {
		return nil, fmt.Errorf("cannot quantize non-float tensor (dtype=%s)", dtype)
	}

	// Apply quantization based on scheme
	switch targetScheme {
	case quant.SchemePerChannelSymmetricInt8:
		return quant.QuantizeInt8PerChannel(tensor)
	case quant.SchemePerTensorSymmetricInt8:
		return quant.QuantizeInt8PerTensor(tensor)
	default:
		return nil, fmt.Errorf("unsupported scheme: %s", targetScheme)
	}
}

// QuantStats tracks quantization progress.
type QuantStats struct {
	Quantized      int
	Copied         int
	Skipped        int
	OriginalBytes  int64
	QuantizedBytes int64
}

// getFileSize returns file size or 0 on error.
func getFileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}
