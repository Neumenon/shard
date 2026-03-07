package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/phenomenon0/Agent-GO/pkg/quant"
	"github.com/phenomenon0/Agent-GO/cowrie/ucodec"
)

func runPrune(args []string) error {
	fs := flag.NewFlagSet("prune", flag.ExitOnError)
	configPath := fs.String("config", "", "Path to compression config YAML")
	fs.String("c", "", "Config path (shorthand)")
	inputPath := fs.String("in", "", "Input UMSH file (overrides config)")
	fs.String("i", "", "Input path (shorthand)")
	outputPath := fs.String("out", "", "Output UMSH file (overrides config)")
	fs.String("o", "", "Output path (shorthand)")
	pattern := fs.String("pattern", "", "Sparsity pattern (overrides config, e.g. 2:4)")
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
	if *pattern != "" {
		cfg.Sparsity.Pattern = *pattern
	}

	// Validate paths
	if cfg.InputUMSH == "" {
		return fmt.Errorf("input UMSH path required (--in or in config)")
	}
	if cfg.OutputUMSH == "" {
		return fmt.Errorf("output UMSH path required (--out or in config)")
	}

	// Run pruning
	return pruneUMSH(cfg, *verbose)
}

func pruneUMSH(cfg *quant.CompressionConfig, verbose bool) error {
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
	fmt.Printf("Pruning %d tensors from %s (pattern: %s)\n",
		len(tensorNames), cfg.InputUMSH, cfg.Sparsity.Pattern)

	var stats PruneStats

	// Process each tensor
	for i, name := range tensorNames {
		tensor, err := reader.GetTensor(name)
		if err != nil {
			return fmt.Errorf("read tensor %q: %w", name, err)
		}

		// Check if tensor matches a sparsity rule
		rule := cfg.FindSparsityRule(name)
		shouldPrune := cfg.Sparsity.Enabled && rule != nil && !rule.Skip

		if verbose {
			fmt.Printf("[%d/%d] %s ", i+1, len(tensorNames), name)
		}

		if shouldPrune {
			// Apply pruning
			pt, err := pruneTensor(tensor, cfg.Sparsity.Pattern)
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

			// Write pruned tensor
			if err := writer.AddTensor(name, pt.Tensor); err != nil {
				return fmt.Errorf("write pruned tensor %q: %w", name, err)
			}

			if verbose {
				fmt.Printf("PRUNED (%.1f%% sparse, avg kept=%.3f, avg pruned=%.3f)\n",
					pt.Stats.ActualSparsity*100, pt.Stats.AvgMagnitude, pt.Stats.AvgPrunedMag)
			}

			stats.Pruned++
			stats.TotalElements += pt.Stats.TotalElements
			stats.PrunedElements += pt.Stats.PrunedElements
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
		}
	}

	// Close writer
	if err := writer.Close(); err != nil {
		return fmt.Errorf("close output: %w", err)
	}

	elapsed := time.Since(start)

	// Print summary
	fmt.Println()
	fmt.Printf("Pruning complete in %v\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  Tensors pruned:  %d\n", stats.Pruned)
	fmt.Printf("  Tensors copied:  %d\n", stats.Copied)
	fmt.Printf("  Tensors skipped: %d\n", stats.Skipped)

	if stats.TotalElements > 0 {
		sparsity := float64(stats.PrunedElements) / float64(stats.TotalElements)
		fmt.Printf("  Overall sparsity: %.1f%% (%d/%d elements pruned)\n",
			sparsity*100, stats.PrunedElements, stats.TotalElements)
	}

	fmt.Printf("  Output: %s\n", cfg.OutputUMSH)

	return nil
}

func pruneTensor(tensor *ucodec.TensorV1, pattern string) (*quant.PrunedTensor, error) {
	// Select pruning function based on pattern
	switch pattern {
	case quant.SparsityPattern2x4, "":
		return quant.Prune2x4(tensor)
	case quant.SparsityPattern1x4:
		return quant.Prune1x4(tensor)
	default:
		return nil, fmt.Errorf("unsupported sparsity pattern: %s", pattern)
	}
}

// PruneStats tracks pruning progress.
type PruneStats struct {
	Pruned         int
	Copied         int
	Skipped        int
	TotalElements  int64
	PrunedElements int64
}
