package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/phenomenon0/Agent-GO/pkg/llm"
	"github.com/phenomenon0/Agent-GO/pkg/llm/inference"
	"github.com/phenomenon0/Agent-GO/pkg/llm/inference/backend"
	"github.com/phenomenon0/Agent-GO/pkg/llm/inference/layers"
	"github.com/phenomenon0/Agent-GO/pkg/quant"
)

func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	umshPath := fs.String("umsh", "", "Path to UMSH/MoSH model file (.umsh or .mosh)")
	fs.String("m", "", "Model path (shorthand)")
	fs.String("mosh", "", "Path to MoSH model file (alias for -umsh)")
	prompt := fs.String("prompt", "", "Input prompt for generation")
	fs.String("p", "", "Prompt (shorthand)")
	maxTokens := fs.Int("max-tokens", 100, "Maximum tokens to generate")
	temperature := fs.Float64("temperature", 0.7, "Sampling temperature")
	interactive := fs.Bool("interactive", false, "Interactive chat mode")
	fs.Bool("i", false, "Interactive (shorthand)")
	serve := fs.Bool("serve", false, "Start HTTP server mode")
	port := fs.Int("port", 11434, "Server port (for serve mode)")
	legacy := fs.Bool("legacy", false, "Use legacy pkg/quant engine (limited quantization support)")
	backendStr := fs.String("backend", "auto", "Backend: auto, cuda, fastblas, pure-go")
	verbose := fs.Bool("verbose", false, "Show detailed info")
	fs.Bool("v", false, "Verbose (shorthand)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Handle shorthand flags
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "m":
			if *umshPath == "" {
				*umshPath = fs.Lookup("m").Value.String()
			}
		case "mosh":
			if *umshPath == "" {
				*umshPath = fs.Lookup("mosh").Value.String()
			}
		case "p":
			if *prompt == "" {
				*prompt = fs.Lookup("p").Value.String()
			}
		case "i":
			*interactive = true
		case "v":
			*verbose = true
		}
	})

	// Check for positional argument (model path)
	if *umshPath == "" && fs.NArg() > 0 {
		*umshPath = fs.Arg(0)
	}

	if *umshPath == "" {
		return fmt.Errorf("model path required (--umsh, --mosh, -m, or positional argument)")
	}

	// Use legacy engine if requested
	if *legacy {
		return runWithLegacyEngine(*umshPath, *prompt, *maxTokens, float32(*temperature), *interactive, *serve, *port, *verbose)
	}

	// Default: use inference.ModelLoader (supports GGUF quantization)
	return runWithInferenceLoader(*umshPath, *prompt, *maxTokens, *temperature, *interactive, *serve, *port, *verbose, *backendStr)
}

// runWithInferenceLoader uses the pkg/llm/inference loader which supports GGUF Q4_K/Q5_K/Q6_K quantization
func runWithInferenceLoader(umshPath, prompt string, maxTokens int, temperature float64, interactive, serve bool, port int, verbose bool, backendStr string) error {
	startTime := time.Now()

	fmt.Printf("Loading model: %s\n", umshPath)

	// Parse backend choice
	var backendKind backend.BackendKind
	switch strings.ToLower(backendStr) {
	case "cuda":
		backendKind = backend.BackendCUDA
	case "fastblas":
		backendKind = backend.BackendFastBLAS
	case "pure-go", "purego":
		backendKind = backend.BackendPureGo
	default:
		backendKind = backend.BackendAuto
	}

	// Create model loader
	loader, err := inference.NewModelLoader(umshPath)
	if err != nil {
		return fmt.Errorf("create loader: %w", err)
	}
	defer loader.Close()

	// Set backend preference
	loader.SetBackend(backendKind)

	// Detect configuration from embedded metadata
	config, err := loader.DetectConfig()
	if err != nil {
		return fmt.Errorf("detect config: %w", err)
	}

	if verbose {
		fmt.Printf("\nModel Configuration:\n")
		fmt.Printf("  Architecture:    %s\n", config.ModelType)
		fmt.Printf("  Vocab Size:      %d\n", config.VocabSize)
		fmt.Printf("  Hidden Size:     %d\n", config.HiddenSize)
		fmt.Printf("  FFN Dim:         %d\n", config.FFNDim)
		fmt.Printf("  Num Layers:      %d\n", config.NumLayers)
		fmt.Printf("  Num Heads:       %d (KV: %d)\n", config.NumHeads, config.NumKVHeads)
		fmt.Printf("  Head Dim:        %d\n", config.HeadDim)
		fmt.Printf("  Max Seq Len:     %d\n", config.MaxSeqLen)
		fmt.Printf("  RoPE Theta:      %.0f\n", config.RopeTheta)
		logMemoryUsage("After config detection")
	}

	// Load model (this loads quantized weights - memory efficient)
	fmt.Printf("Loading weights (Q4_K quantized)...\n")
	model, err := loader.LoadModel()
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	loadTime := time.Since(startTime)
	if verbose {
		fmt.Printf("  Load time: %v\n", loadTime)
		logMemoryUsage("After model load")
	}

	// Load tokenizer
	tokenizer, err := loadTokenizer(umshPath, verbose)
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}

	fmt.Printf("\nModel ready! (loaded in %v)\n", loadTime)

	// Choose mode
	if serve {
		return runInferenceServer(model, tokenizer, port, verbose)
	}

	if interactive {
		return runInferenceInteractive(model, tokenizer, maxTokens, temperature, verbose)
	}

	if prompt != "" {
		return runInferencePrompt(model, tokenizer, prompt, maxTokens, temperature, verbose)
	}

	// Show model info if no action specified
	fmt.Println(loader.ModelInfo())
	fmt.Println("\nUsage:")
	fmt.Println("  ucodec run -m model.umsh -p \"Hello, how are you?\"")
	fmt.Println("  ucodec run -m model.umsh --interactive")
	fmt.Println("  ucodec run -m model.umsh --serve --port 11434")

	return nil
}

// loadTokenizer loads the HuggingFace tokenizer for the model
func loadTokenizer(umshPath string, verbose bool) (llm.Tokenizer, error) {
	// Look for tokenizer.json in various locations
	umshDir := filepath.Dir(umshPath)
	baseName := strings.TrimSuffix(filepath.Base(umshPath), filepath.Ext(umshPath))

	// Handle double extensions like .umsh.mosh
	if strings.HasSuffix(baseName, ".umsh") {
		baseName = strings.TrimSuffix(baseName, ".umsh")
	}

	tokenizerPaths := []string{
		filepath.Join(umshDir, baseName+".tokenizer.json"),                       // models/qwen2.5-7b.tokenizer.json
		filepath.Join(umshDir, "tokenizer.json"),                                 // models/tokenizer.json
		strings.TrimSuffix(umshPath, filepath.Ext(umshPath)) + ".tokenizer.json", // same as first
	}

	for _, tokPath := range tokenizerPaths {
		if _, err := os.Stat(tokPath); err == nil {
			if verbose {
				fmt.Printf("  Loading tokenizer: %s\n", tokPath)
			}
			tokenizer, err := llm.LoadHFTokenizer(tokPath)
			if err != nil {
				if verbose {
					fmt.Printf("  Warning: failed to load %s: %v\n", tokPath, err)
				}
				continue
			}
			if verbose {
				fmt.Printf("  Tokenizer loaded (vocab=%d)\n", tokenizer.VocabSize())
			}
			return tokenizer, nil
		}
	}

	// Fallback to char tokenizer
	if verbose {
		fmt.Println("  Warning: No tokenizer.json found, using CharTokenizer (outputs may be incorrect)")
		fmt.Println("  Download tokenizer: curl -L 'https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer.json' -o models/qwen2.5-7b.tokenizer.json")
	}
	return llm.NewCharTokenizer(), nil
}

// runInferencePrompt runs a single prompt through the model
func runInferencePrompt(model *inference.LLaMAModel, tokenizer llm.Tokenizer, prompt string, maxTokens int, temperature float64, verbose bool) error {
	fmt.Printf("\nPrompt: %s\n", prompt)
	fmt.Printf("Generating up to %d tokens (temperature=%.2f)...\n", maxTokens, temperature)

	// Encode prompt
	tokens32, err := tokenizer.Encode(prompt)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}

	// Convert []int32 to []int
	tokens := make([]int, len(tokens32))
	for i, t := range tokens32 {
		tokens[i] = int(t)
	}

	if verbose {
		fmt.Printf("  Prompt tokens: %d\n", len(tokens))
	}

	// Generate
	genConfig := &inference.GenerationConfig{
		MaxTokens:   maxTokens,
		Temperature: float32(temperature),
		TopK:        40,
		TopP:        0.9,
	}

	startGen := time.Now()
	output, err := model.Generate(tokens, genConfig)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}
	genTime := time.Since(startGen)

	// Extract generated tokens (excluding prompt)
	genTokens := output[len(tokens):]

	if verbose {
		fmt.Printf("  Generated tokens: %d\n", len(genTokens))
		fmt.Printf("  Generation time: %v\n", genTime)
		if len(genTokens) > 0 {
			tokPerSec := float64(len(genTokens)) / genTime.Seconds()
			fmt.Printf("  Throughput: %.2f tok/s\n", tokPerSec)
		}
		logMemoryUsage("After generation")

		// Show dequantization cache stats
		hits, misses, totalTimeMs := layers.DequantStats()
		if hits > 0 || misses > 0 {
			fmt.Printf("  [Dequant Cache] Hits: %d, Misses: %d, Total dequant time: %.1fms\n", hits, misses, totalTimeMs)
		}
	}

	// Convert []int to []int32 for decoder
	genTokens32 := make([]int32, len(genTokens))
	for i, t := range genTokens {
		genTokens32[i] = int32(t)
	}

	// Decode
	text, err := tokenizer.Decode(genTokens32)
	if err != nil {
		return fmt.Errorf("decode: %w", err)
	}

	fmt.Printf("\nGenerated:\n%s\n", text)
	return nil
}

// runInferenceInteractive runs an interactive chat session
func runInferenceInteractive(model *inference.LLaMAModel, tokenizer llm.Tokenizer, maxTokens int, temperature float64, verbose bool) error {
	fmt.Println("\nInteractive mode. Type 'quit' to exit.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	genConfig := &inference.GenerationConfig{
		MaxTokens:   maxTokens,
		Temperature: float32(temperature),
		TopK:        40,
		TopP:        0.9,
	}

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "quit" || input == "exit" {
			break
		}

		// Encode
		tokens32, err := tokenizer.Encode(input)
		if err != nil {
			fmt.Printf("Error encoding: %v\n", err)
			continue
		}

		tokens := make([]int, len(tokens32))
		for i, t := range tokens32 {
			tokens[i] = int(t)
		}

		// Generate
		startGen := time.Now()
		output, err := model.Generate(tokens, genConfig)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		genTime := time.Since(startGen)

		// Decode generated part
		genTokens := output[len(tokens):]
		genTokens32 := make([]int32, len(genTokens))
		for i, t := range genTokens {
			genTokens32[i] = int32(t)
		}

		text, err := tokenizer.Decode(genTokens32)
		if err != nil {
			fmt.Printf("Error decoding: %v\n", err)
			continue
		}

		if verbose {
			tokPerSec := float64(len(genTokens)) / genTime.Seconds()
			fmt.Printf("\n%s\n\n[%d tokens, %.1f tok/s]\n", text, len(genTokens), tokPerSec)
		} else {
			fmt.Printf("\n%s\n\n", text)
		}
	}

	return nil
}

// runInferenceServer starts an HTTP server for inference
func runInferenceServer(model *inference.LLaMAModel, tokenizer llm.Tokenizer, port int, verbose bool) error {
	fmt.Printf("Starting server on port %d\n", port)
	fmt.Printf("  POST /api/generate - Generate text\n")
	fmt.Printf("  GET  /api/health   - Health check\n\n")

	http.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Prompt      string  `json:"prompt"`
			MaxTokens   int     `json:"max_tokens"`
			Temperature float64 `json:"temperature"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if req.MaxTokens <= 0 {
			req.MaxTokens = 100
		}
		if req.Temperature <= 0 {
			req.Temperature = 0.7
		}

		// Encode
		tokens32, err := tokenizer.Encode(req.Prompt)
		if err != nil {
			http.Error(w, fmt.Sprintf("encode error: %v", err), http.StatusBadRequest)
			return
		}

		tokens := make([]int, len(tokens32))
		for i, t := range tokens32 {
			tokens[i] = int(t)
		}

		// Generate
		genConfig := &inference.GenerationConfig{
			MaxTokens:   req.MaxTokens,
			Temperature: float32(req.Temperature),
			TopK:        40,
			TopP:        0.9,
		}

		startGen := time.Now()
		output, err := model.Generate(tokens, genConfig)
		if err != nil {
			http.Error(w, fmt.Sprintf("generate error: %v", err), http.StatusInternalServerError)
			return
		}
		genTime := time.Since(startGen)

		// Decode
		genTokens := output[len(tokens):]
		genTokens32 := make([]int32, len(genTokens))
		for i, t := range genTokens {
			genTokens32[i] = int32(t)
		}

		text, err := tokenizer.Decode(genTokens32)
		if err != nil {
			http.Error(w, fmt.Sprintf("decode error: %v", err), http.StatusInternalServerError)
			return
		}

		resp := struct {
			Response     string  `json:"response"`
			Tokens       int     `json:"tokens"`
			PromptTokens int     `json:"prompt_tokens"`
			GenTimeMs    float64 `json:"gen_time_ms"`
			TokPerSec    float64 `json:"tok_per_sec"`
		}{
			Response:     text,
			Tokens:       len(genTokens),
			PromptTokens: len(tokens),
			GenTimeMs:    float64(genTime.Milliseconds()),
			TokPerSec:    float64(len(genTokens)) / genTime.Seconds(),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	http.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	return http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
}

// logMemoryUsage logs current memory statistics
func logMemoryUsage(label string) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("  [Memory %s] Alloc: %d MB, Sys: %d MB, HeapInuse: %d MB\n",
		label,
		m.Alloc/1024/1024,
		m.Sys/1024/1024,
		m.HeapInuse/1024/1024,
	)
}

// =============================================================================
// Legacy engine support (--legacy flag)
// =============================================================================

func runWithLegacyEngine(umshPath, prompt string, maxTokens int, temperature float32, interactive, serve bool, port int, verbose bool) error {
	fmt.Printf("Loading model (legacy engine): %s\n", umshPath)
	fmt.Println("Note: Legacy engine has limited quantization support. Consider using default engine for GGUF models.")

	engine, err := quant.NewInferenceEngine(umshPath, nil)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer engine.Close()

	if verbose {
		fmt.Println(engine.ModelInfo())
	}

	if serve {
		return runLegacyServer(engine, port)
	}

	if interactive {
		return runLegacyInteractive(engine, maxTokens, temperature)
	}

	if prompt != "" {
		return runLegacySinglePrompt(engine, prompt, maxTokens, temperature)
	}

	fmt.Println(engine.ModelInfo())
	fmt.Println("\nUsage:")
	fmt.Println("  ucodec run --legacy -m model.umsh -p \"Hello\"")
	fmt.Println("  ucodec run --legacy -m model.umsh --interactive")

	return nil
}

func runLegacySinglePrompt(engine *quant.InferenceEngine, prompt string, maxTokens int, temperature float32) error {
	fmt.Printf("\nPrompt: %s\n", prompt)
	fmt.Printf("Generating up to %d tokens (temperature=%.2f)...\n\n", maxTokens, temperature)

	tokens := simpleTokenize(prompt)
	output, err := engine.Generate(tokens, maxTokens, temperature)
	if err != nil {
		return err
	}

	text := simpleDetokenize(output[len(tokens):])
	fmt.Printf("Generated: %s\n", text)
	return nil
}

func runLegacyInteractive(engine *quant.InferenceEngine, maxTokens int, temperature float32) error {
	fmt.Println("\nInteractive mode (legacy). Type 'quit' to exit.")
	fmt.Println("Note: Using placeholder tokenizer")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "quit" || input == "exit" {
			break
		}

		tokens := simpleTokenize(input)
		output, err := engine.Generate(tokens, maxTokens, temperature)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		text := simpleDetokenize(output[len(tokens):])
		fmt.Printf("\n%s\n\n", text)
	}

	return nil
}

func runLegacyServer(engine *quant.InferenceEngine, port int) error {
	fmt.Printf("Starting legacy server on port %d\n", port)

	http.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Prompt      string  `json:"prompt"`
			MaxTokens   int     `json:"max_tokens"`
			Temperature float32 `json:"temperature"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if req.MaxTokens <= 0 {
			req.MaxTokens = 100
		}
		if req.Temperature <= 0 {
			req.Temperature = 0.7
		}

		tokens := simpleTokenize(req.Prompt)
		output, err := engine.Generate(tokens, req.MaxTokens, req.Temperature)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		text := simpleDetokenize(output[len(tokens):])

		resp := struct {
			Response string `json:"response"`
			Tokens   int    `json:"tokens"`
		}{
			Response: text,
			Tokens:   len(output) - len(tokens),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	http.HandleFunc("/api/info", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, engine.ModelInfo())
	})

	return http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
}

// Placeholder tokenizer for legacy engine
func simpleTokenize(text string) []int {
	tokens := make([]int, len(text))
	for i, c := range text {
		tokens[i] = int(c) % 32000
	}
	return tokens
}

func simpleDetokenize(tokens []int) string {
	chars := make([]rune, len(tokens))
	for i, t := range tokens {
		if t >= 32 && t < 127 {
			chars[i] = rune(t)
		} else {
			chars[i] = '?'
		}
	}
	return string(chars)
}
