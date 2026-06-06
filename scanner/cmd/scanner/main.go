// WRAITH Scanner — High-Speed gRPC Attack Engine
//
// This is the entrypoint for the Go scanner service. It starts a gRPC
// server that listens for attack requests from the Python AI agent.


package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"
	"go.uber.org/zap"//very fast production logger
	"google.golang.org/grpc"//grpc framework for AI-scanner configuration
	//Internal Projects Import 
	"github.com/VectorCipher/Wraith/scanner/internal/models"
	"github.com/VectorCipher/Wraith/scanner/internal/server"
	"github.com/VectorCipher/Wraith/scanner/internal/utils"
	pb "github.com/VectorCipher/Wraith/scanner/proto"
)

func main() {
	// -----------------------------------------------------------------
	// Parse CLI flags
	// -----------------------------------------------------------------
	cfg := models.DefaultConfig()

	flag.IntVar(&cfg.Port, "port", cfg.Port, "gRPC listen port")
	flag.IntVar(&cfg.RateLimitPerSecond, "rate-limit", cfg.RateLimitPerSecond, "Max requests per second to target")
	flag.IntVar(&cfg.MaxConcurrentRequests, "concurrency", cfg.MaxConcurrentRequests, "Max concurrent HTTP requests")
	flag.IntVar(&cfg.RequestTimeoutSeconds, "timeout", cfg.RequestTimeoutSeconds, "Per-request timeout in seconds")
	flag.StringVar(&cfg.LogLevel, "log-level", cfg.LogLevel, "Log level: debug, info, warn, error")
	flag.BoolVar(&cfg.LogJSON, "log-json", cfg.LogJSON, "Output logs as structured JSON")
	flag.Parse()

	// -----------------------------------------------------------------
	// Initialize logger
	// -----------------------------------------------------------------
	logger, err := utils.NewLogger(cfg.LogLevel, cfg.LogJSON)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync() //nolint:errcheck

	logger.Info("Starting WRAITH Scanner",
		zap.String("version", models.Version),
		zap.Int("port", cfg.Port),
		zap.Int("rate_limit", cfg.RateLimitPerSecond),
		zap.Int("concurrency", cfg.MaxConcurrentRequests),
		zap.Int("timeout", cfg.RequestTimeoutSeconds),
	)

	// -----------------------------------------------------------------
	// Create gRPC server with interceptors
	// -----------------------------------------------------------------
	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(
			server.RecoveryInterceptor(logger),
			server.LoggingInterceptor(logger),
		),
		grpc.ChainStreamInterceptor(
			server.StreamRecoveryInterceptor(logger),
			server.StreamLoggingInterceptor(logger),
		),
	)

	// Register the scanner service
	scannerServer := server.NewScannerServer(cfg, logger)
	pb.RegisterScannerServiceServer(grpcServer, scannerServer)

	// -----------------------------------------------------------------
	// Start listening
	// -----------------------------------------------------------------
	addr := fmt.Sprintf(":%d", cfg.Port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		logger.Fatal("Failed to listen",
			zap.String("address", addr),
			zap.Error(err),
		)
	}

	logger.Info("WRAITH Scanner ready",
		zap.String("address", addr),
		zap.String("version", models.Version),
	)

	// -----------------------------------------------------------------
	// Graceful shutdown on SIGINT/SIGTERM
	// -----------------------------------------------------------------
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		logger.Info("Shutdown signal received", zap.String("signal", sig.String()))
		grpcServer.GracefulStop()
	}()

	// -----------------------------------------------------------------
	// Serve
	// -----------------------------------------------------------------
	if err := grpcServer.Serve(listener); err != nil {
		logger.Fatal("gRPC server failed", zap.Error(err))
	}

	logger.Info("WRAITH Scanner stopped")
}
