// Package server implements the gRPC ScannerService for WRAITH.
//
// This is the bridge between the Python AI agent (gRPC client) and the
// Go attack engine. Each RPC method translates proto messages into
// engine calls and returns the results.
package server

import (
	"context"
	"io"
	"runtime"
	"time"

	"go.uber.org/zap"

	"github.com/VectorCipher/Wraith/scanner/internal/engine"
	"github.com/VectorCipher/Wraith/scanner/internal/models"
	"github.com/VectorCipher/Wraith/scanner/internal/utils"
	pb "github.com/VectorCipher/Wraith/scanner/proto"
)

// ScannerServer implements the pb.ScannerServiceServer interface.
type ScannerServer struct {
	pb.UnimplementedScannerServiceServer

	engine       *engine.Engine
	crawler      *engine.Crawler
	fingerprinter *engine.Fingerprinter
	logger       *zap.Logger
	config       *models.Config
	startTime    time.Time
}

// NewScannerServer creates a new gRPC scanner server.
func NewScannerServer(cfg *models.Config, logger *zap.Logger) *ScannerServer {
	eng := engine.New(cfg, logger)

	httpClient := utils.NewHTTPClient(cfg.RequestTimeoutSeconds, true)
	crawl := engine.NewCrawler(httpClient, logger)
	fp := engine.NewFingerprinter(httpClient, logger)

	return &ScannerServer{
		engine:       eng,
		crawler:      crawl,
		fingerprinter: fp,
		logger:       logger.Named("grpc"),
		config:       cfg,
		startTime:    time.Now(),
	}
}

// =========================================================================
// Health & Status
// =========================================================================

// HealthCheck verifies the scanner is alive and responsive.
func (s *ScannerServer) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	stats := s.engine.GetStats()

	return &pb.HealthCheckResponse{
		Healthy:       true,
		Version:       models.Version,
		Pong:          req.GetPing(),
		UptimeSeconds: int64(time.Since(s.startTime).Seconds()),
		ActiveAttacks: stats.ActiveAttacks,
	}, nil
}

// GetStatus returns detailed scanner status and resource usage.
func (s *ScannerServer) GetStatus(ctx context.Context, req *pb.StatusRequest) (*pb.StatusResponse, error) {
	stats := s.engine.GetStats()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	return &pb.StatusResponse{
		Ready:                  true,
		Version:                models.Version,
		UptimeSeconds:          int64(time.Since(s.startTime).Seconds()),
		ActiveGoroutines:       int32(runtime.NumGoroutine()),
		MemoryUsedBytes:        int64(memStats.Alloc),
		ActiveAttacks:          stats.ActiveAttacks,
		TotalRequestsSent:      stats.TotalRequestsSent,
		TotalAttacksCompleted:  stats.TotalAttacksCompleted,
		MaxConcurrentRequests:  int32(s.config.MaxConcurrentRequests),
		RateLimitPerSecond:     int32(s.config.RateLimitPerSecond),
		RequestTimeoutSeconds:  int32(s.config.RequestTimeoutSeconds),
	}, nil
}

// =========================================================================
// Attack Execution
// =========================================================================

// ExecuteAttack runs a complete attack and returns the aggregated result.
func (s *ScannerServer) ExecuteAttack(ctx context.Context, req *pb.AttackRequest) (*pb.AttackResult, error) {
	s.logger.Info("ExecuteAttack RPC",
		zap.String("attack_id", req.GetAttackId()),
		zap.String("type", req.GetAttackType()),
		zap.Int("payloads", len(req.GetPayloads())),
	)

	return s.engine.ExecuteAttack(ctx, req)
}

// ExecuteAttackStream runs an attack and streams each PayloadResult back
// as it completes. The Python client receives results in real-time.
func (s *ScannerServer) ExecuteAttackStream(req *pb.AttackRequest, stream pb.ScannerService_ExecuteAttackStreamServer) error {
	s.logger.Info("ExecuteAttackStream RPC",
		zap.String("attack_id", req.GetAttackId()),
		zap.String("type", req.GetAttackType()),
		zap.Int("payloads", len(req.GetPayloads())),
	)

	resultCh := make(chan *pb.PayloadResult, 100)

	// Run attack in background, send results as they arrive
	errCh := make(chan error, 1)
	go func() {
		defer close(resultCh)
		errCh <- s.engine.ExecuteAttackStream(stream.Context(), req, resultCh)
	}()

	// Stream results to client
	for result := range resultCh {
		if err := stream.Send(result); err != nil {
			s.logger.Error("Failed to stream result", zap.Error(err))
			return err
		}
	}

	return <-errCh
}

// ExecuteAttackBatch receives multiple AttackRequests via client stream,
// executes them all, and returns a combined BatchAttackResult.
func (s *ScannerServer) ExecuteAttackBatch(stream pb.ScannerService_ExecuteAttackBatchServer) error {
	s.logger.Info("ExecuteAttackBatch RPC")

	var results []*pb.AttackResult
	var totalDuration float64
	successful := 0
	failed := 0

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			s.logger.Error("Batch receive error", zap.Error(err))
			return err
		}

		result, execErr := s.engine.ExecuteAttack(stream.Context(), req)
		if execErr != nil {
			failed++
			results = append(results, &pb.AttackResult{
				AttackId:   req.GetAttackId(),
				AttackType: req.GetAttackType(),
				Status:     "error",
				Error:      execErr.Error(),
			})
			continue
		}

		results = append(results, result)
		totalDuration += result.GetDurationMs()

		if result.GetStatus() == "success" {
			successful++
		} else {
			failed++
		}
	}

	return stream.SendAndClose(&pb.BatchAttackResult{
		Results:           results,
		TotalAttacks:      int32(len(results)),
		SuccessfulAttacks: int32(successful),
		FailedAttacks:     int32(failed),
		TotalDurationMs:   totalDuration,
	})
}

// =========================================================================
// Reconnaissance
// =========================================================================

// CrawlTarget crawls the target website and streams discovered pages.
func (s *ScannerServer) CrawlTarget(req *pb.CrawlRequest, stream pb.ScannerService_CrawlTargetServer) error {
	s.logger.Info("CrawlTarget RPC",
		zap.String("url", req.GetTargetUrl()),
		zap.Int32("max_depth", req.GetMaxDepth()),
		zap.Int32("max_pages", req.GetMaxPages()),
	)

	resultCh := make(chan *pb.CrawlResult, 100)

	errCh := make(chan error, 1)
	go func() {
		defer close(resultCh)
		errCh <- s.crawler.Crawl(stream.Context(), req, resultCh)
	}()

	for result := range resultCh {
		if err := stream.Send(result); err != nil {
			s.logger.Error("Failed to stream crawl result", zap.Error(err))
			return err
		}
	}

	return <-errCh
}

// FingerprintTarget identifies the target's technology stack.
func (s *ScannerServer) FingerprintTarget(ctx context.Context, req *pb.FingerprintRequest) (*pb.FingerprintResult, error) {
	s.logger.Info("FingerprintTarget RPC", zap.String("url", req.GetTargetUrl()))
	return s.fingerprinter.Fingerprint(ctx, req)
}

// SendBaselineRequest sends a normal request for response comparison.
func (s *ScannerServer) SendBaselineRequest(ctx context.Context, req *pb.BaselineRequest) (*pb.BaselineResponse, error) {
	s.logger.Info("SendBaselineRequest RPC", zap.String("url", req.GetUrl()))
	return s.engine.SendBaseline(ctx, req)
}

// =========================================================================
// Utility
// =========================================================================

// SendRawRequest sends a single HTTP request through the scanner.
func (s *ScannerServer) SendRawRequest(ctx context.Context, req *pb.RawHTTPRequest) (*pb.RawHTTPResponse, error) {
	s.logger.Info("SendRawRequest RPC",
		zap.String("method", req.GetMethod()),
		zap.String("url", req.GetUrl()),
	)
	return s.engine.SendRawRequest(ctx, req)
}

// AbortAll cancels all running attacks immediately.
func (s *ScannerServer) AbortAll(ctx context.Context, req *pb.AbortRequest) (*pb.AbortResponse, error) {
	s.logger.Warn("AbortAll RPC",
		zap.String("reason", req.GetReason()),
		zap.Bool("force", req.GetForce()),
	)

	count := s.engine.AbortAll()

	return &pb.AbortResponse{
		Success:           true,
		AttacksAborted:    int32(count),
		RequestsCancelled: 0, // In-flight requests cancel via context
		Message:           "All attacks aborted",
	}, nil
}
