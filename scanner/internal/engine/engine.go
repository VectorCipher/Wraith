// Package engine implements the core attack execution engine for WRAITH.
//
// The engine receives attack requests from the gRPC server, dispatches
// HTTP requests concurrently with rate limiting, and collects results.
// It is designed for high throughput while respecting target rate limits.
package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
	"golang.org/x/time/rate"

	"github.com/VectorCipher/Wraith/blob/main/scanner/internal/models"
	"github.com/VectorCipher/Wraith/blob/main/scanner/internal/utils"
	pb "github.com/VectorCipher/Wraith/blob/main/scanner/proto"
)

// Engine is the core attack execution engine.
//
// It manages HTTP client pooling, rate limiting, bounded concurrency,
// and attack lifecycle (start, execute, abort). All methods are safe
// for concurrent use.
type Engine struct {
	httpClient *http.Client
	limiter    *rate.Limiter
	logger     *zap.Logger
	config     *models.Config

	// Statistics (atomic for thread safety)
	totalRequests atomic.Int64
	totalAttacks  atomic.Int64
	activeAttacks atomic.Int32

	// Abort handling: attack_id → cancel function
	mu          sync.Mutex
	cancelFuncs map[string]context.CancelFunc
}

// New creates a new Engine with the given configuration.
func New(cfg *models.Config, logger *zap.Logger) *Engine {
	client := utils.NewHTTPClient(cfg.RequestTimeoutSeconds, false)
	limiter := rate.NewLimiter(rate.Limit(cfg.RateLimitPerSecond), cfg.RateLimitPerSecond)

	return &Engine{
		httpClient:  client,
		limiter:     limiter,
		logger:      logger.Named("engine"),
		config:      cfg,
		cancelFuncs: make(map[string]context.CancelFunc),
	}
}

// ExecuteAttack runs a full attack: sends all payloads concurrently and
// returns the aggregated result. This is the unary RPC handler.
func (e *Engine) ExecuteAttack(ctx context.Context, req *pb.AttackRequest) (*pb.AttackResult, error) {
	attackID := req.GetAttackId()
	e.logger.Info("Starting attack",
		zap.String("attack_id", attackID),
		zap.String("type", req.GetAttackType()),
		zap.String("target", req.GetTargetUrl()),
		zap.Int("payloads", len(req.GetPayloads())),
	)

	// Register attack for abort support
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	e.registerAttack(attackID, cancel)
	defer e.deregisterAttack(attackID)

	e.activeAttacks.Add(1)
	defer e.activeAttacks.Add(-1)

	start := time.Now()

	// Execute all payloads
	results := e.executePayloads(ctx, req)

	duration := time.Since(start)

	// Build result
	result := e.buildAttackResult(req, results, duration)

	e.totalAttacks.Add(1)
	e.logger.Info("Attack completed",
		zap.String("attack_id", attackID),
		zap.String("status", result.GetStatus()),
		zap.Int("total_requests", int(result.GetTotalRequests())),
		zap.Duration("duration", duration),
	)

	return result, nil
}

// ExecuteAttackStream runs an attack and sends each PayloadResult to the
// provided channel as it completes. Used for the streaming RPC.
func (e *Engine) ExecuteAttackStream(ctx context.Context, req *pb.AttackRequest, resultCh chan<- *pb.PayloadResult) error {
	attackID := req.GetAttackId()

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	e.registerAttack(attackID, cancel)
	defer e.deregisterAttack(attackID)

	e.activeAttacks.Add(1)
	defer e.activeAttacks.Add(-1)

	payloads := req.GetPayloads()
	if len(payloads) == 0 {
		return nil
	}

	maxConcurrent := int(req.GetMaxConcurrent())
	if maxConcurrent <= 0 {
		maxConcurrent = e.config.MaxConcurrentRequests
	}

	sem := make(chan struct{}, maxConcurrent)
	var wg sync.WaitGroup

	payloadLoop:
	for i, payload := range payloads {
		select {
		case <-ctx.Done():
			break payloadLoop
		default:
		}

		wg.Add(1)
		sem <- struct{}{} // Acquire semaphore

		go func(payload string, index int) {
			defer wg.Done()
			defer func() { <-sem }() // Release semaphore

			result := e.sendPayload(ctx, req, payload, index)
			select {
			case resultCh <- result:
			case <-ctx.Done():
			}
		}(payload, i)

		// Optional delay between payloads
		if delay := req.GetDelayBetweenMs(); delay > 0 {
			time.Sleep(time.Duration(delay) * time.Millisecond)
		}
	}

	wg.Wait()
	return ctx.Err()
}

// AbortAttack cancels a running attack by its ID.
// Returns true if the attack was found and cancelled.
func (e *Engine) AbortAttack(attackID string) bool {
	e.mu.Lock()
	cancel, exists := e.cancelFuncs[attackID]
	e.mu.Unlock()

	if exists {
		cancel()
		e.logger.Info("Attack aborted", zap.String("attack_id", attackID))
		return true
	}
	return false
}

// AbortAll cancels all running attacks. Returns the count of aborted attacks.
func (e *Engine) AbortAll() int {
	e.mu.Lock()
	count := len(e.cancelFuncs)
	for id, cancel := range e.cancelFuncs {
		cancel()
		e.logger.Info("Aborting attack", zap.String("attack_id", id))
	}
	e.cancelFuncs = make(map[string]context.CancelFunc)
	e.mu.Unlock()

	return count
}

// SendRawRequest sends a single HTTP request and returns the raw response.
func (e *Engine) SendRawRequest(ctx context.Context, reqProto *pb.RawHTTPRequest) (*pb.RawHTTPResponse, error) {
	httpReq, err := http.NewRequestWithContext(
		ctx,
		reqProto.GetMethod(),
		reqProto.GetUrl(),
		strings.NewReader(reqProto.GetBody()),
	)
	if err != nil {
		return &pb.RawHTTPResponse{Error: fmt.Sprintf("failed to build request: %v", err)}, nil
	}

	utils.ApplyHeaders(httpReq, reqProto.GetHeaders())
	utils.ApplyCookies(httpReq, reqProto.GetCookies())

	// Use a client with redirect behavior matching the request
	client := utils.NewHTTPClient(
		int(reqProto.GetTimeoutSeconds()),
		reqProto.GetFollowRedirects(),
	)

	start := time.Now()
	resp, err := client.Do(httpReq)
	elapsed := time.Since(start)

	if err != nil {
		return &pb.RawHTTPResponse{Error: fmt.Sprintf("request failed: %v", err)}, nil
	}

	body, _, _ := utils.ReadBodyLimited(resp.Body, e.config.MaxResponseBodySize)

	return &pb.RawHTTPResponse{
		StatusCode:     int32(resp.StatusCode),
		Body:           body,
		Headers:        utils.FlattenHeaders(resp.Header),
		ContentLength:  resp.ContentLength,
		ResponseTimeMs: float64(elapsed.Milliseconds()),
		ContentType:    resp.Header.Get("Content-Type"),
		Redirected:     resp.Request.URL.String() != reqProto.GetUrl(),
		FinalUrl:       resp.Request.URL.String(),
	}, nil
}

// SendBaseline sends a baseline request for comparison during attack analysis.
func (e *Engine) SendBaseline(ctx context.Context, req *pb.BaselineRequest) (*pb.BaselineResponse, error) {
	httpReq, err := http.NewRequestWithContext(
		ctx,
		req.GetMethod(),
		req.GetUrl(),
		strings.NewReader(req.GetBody()),
	)
	if err != nil {
		return &pb.BaselineResponse{Error: fmt.Sprintf("failed to build request: %v", err)}, nil
	}

	utils.ApplyHeaders(httpReq, req.GetHeaders())
	utils.ApplyCookies(httpReq, req.GetCookies())

	start := time.Now()
	resp, err := e.httpClient.Do(httpReq)
	elapsed := time.Since(start)

	if err != nil {
		return &pb.BaselineResponse{Error: fmt.Sprintf("request failed: %v", err)}, nil
	}

	body, _, _ := utils.ReadBodyLimited(resp.Body, e.config.MaxResponseBodySize)

	return &pb.BaselineResponse{
		StatusCode:     int32(resp.StatusCode),
		Body:           body,
		Headers:        utils.FlattenHeaders(resp.Header),
		ContentLength:  resp.ContentLength,
		ResponseTimeMs: float64(elapsed.Milliseconds()),
		ContentType:    resp.Header.Get("Content-Type"),
	}, nil
}

// GetStats returns current engine statistics.
func (e *Engine) GetStats() models.Stats {
	return models.Stats{
		TotalRequestsSent:     e.totalRequests.Load(),
		TotalAttacksCompleted: e.totalAttacks.Load(),
		ActiveAttacks:         e.activeAttacks.Load(),
	}
}

// ---------------------------------------------------------------------------
// Internal: Payload Execution
// ---------------------------------------------------------------------------

// executePayloads dispatches all payloads concurrently with rate limiting
// and bounded concurrency, returning the collected results.
func (e *Engine) executePayloads(ctx context.Context, req *pb.AttackRequest) []*pb.PayloadResult {
	payloads := req.GetPayloads()
	results := make([]*pb.PayloadResult, len(payloads))

	maxConcurrent := int(req.GetMaxConcurrent())
	if maxConcurrent <= 0 {
		maxConcurrent = e.config.MaxConcurrentRequests
	}

	sem := make(chan struct{}, maxConcurrent)
	var wg sync.WaitGroup

	payloadLoop:
	for i, payload := range payloads {
		select {
		case <-ctx.Done():
			// Fill remaining with context error
			for j := i; j < len(payloads); j++ {
				results[j] = &pb.PayloadResult{
					Payload:      payloads[j],
					PayloadIndex: int32(j),
					Error:        "attack aborted",
				}
			}
			break payloadLoop
		default:
		}

		wg.Add(1)
		sem <- struct{}{} // Acquire

		go func(payload string, index int) {
			defer wg.Done()
			defer func() { <-sem }() // Release

			results[index] = e.sendPayload(ctx, req, payload, index)
		}(payload, i)

		// Optional inter-payload delay
		if delay := req.GetDelayBetweenMs(); delay > 0 {
			time.Sleep(time.Duration(delay) * time.Millisecond)
		}
	}

	wg.Wait()
	return results
}

// sendPayload sends a single payload and returns the result.
func (e *Engine) sendPayload(ctx context.Context, req *pb.AttackRequest, payload string, index int) *pb.PayloadResult {
	result := &pb.PayloadResult{
		Payload:      payload,
		PayloadIndex: int32(index),
	}

	// Rate limit
	if err := e.limiter.Wait(ctx); err != nil {
		result.Error = fmt.Sprintf("rate limit wait cancelled: %v", err)
		return result
	}

	// Build and inject
	httpReq, err := e.buildRequest(ctx, req, payload)
	if err != nil {
		result.Error = fmt.Sprintf("failed to build request: %v", err)
		return result
	}

	// Send
	start := time.Now()
	resp, err := e.httpClient.Do(httpReq)
	elapsed := time.Since(start)
	e.totalRequests.Add(1)

	if err != nil {
		result.Error = fmt.Sprintf("request failed: %v", err)
		result.ResponseTimeMs = float64(elapsed.Milliseconds())
		return result
	}

	// Read response
	body, truncated, _ := utils.ReadBodyLimited(resp.Body, e.config.MaxResponseBodySize)

	result.StatusCode = int32(resp.StatusCode)
	result.ResponseBody = body
	result.ResponseHeaders = utils.FlattenHeaders(resp.Header)
	result.ContentLength = resp.ContentLength
	result.ContentType = resp.Header.Get("Content-Type")
	result.ResponseTimeMs = float64(elapsed.Milliseconds())
	result.BodyTruncated = truncated
	result.OriginalBodyLength = int64(len(body))
	if truncated {
		result.OriginalBodyLength = resp.ContentLength
	}

	// Check redirect
	if resp.Request.URL.String() != httpReq.URL.String() {
		result.Redirected = true
		result.FinalUrl = resp.Request.URL.String()
	}

	return result
}

// buildRequest creates an HTTP request with the payload injected at the
// configured injection point.
func (e *Engine) buildRequest(ctx context.Context, req *pb.AttackRequest, payload string) (*http.Request, error) {
	targetURL := req.GetTargetUrl()
	method := req.GetMethod()
	if method == "" {
		method = "GET"
	}

	bodyStr := req.GetBody()
	injectionPoint := req.GetInjectionPoint()
	paramName := req.GetParameterName()

	// Inject payload based on injection point
	switch strings.ToLower(injectionPoint) {
	case "query":
		u, err := url.Parse(targetURL)
		if err != nil {
			return nil, fmt.Errorf("invalid URL: %w", err)
		}
		q := u.Query()
		q.Set(paramName, payload)
		u.RawQuery = q.Encode()
		targetURL = u.String()

	case "body":
		bodyStr = strings.ReplaceAll(bodyStr, fmt.Sprintf("{%s}", paramName), payload)
		if bodyStr == req.GetBody() {
			// No placeholder found — set as form data
			bodyStr = fmt.Sprintf("%s=%s", url.QueryEscape(paramName), url.QueryEscape(payload))
		}

	case "json_body":
		var bodyMap map[string]interface{}
		if err := json.Unmarshal([]byte(bodyStr), &bodyMap); err != nil {
			bodyMap = make(map[string]interface{})
		}
		bodyMap[paramName] = payload
		jsonBytes, _ := json.Marshal(bodyMap)
		bodyStr = string(jsonBytes)

	case "header":
		// Handled below when applying headers

	case "cookie":
		// Handled below when applying cookies

	case "path":
		targetURL = strings.ReplaceAll(targetURL, fmt.Sprintf("{%s}", paramName), url.PathEscape(payload))
	}

	// Create the request
	var bodyReader *bytes.Reader
	if bodyStr != "" {
		bodyReader = bytes.NewReader([]byte(bodyStr))
	} else {
		bodyReader = bytes.NewReader(nil)
	}

	httpReq, err := http.NewRequestWithContext(ctx, method, targetURL, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set content type
	if ct := req.GetContentType(); ct != "" {
		httpReq.Header.Set("Content-Type", ct)
	} else if strings.ToLower(injectionPoint) == "json_body" {
		httpReq.Header.Set("Content-Type", "application/json")
	}

	// Apply custom headers
	utils.ApplyHeaders(httpReq, req.GetHeaders())

	// Inject into header if that's the injection point
	if strings.ToLower(injectionPoint) == "header" {
		httpReq.Header.Set(paramName, payload)
	}

	// Apply cookies
	utils.ApplyCookies(httpReq, req.GetCookies())

	// Inject into cookie if that's the injection point
	if strings.ToLower(injectionPoint) == "cookie" {
		httpReq.AddCookie(&http.Cookie{Name: paramName, Value: payload})
	}

	// Apply auth token
	if token := req.GetAuthToken(); token != "" {
		headerName := req.GetAuthHeaderName()
		if headerName == "" {
			headerName = "Authorization"
		}
		httpReq.Header.Set(headerName, token)
	}

	return httpReq, nil
}

// buildAttackResult aggregates PayloadResults into a final AttackResult.
func (e *Engine) buildAttackResult(req *pb.AttackRequest, results []*pb.PayloadResult, duration time.Duration) *pb.AttackResult {
	var successful, failed int32
	var totalResponseTime float64

	for _, r := range results {
		if r == nil {
			continue
		}
		if r.Error != "" {
			failed++
		} else {
			successful++
			totalResponseTime += r.ResponseTimeMs
		}
	}

	totalReqs := successful + failed
	var avgResponseTime float64
	if successful > 0 {
		avgResponseTime = totalResponseTime / float64(successful)
	}

	status := "success"
	if failed == totalReqs && totalReqs > 0 {
		status = "failed"
	}

	return &pb.AttackResult{
		AttackId:           req.GetAttackId(),
		AttackType:         req.GetAttackType(),
		Status:             status,
		PayloadResults:     results,
		TotalRequests:      totalReqs,
		SuccessfulRequests: successful,
		FailedRequests:     failed,
		DurationMs:         float64(duration.Milliseconds()),
		AvgResponseTimeMs:  avgResponseTime,
	}
}

// ---------------------------------------------------------------------------
// Internal: Attack Registration (for abort support)
// ---------------------------------------------------------------------------

func (e *Engine) registerAttack(id string, cancel context.CancelFunc) {
	e.mu.Lock()
	e.cancelFuncs[id] = cancel
	e.mu.Unlock()
}

func (e *Engine) deregisterAttack(id string) {
	e.mu.Lock()
	delete(e.cancelFuncs, id)
	e.mu.Unlock()
}
