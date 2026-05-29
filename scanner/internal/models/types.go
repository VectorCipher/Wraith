// Package models defines internal data types for the WRAITH scanner.
package models

import "time"

const (
	// DefaultPort is the default gRPC listen port.
	DefaultPort = 9090

	// DefaultMaxConcurrent is the default max concurrent HTTP requests.
	DefaultMaxConcurrent = 50

	// DefaultRateLimit is the default requests per second.
	DefaultRateLimit = 100

	// DefaultRequestTimeout is the default per-request timeout in seconds.
	DefaultRequestTimeout = 30

	// DefaultMaxResponseBody is the max response body size to capture (1MB).
	DefaultMaxResponseBody int64 = 1 * 1024 * 1024

	// Version is the scanner service version.
	Version = "0.1.0"
)

// Config holds the scanner's runtime configuration.
// Populated from CLI flags or environment variables at startup.
type Config struct {
	// gRPC server port
	Port int

	// HTTP scanning behavior
	MaxConcurrentRequests int
	RateLimitPerSecond    int
	RequestTimeoutSeconds int
	MaxResponseBodySize   int64

	// Logging
	LogLevel string
	LogJSON  bool
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		Port:                  DefaultPort,
		MaxConcurrentRequests: DefaultMaxConcurrent,
		RateLimitPerSecond:    DefaultRateLimit,
		RequestTimeoutSeconds: DefaultRequestTimeout,
		MaxResponseBodySize:   DefaultMaxResponseBody,
		LogLevel:              "info",
		LogJSON:               false,
	}
}

// Stats tracks scanner-wide statistics.
// Fields are updated atomically by the engine.
type Stats struct {
	StartTime             time.Time
	TotalRequestsSent     int64
	TotalAttacksCompleted int64
	ActiveAttacks         int32
}

// UptimeSeconds returns the number of seconds since the scanner started.
func (s *Stats) UptimeSeconds() int64 {
	return int64(time.Since(s.StartTime).Seconds())
}
