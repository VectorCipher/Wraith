// Package server implements the gRPC middleware for the WRAITH scanner.
package server

import (
	"context"
	"runtime/debug"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// LoggingInterceptor returns a gRPC unary server interceptor that logs
// every RPC call with method name, duration, and status.
func LoggingInterceptor(logger *zap.Logger) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		start := time.Now()

		logger.Info("gRPC request",
			zap.String("method", info.FullMethod),
		)

		resp, err := handler(ctx, req)
		duration := time.Since(start)

		if err != nil {
			st, _ := status.FromError(err)
			logger.Error("gRPC error",
				zap.String("method", info.FullMethod),
				zap.String("code", st.Code().String()),
				zap.String("message", st.Message()),
				zap.Duration("duration", duration),
			)
		} else {
			logger.Info("gRPC response",
				zap.String("method", info.FullMethod),
				zap.Duration("duration", duration),
			)
		}

		return resp, err
	}
}

// StreamLoggingInterceptor returns a gRPC stream server interceptor
// that logs stream lifecycle events.
func StreamLoggingInterceptor(logger *zap.Logger) grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		start := time.Now()

		logger.Info("gRPC stream started",
			zap.String("method", info.FullMethod),
			zap.Bool("client_stream", info.IsClientStream),
			zap.Bool("server_stream", info.IsServerStream),
		)

		err := handler(srv, ss)
		duration := time.Since(start)

		if err != nil {
			st, _ := status.FromError(err)
			logger.Error("gRPC stream error",
				zap.String("method", info.FullMethod),
				zap.String("code", st.Code().String()),
				zap.Duration("duration", duration),
			)
		} else {
			logger.Info("gRPC stream completed",
				zap.String("method", info.FullMethod),
				zap.Duration("duration", duration),
			)
		}

		return err
	}
}

// RecoveryInterceptor returns a gRPC unary interceptor that recovers
// from panics and returns an Internal error instead of crashing.
func RecoveryInterceptor(logger *zap.Logger) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (resp interface{}, err error) {
		defer func() {
			if r := recover(); r != nil {
				logger.Error("gRPC panic recovered",
					zap.String("method", info.FullMethod),
					zap.Any("panic", r),
					zap.String("stack", string(debug.Stack())),
				)
				err = status.Errorf(codes.Internal, "internal server error")
			}
		}()

		return handler(ctx, req)
	}
}

// StreamRecoveryInterceptor returns a gRPC stream interceptor that
// recovers from panics during stream handling.
func StreamRecoveryInterceptor(logger *zap.Logger) grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) (err error) {
		defer func() {
			if r := recover(); r != nil {
				logger.Error("gRPC stream panic recovered",
					zap.String("method", info.FullMethod),
					zap.Any("panic", r),
					zap.String("stack", string(debug.Stack())),
				)
				err = status.Errorf(codes.Internal, "internal server error")
			}
		}()

		return handler(srv, ss)
	}
}
