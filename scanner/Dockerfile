# Stage 1: Build the Go Scanner
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./
RUN go mod download

# Copy the source code
COPY . .

# Build the executable
RUN CGO_ENABLED=0 GOOS=linux go build -o /wraith-scanner ./cmd/scanner/main.go

# Stage 2: Minimal runtime image
FROM alpine:latest

WORKDIR /root/

# Copy the pre-built binary from builder
COPY --from=builder /wraith-scanner .

# Expose gRPC port
EXPOSE 50051

# Run the scanner
ENTRYPOINT ["./wraith-scanner"]
CMD ["-port", "50051"]
