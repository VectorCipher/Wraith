package utils

import (
	"crypto/tls"
	"io"
	"net"
	"net/http"
	"net/http/cookiejar"
	"strings"
	"time"
)

// NewHTTPClient creates a configured HTTP client optimized for scanning.
//
// Uses aggressive connection pooling and accepts self-signed TLS certs
// (common in pentest targets). The followRedirects parameter controls
// whether the client follows HTTP 3xx redirects.
func NewHTTPClient(timeoutSeconds int, followRedirects bool) *http.Client {
	transport := &http.Transport{
		// Connection pooling — reuse connections aggressively
		MaxIdleConns:        200,
		MaxIdleConnsPerHost: 50,
		MaxConnsPerHost:     100,
		IdleConnTimeout:     90 * time.Second,

		// Timeouts
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: time.Duration(timeoutSeconds) * time.Second,
		ExpectContinueTimeout: 1 * time.Second,

		// Dialer with timeout and keepalive
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,

		// TLS — accept self-signed certs (intentional for pentest targets)
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true, //nolint:gosec
		},

		DisableCompression: false,
	}

	jar, _ := cookiejar.New(nil)

	client := &http.Client{
		Transport: transport,
		Timeout:   time.Duration(timeoutSeconds) * time.Second,
		Jar:       jar,
	}

	if !followRedirects {
		client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		}
	}

	return client
}

// ReadBodyLimited reads up to maxBytes from a response body.
// Returns the body string, whether it was truncated, and any error.
func ReadBodyLimited(body io.ReadCloser, maxBytes int64) (string, bool, error) {
	if body == nil {
		return "", false, nil
	}
	defer body.Close()

	limited := io.LimitReader(body, maxBytes+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return "", false, err
	}

	truncated := int64(len(data)) > maxBytes
	if truncated {
		data = data[:maxBytes]
	}

	return string(data), truncated, nil
}

// FlattenHeaders converts http.Header (multi-value) to a single-value map.
// Duplicate header values are joined with ", ".
func FlattenHeaders(h http.Header) map[string]string {
	flat := make(map[string]string, len(h))
	for key, values := range h {
		flat[key] = strings.Join(values, ", ")
	}
	return flat
}

// ApplyHeaders sets headers on an HTTP request from a string map.
func ApplyHeaders(req *http.Request, headers map[string]string) {
	for k, v := range headers {
		req.Header.Set(k, v)
	}
}

// ApplyCookies adds cookies to an HTTP request from a string map.
func ApplyCookies(req *http.Request, cookies map[string]string) {
	for name, value := range cookies {
		req.AddCookie(&http.Cookie{Name: name, Value: value})
	}
}
