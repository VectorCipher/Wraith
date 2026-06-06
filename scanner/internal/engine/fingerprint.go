package engine

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/VectorCipher/Wraith/scanner/internal/utils"
	pb "github.com/VectorCipher/Wraith/scanner/proto"
)

// Fingerprinter identifies the target's technology stack by analyzing
// HTTP responses, headers, error pages, and known path probes.
type Fingerprinter struct {
	httpClient *http.Client
	logger     *zap.Logger
}

// NewFingerprinter creates a new Fingerprinter instance.
func NewFingerprinter(client *http.Client, logger *zap.Logger) *Fingerprinter {
	return &Fingerprinter{
		httpClient: client,
		logger:     logger.Named("fingerprint"),
	}
}

// Fingerprint probes the target and identifies its technology stack.
func (f *Fingerprinter) Fingerprint(ctx context.Context, req *pb.FingerprintRequest) (*pb.FingerprintResult, error) {
	targetURL := req.GetTargetUrl()
	f.logger.Info("Starting fingerprint", zap.String("target", targetURL))

	start := time.Now()
	result := &pb.FingerprintResult{
		ConfidenceScores: make(map[string]float32),
	}

	requestCount := 0

	// ------------------------------------------------------------------
	// 1. Main page analysis
	// ------------------------------------------------------------------
	mainResp, mainHeaders, err := f.probe(ctx, req, targetURL)
	requestCount++

	if err != nil {
		result.Error = fmt.Sprintf("main page probe failed: %v", err)
		result.RequestsSent = int32(requestCount)
		result.DurationMs = float64(time.Since(start).Milliseconds())
		return result, nil
	}

	// Analyze main response headers
	f.analyzeHeaders(mainHeaders, result)

	// Collect response headers as evidence
	result.ResponseHeaders = mainHeaders

	// ------------------------------------------------------------------
	// 2. Error page analysis (trigger 404 and 500 errors)
	// ------------------------------------------------------------------
	errorPaths := []string{
		"/wraith-nonexistent-path-404",
		"/wraith-test-path.php",
		"/wraith-test-path.asp",
		"/wraith-test-path.jsp",
	}

	errorLoop:
	for _, path := range errorPaths {
		select {
		case <-ctx.Done():
			break errorLoop
		default:
		}

		body, _, err := f.probe(ctx, req, targetURL+path)
		requestCount++
		if err != nil {
			continue
		}
		f.analyzeErrorPage(body, result)
	}

	// ------------------------------------------------------------------
	// 3. Known path probing
	// ------------------------------------------------------------------
	knownPaths := map[string]string{
		"/robots.txt":       "robots.txt",
		"/sitemap.xml":      "sitemap",
		"/wp-admin/":        "WordPress",
		"/wp-login.php":     "WordPress",
		"/administrator/":   "Joomla",
		"/admin/":           "admin_panel",
		"/phpmyadmin/":      "phpMyAdmin",
		"/.env":             "dotenv_exposed",
		"/.git/HEAD":        "git_exposed",
		"/server-status":    "Apache",
		"/elmah.axd":        "ASP.NET",
		"/swagger-ui.html":  "Swagger",
		"/api/swagger.json": "Swagger",
		"/graphql":          "GraphQL",
		"/health":           "health_endpoint",
		"/actuator":         "Spring Boot",
		"/actuator/health":  "Spring Boot",
	}

	pathLoop:
	for path, tech := range knownPaths {
		select {
		case <-ctx.Done():
			break pathLoop
		default:
		}

		_, headers, err := f.probe(ctx, req, targetURL+path)
		requestCount++
		if err != nil {
			continue
		}

		// If we got a non-404 response, the path likely exists
		if statusCode := f.getStatusFromHeaders(headers); statusCode > 0 && statusCode != 404 {
			switch tech {
			case "WordPress":
				result.Framework = "WordPress"
				result.Language = "PHP"
				result.ConfidenceScores["WordPress"] = 0.9
			case "Joomla":
				result.Framework = "Joomla"
				result.Language = "PHP"
				result.ConfidenceScores["Joomla"] = 0.8
			case "Spring Boot":
				result.Framework = "Spring Boot"
				result.Language = "Java"
				result.ConfidenceScores["Spring Boot"] = 0.85
			case "GraphQL":
				result.OtherTech = append(result.OtherTech, "GraphQL")
				result.ConfidenceScores["GraphQL"] = 0.9
			case "Swagger":
				result.OtherTech = append(result.OtherTech, "Swagger/OpenAPI")
				result.ConfidenceScores["Swagger"] = 0.9
			case "git_exposed":
				result.OtherTech = append(result.OtherTech, "Git repository exposed")
				result.ConfidenceScores["git_exposed"] = 0.95
			case "dotenv_exposed":
				result.OtherTech = append(result.OtherTech, ".env file exposed")
				result.ConfidenceScores["dotenv_exposed"] = 0.95
			}
		}
	}

	// ------------------------------------------------------------------
	// 4. Cookie analysis
	// ------------------------------------------------------------------
	_ = mainResp // Using the response string for additional analysis
	f.analyzeCookies(mainHeaders, result)

	result.RequestsSent = int32(requestCount)
	result.DurationMs = float64(time.Since(start).Milliseconds())

	f.logger.Info("Fingerprint complete",
		zap.String("language", result.GetLanguage()),
		zap.String("framework", result.GetFramework()),
		zap.String("server", result.GetWebServer()),
		zap.Int("requests", requestCount),
		zap.Duration("duration", time.Since(start)),
	)

	return result, nil
}

// ---------------------------------------------------------------------------
// Internal: HTTP Probing
// ---------------------------------------------------------------------------

// probe sends a GET request and returns (body, headers_map, error).
func (f *Fingerprinter) probe(ctx context.Context, req *pb.FingerprintRequest, targetURL string) (string, map[string]string, error) {
	timeout := int(req.GetTimeoutSeconds())
	if timeout <= 0 {
		timeout = 10
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", targetURL, nil)
	if err != nil {
		return "", nil, err
	}

	utils.ApplyHeaders(httpReq, req.GetHeaders())
	utils.ApplyCookies(httpReq, req.GetCookies())
	httpReq.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

	resp, err := f.httpClient.Do(httpReq)
	if err != nil {
		return "", nil, err
	}

	body, _, _ := utils.ReadBodyLimited(resp.Body, 64*1024) // 64KB limit for fingerprinting
	headers := utils.FlattenHeaders(resp.Header)

	// Store status code in headers map for later use
	headers["_status_code"] = fmt.Sprintf("%d", resp.StatusCode)

	return body, headers, nil
}

// getStatusFromHeaders extracts the status code stored by probe().
func (f *Fingerprinter) getStatusFromHeaders(headers map[string]string) int {
	if s, ok := headers["_status_code"]; ok {
		var code int
		fmt.Sscanf(s, "%d", &code)
		return code
	}
	return 0
}

// ---------------------------------------------------------------------------
// Internal: Header Analysis
// ---------------------------------------------------------------------------

func (f *Fingerprinter) analyzeHeaders(headers map[string]string, result *pb.FingerprintResult) {
	// Server header
	if server, ok := headers["Server"]; ok {
		result.WebServer = server
		s := strings.ToLower(server)
		switch {
		case strings.Contains(s, "nginx"):
			result.ConfidenceScores["nginx"] = 0.95
		case strings.Contains(s, "apache"):
			result.ConfidenceScores["Apache"] = 0.95
		case strings.Contains(s, "gunicorn"):
			result.WebServer = server
			result.Language = "Python"
			result.ConfidenceScores["Python"] = 0.8
		case strings.Contains(s, "microsoft-iis"):
			result.ConfidenceScores["IIS"] = 0.95
			result.Language = "C#"
		case strings.Contains(s, "openresty"):
			result.ConfidenceScores["OpenResty"] = 0.9
		}
	}

	// X-Powered-By header
	if powered, ok := headers["X-Powered-By"]; ok {
		p := strings.ToLower(powered)
		switch {
		case strings.Contains(p, "php"):
			result.Language = "PHP"
			result.ConfidenceScores["PHP"] = 0.9
		case strings.Contains(p, "asp.net"):
			result.Language = "C#"
			result.Framework = "ASP.NET"
			result.ConfidenceScores["ASP.NET"] = 0.9
		case strings.Contains(p, "express"):
			result.Language = "JavaScript"
			result.Framework = "Express"
			result.ConfidenceScores["Express"] = 0.85
		case strings.Contains(p, "next.js"):
			result.Language = "JavaScript"
			result.Framework = "Next.js"
			result.ConfidenceScores["Next.js"] = 0.9
		}
	}

	// X-AspNet-Version
	if _, ok := headers["X-Aspnet-Version"]; ok {
		result.Language = "C#"
		result.Framework = "ASP.NET"
		result.ConfidenceScores["ASP.NET"] = 0.95
	}

	// X-Django-* headers
	if _, ok := headers["X-Frame-Options"]; ok {
		// Not definitive, but Django sets it by default
	}
}

// ---------------------------------------------------------------------------
// Internal: Error Page Analysis
// ---------------------------------------------------------------------------

func (f *Fingerprinter) analyzeErrorPage(body string, result *pb.FingerprintResult) {
	lower := strings.ToLower(body)

	signatures := map[string][2]string{
		"Traceback (most recent call last)": {"Python", ""},
		"Django":                            {"Python", "Django"},
		"flask":                             {"Python", "Flask"},
		"laravel":                           {"PHP", "Laravel"},
		"symfony":                           {"PHP", "Symfony"},
		"codeigniter":                       {"PHP", "CodeIgniter"},
		"at javax.servlet":                  {"Java", "Servlet"},
		"spring":                            {"Java", "Spring"},
		"asp.net":                           {"C#", "ASP.NET"},
		"ruby on rails":                     {"Ruby", "Rails"},
		"sinatra":                           {"Ruby", "Sinatra"},
		"express":                           {"JavaScript", "Express"},
		"next.js":                           {"JavaScript", "Next.js"},
		"nuxt":                              {"JavaScript", "Nuxt"},
	}

	for sig, langFramework := range signatures {
		if strings.Contains(lower, strings.ToLower(sig)) {
			if langFramework[0] != "" && result.Language == "" {
				result.Language = langFramework[0]
				result.ConfidenceScores[langFramework[0]] = 0.7
			}
			if langFramework[1] != "" && result.Framework == "" {
				result.Framework = langFramework[1]
				result.ConfidenceScores[langFramework[1]] = 0.7
			}
			result.ErrorSignatures = append(result.ErrorSignatures, sig)
		}
	}

	// Database detection from error messages
	dbSignatures := map[string]string{
		"mysql":                "MySQL",
		"postgresql":           "PostgreSQL",
		"sqlite":               "SQLite",
		"microsoft sql server": "MSSQL",
		"oracle":               "Oracle",
		"mongodb":              "MongoDB",
	}

	for sig, db := range dbSignatures {
		if strings.Contains(lower, sig) {
			result.Database = db
			result.ConfidenceScores[db] = 0.75
		}
	}
}

// ---------------------------------------------------------------------------
// Internal: Cookie Analysis
// ---------------------------------------------------------------------------

func (f *Fingerprinter) analyzeCookies(headers map[string]string, result *pb.FingerprintResult) {
	setCookie, ok := headers["Set-Cookie"]
	if !ok {
		return
	}

	cookieLower := strings.ToLower(setCookie)

	cookieSignatures := map[string][2]string{
		"phpsessid":         {"PHP", ""},
		"jsessionid":        {"Java", ""},
		"asp.net_sessionid": {"C#", "ASP.NET"},
		"csrftoken":         {"Python", "Django"},
		"_rails_session":    {"Ruby", "Rails"},
		"laravel_session":   {"PHP", "Laravel"},
		"connect.sid":       {"JavaScript", "Express"},
	}

	for sig, langFramework := range cookieSignatures {
		if strings.Contains(cookieLower, sig) {
			result.DetectedCookies = append(result.DetectedCookies, sig)

			if langFramework[0] != "" && result.Language == "" {
				result.Language = langFramework[0]
				result.ConfidenceScores[langFramework[0]] = 0.8
			}
			if langFramework[1] != "" && result.Framework == "" {
				result.Framework = langFramework[1]
				result.ConfidenceScores[langFramework[1]] = 0.8
			}
		}
	}
}
