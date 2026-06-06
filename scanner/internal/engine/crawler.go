package engine

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/PuerkitoBio/goquery"
	"go.uber.org/zap"

	"github.com/VectorCipher/Wraith/scanner/internal/utils"
	pb "github.com/VectorCipher/Wraith/scanner/proto"
)

// Crawler performs BFS web crawling to discover endpoints, forms, and links.
type Crawler struct {
	httpClient *http.Client
	logger     *zap.Logger
}

// NewCrawler creates a new Crawler instance.
func NewCrawler(client *http.Client, logger *zap.Logger) *Crawler {
	return &Crawler{
		httpClient: client,
		logger:     logger.Named("crawler"),
	}
}

// Crawl performs a BFS crawl starting from the target URL.
// Discovered pages are streamed to resultCh in real-time.
func (c *Crawler) Crawl(ctx context.Context, req *pb.CrawlRequest, resultCh chan<- *pb.CrawlResult) error {
	startURL := req.GetTargetUrl()
	maxDepth := int(req.GetMaxDepth())
	maxPages := int(req.GetMaxPages())
	delayMs := int(req.GetDelayBetweenMs())

	if maxDepth <= 0 {
		maxDepth = 3
	}
	if maxPages <= 0 {
		maxPages = 100
	}

	baseURL, err := url.Parse(startURL)
	if err != nil {
		return fmt.Errorf("invalid start URL: %w", err)
	}

	c.logger.Info("Starting crawl",
		zap.String("url", startURL),
		zap.Int("max_depth", maxDepth),
		zap.Int("max_pages", maxPages),
	)

	// BFS state
	type queueItem struct {
		url   string
		depth int
	}

	visited := &sync.Map{}
	queue := make(chan queueItem, maxPages)
	queue <- queueItem{url: startURL, depth: 0}
	visited.Store(startURL, true)

	pagesVisited := 0

	for {
		select {
		case <-ctx.Done():
			c.logger.Info("Crawl cancelled", zap.Int("pages_visited", pagesVisited))
			return ctx.Err()
		case item := <-queue:
			if pagesVisited >= maxPages {
				c.logger.Info("Max pages reached", zap.Int("pages", pagesVisited))
				return nil
			}
			if item.depth > maxDepth {
				continue
			}

			result := c.crawlPage(ctx, req, item.url, item.depth, baseURL)
			pagesVisited++

			// Send result to stream
			select {
			case resultCh <- result:
			case <-ctx.Done():
				return ctx.Err()
			}

			// Enqueue discovered links
			if item.depth < maxDepth && result.GetError() == "" {
				for _, link := range result.GetLinks() {
					absLink := c.resolveURL(baseURL, link)
					if absLink == "" || !c.isInScope(baseURL, absLink) {
						continue
					}

					if _, loaded := visited.LoadOrStore(absLink, true); !loaded {
						select {
						case queue <- queueItem{url: absLink, depth: item.depth + 1}:
						default:
							// Queue full
						}
					}
				}
			}

			// Politeness delay
			if delayMs > 0 {
				time.Sleep(time.Duration(delayMs) * time.Millisecond)
			}

		default:
			// Queue empty and no more work
			if len(queue) == 0 {
				c.logger.Info("Crawl complete", zap.Int("pages_visited", pagesVisited))
				return nil
			}
		}
	}
}

// crawlPage fetches a single page and extracts data from it.
func (c *Crawler) crawlPage(ctx context.Context, req *pb.CrawlRequest, pageURL string, depth int, baseURL *url.URL) *pb.CrawlResult {
	result := &pb.CrawlResult{
		Url:    pageURL,
		Method: "GET",
		Depth:  int32(depth),
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", pageURL, nil)
	if err != nil {
		result.Error = fmt.Sprintf("failed to create request: %v", err)
		return result
	}

	// Apply custom headers and cookies
	utils.ApplyHeaders(httpReq, req.GetHeaders())
	utils.ApplyCookies(httpReq, req.GetCookies())
	httpReq.Header.Set("User-Agent", "WRAITH-Scanner/0.1.0")

	start := time.Now()
	resp, err := c.httpClient.Do(httpReq)
	elapsed := time.Since(start)

	if err != nil {
		result.Error = fmt.Sprintf("request failed: %v", err)
		result.ResponseTimeMs = float64(elapsed.Milliseconds())
		return result
	}
	defer resp.Body.Close()

	result.StatusCode = int32(resp.StatusCode)
	result.ResponseTimeMs = float64(elapsed.Milliseconds())
	result.ContentType = resp.Header.Get("Content-Type")
	result.ContentLength = resp.ContentLength
	result.ResponseHeaders = utils.FlattenHeaders(resp.Header)

	// Only parse HTML responses
	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "text/html") {
		return result
	}

	// Parse HTML with goquery
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		result.Error = fmt.Sprintf("failed to parse HTML: %v", err)
		return result
	}

	// Extract page title
	result.PageTitle = doc.Find("title").First().Text()

	// Extract links
	doc.Find("a[href]").Each(func(_ int, s *goquery.Selection) {
		if href, exists := s.Attr("href"); exists {
			result.Links = append(result.Links, href)
		}
	})

	// Extract scripts
	doc.Find("script[src]").Each(func(_ int, s *goquery.Selection) {
		if src, exists := s.Attr("src"); exists {
			result.Scripts = append(result.Scripts, src)
		}
	})

	// Extract forms
	doc.Find("form").Each(func(_ int, s *goquery.Selection) {
		form := c.extractForm(s)
		result.Forms = append(result.Forms, form)
	})

	// Detect API endpoints from JavaScript
	doc.Find("script").Each(func(_ int, s *goquery.Selection) {
		text := s.Text()
		endpoints := extractAPIEndpoints(text)
		result.ApiEndpoints = append(result.ApiEndpoints, endpoints...)
	})

	// Detect technologies from HTML
	result.DetectedTech = c.detectTechFromHTML(doc, resp.Header)

	return result
}

// extractForm parses an HTML form element into a FormData proto message.
func (c *Crawler) extractForm(s *goquery.Selection) *pb.FormData {
	action, _ := s.Attr("action")
	method, _ := s.Attr("method")
	enctype, _ := s.Attr("enctype")

	if method == "" {
		method = "GET"
	}

	form := &pb.FormData{
		Action:  action,
		Method:  strings.ToUpper(method),
		Enctype: enctype,
	}

	// Extract input fields
	s.Find("input, textarea, select").Each(func(_ int, input *goquery.Selection) {
		name, _ := input.Attr("name")
		if name == "" {
			return
		}

		fieldType, _ := input.Attr("type")
		if fieldType == "" {
			fieldType = "text"
		}

		value, _ := input.Attr("value")
		_, required := input.Attr("required")

		form.Fields = append(form.Fields, &pb.FormField{
			Name:      name,
			FieldType: fieldType,
			Value:     value,
			Required:  required,
		})
	})

	return form
}

// detectTechFromHTML identifies technologies from HTML content and headers.
func (c *Crawler) detectTechFromHTML(doc *goquery.Document, headers http.Header) []string {
	var tech []string

	// Check meta generators
	doc.Find("meta[name=generator]").Each(func(_ int, s *goquery.Selection) {
		if content, exists := s.Attr("content"); exists {
			tech = append(tech, content)
		}
	})

	// Check common framework indicators
	indicators := map[string]string{
		"script[src*='react']":    "React",
		"script[src*='angular']":  "Angular",
		"script[src*='vue']":      "Vue.js",
		"script[src*='jquery']":   "jQuery",
		"script[src*='bootstrap']": "Bootstrap",
		"link[href*='bootstrap']": "Bootstrap",
	}

	for selector, name := range indicators {
		if doc.Find(selector).Length() > 0 {
			tech = append(tech, name)
		}
	}

	// Check response headers
	if server := headers.Get("Server"); server != "" {
		tech = append(tech, "Server: "+server)
	}
	if powered := headers.Get("X-Powered-By"); powered != "" {
		tech = append(tech, powered)
	}

	return tech
}

// extractAPIEndpoints finds API-like paths in JavaScript source.
func extractAPIEndpoints(jsSource string) []string {
	var endpoints []string

	// Common API path patterns
	patterns := []string{"/api/", "/v1/", "/v2/", "/graphql", "/rest/"}
	for _, pattern := range patterns {
		idx := 0
		for {
			pos := strings.Index(jsSource[idx:], pattern)
			if pos == -1 {
				break
			}
			pos += idx

			// Extract the full path (find start and end quotes)
			start := strings.LastIndexAny(jsSource[:pos], "\"'`")
			end := strings.IndexAny(jsSource[pos:], "\"'`")
			if start != -1 && end != -1 {
				path := jsSource[start+1 : pos+end]
				if len(path) < 200 && !strings.ContainsAny(path, " \n\t") {
					endpoints = append(endpoints, path)
				}
			}

			idx = pos + len(pattern)
			if idx >= len(jsSource) {
				break
			}
		}
	}

	return endpoints
}

// resolveURL resolves a potentially relative URL against a base URL.
func (c *Crawler) resolveURL(base *url.URL, rawURL string) string {
	if rawURL == "" || strings.HasPrefix(rawURL, "#") || strings.HasPrefix(rawURL, "javascript:") || strings.HasPrefix(rawURL, "mailto:") {
		return ""
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return ""
	}

	resolved := base.ResolveReference(parsed)
	resolved.Fragment = "" // Strip fragments
	return resolved.String()
}

// isInScope checks if a URL is within the same origin as the base.
func (c *Crawler) isInScope(base *url.URL, rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	return parsed.Host == base.Host
}
