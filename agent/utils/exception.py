"""
WRAITH Custom Exceptions

Clean error hierarchy for the entire application.
Every module raises specific exceptions instead of generic ones.
This makes error handling precise and debugging easy.

"""


class WraithError(Exception):
    """
    Base exception for ALL WRAITH errors.
    
    Every custom exception inherits from this.
    You can catch WraithError to catch ANY wraith-specific error,
    or catch a specific subclass for targeted handling.
    """

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class LLMConnectionError(WraithError):
    """
    Cannot connect to Ollama or the LLM is unresponsive.
    
    Raised when:
    - Ollama server is not running
    - Ollama is unreachable at the configured host/port
    - LLM times out during generation
    - Unexpected Ollama API error
    """
    pass


class ModelNotFoundError(WraithError):
    """
    Requested LLM model is not available in Ollama.
    
    Raised when:
    - Model hasn't been pulled yet (ollama pull <model>)
    - Model name is misspelled in config
    - Model was deleted from Ollama
    """
    pass


class ScannerConnectionError(WraithError):
    """
    Cannot connect to the Go scanner via gRPC.
    
    Raised when:
    - Go scanner service is not running
    - gRPC connection refused on configured host/port
    - gRPC call times out
    - Scanner crashes mid-scan
    """
    pass


class TargetUnreachableError(WraithError):
    """
    Target URL is not reachable.
    
    Raised when:
    - Target URL is malformed
    - DNS resolution fails
    - Connection refused / timed out
    - Target returns no response
    """
    pass


class ScanAbortedError(WraithError):
    """
    Scan was aborted by user or due to critical error.
    
    Raised when:
    - User presses Ctrl+C during scan
    - Max scan duration exceeded
    - Too many consecutive errors
    - Critical unrecoverable error during scan
    """
    pass


class ConfigurationError(WraithError):
    """
    Invalid configuration detected.
    
    Raised when:
    - .env file missing required values
    - models.yaml has invalid structure
    - attacks.yaml has invalid structure
    - Invalid scan mode specified
    - Invalid source code path
    """
    pass


class AttackError(WraithError):
    """
    Error during attack execution.
    
    Raised when:
    - Attack module fails to initialize
    - Payload generation fails
    - Attack execution encounters unexpected error
    - Result parsing fails
    """
    pass


class ReportGenerationError(WraithError):
    """
    Error during report generation.
    
    Raised when:
    - Template rendering fails
    - PDF generation fails
    - Output directory not writable
    - Invalid report format specified
    """
    pass