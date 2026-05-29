"""
WRAITH Scanner Client

Async gRPC client that connects the Python AI agent to the Go scanner.
This is the ONLY module that talks to the Go scanner — all other modules
go through this client.

Provides clean async methods for every scanner operation:
    - health_check()          → is the scanner alive?
    - get_status()            → scanner resource usage
    - execute_attack()        → run payloads, get results
    - execute_attack_stream() → stream results in real-time
    - crawl_target()          → discover endpoints
    - fingerprint_target()    → detect tech stack
    - send_baseline()         → get normal response for comparison
    - send_raw_request()      → custom HTTP through the scanner
    - abort_all()             → emergency stop
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncGenerator

import grpc

# Ensure generated stubs are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scanner_pb2 as pb          # noqa: E402
import scanner_pb2_grpc as pb_grpc  # noqa: E402

from config import settings
from utils.exception import ScannerConnectionError
from utils.logger import get_logger
from scanner_client.converters import (
    attack_request_to_proto,
    attack_result_from_proto,
    baseline_response_to_dict,
    build_baseline_request,
    build_crawl_request,
    build_fingerprint_request,
    build_raw_http_request,
    crawl_result_to_dict,
    fingerprint_to_tech_stack,
    payload_result_from_proto,
    raw_response_to_dict,
)
from models.attack_result import (
    AttackRequest,
    AttackResult,
    PayloadResult,
)
from models.target import TechStack

logger = get_logger("scanner_client")


# ===================================================================
# Scanner Status Data
# ===================================================================

class ScannerStatus:
    """Parsed status snapshot from the Go scanner."""

    def __init__(
        self,
        ready: bool,
        version: str,
        uptime_seconds: int,
        active_goroutines: int,
        memory_used_bytes: int,
        active_attacks: int,
        total_requests_sent: int,
        total_attacks_completed: int,
        max_concurrent_requests: int,
        rate_limit_per_second: int,
        request_timeout_seconds: int,
    ) -> None:
        self.ready = ready
        self.version = version
        self.uptime_seconds = uptime_seconds
        self.active_goroutines = active_goroutines
        self.memory_used_bytes = memory_used_bytes
        self.active_attacks = active_attacks
        self.total_requests_sent = total_requests_sent
        self.total_attacks_completed = total_attacks_completed
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_per_second = rate_limit_per_second
        self.request_timeout_seconds = request_timeout_seconds

    @property
    def memory_used_mb(self) -> float:
        return self.memory_used_bytes / (1024 * 1024)

    @property
    def display_summary(self) -> dict[str, str]:
        return {
            "Ready": "✅ Yes" if self.ready else "❌ No",
            "Version": self.version,
            "Uptime": f"{self.uptime_seconds}s",
            "Active Attacks": str(self.active_attacks),
            "Goroutines": str(self.active_goroutines),
            "Memory": f"{self.memory_used_mb:.1f} MB",
            "Total Requests": str(self.total_requests_sent),
            "Total Attacks": str(self.total_attacks_completed),
            "Concurrency": str(self.max_concurrent_requests),
            "Rate Limit": f"{self.rate_limit_per_second} req/s",
        }


# ===================================================================
# ScannerClient — Async gRPC Client
# ===================================================================

class ScannerClient:
    """
    Async gRPC client for the Go scanner service.

    All methods are async. The client manages a single gRPC channel
    that is reused across calls.

    Usage:
        client = ScannerClient()
        await client.connect()
        healthy = await client.health_check()
        result = await client.execute_attack(request)
        await client.close()
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._host = host or settings.scanner_grpc_host
        self._port = port or settings.scanner_grpc_port
        self._address = f"{self._host}:{self._port}"

        self._channel: grpc.aio.Channel | None = None
        self._stub: pb_grpc.ScannerServiceStub | None = None
        self._connected: bool = False

        logger.debug(f"Scanner client created — target: {self._address}")

    # ===================================================================
    # Lifecycle
    # ===================================================================

    async def connect(self) -> None:
        """Open the gRPC channel to the Go scanner."""
        if self._connected:
            return

        try:
            self._channel = grpc.aio.insecure_channel(
                self._address,
                options=[
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50 MB
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 30_000),
                    ("grpc.keepalive_timeout_ms", 10_000),
                    ("grpc.keepalive_permit_without_calls", 1),
                ],
            )
            self._stub = pb_grpc.ScannerServiceStub(self._channel)
            self._connected = True
            logger.info(f"Connected to scanner at {self._address}")
        except Exception as e:
            raise ScannerConnectionError(
                message="Failed to create gRPC channel",
                details=f"Address: {self._address} | Error: {e}",
            )

    async def close(self) -> None:
        """Close the gRPC channel. Safe to call multiple times."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            self._connected = False
            logger.info("Scanner client connection closed")

    def _ensure_connected(self) -> None:
        """Raise if connect() hasn't been called."""
        if not self._connected or not self._stub:
            raise ScannerConnectionError(
                message="Scanner client is not connected",
                details="Call await client.connect() first",
            )

    # ===================================================================
    # Health & Status
    # ===================================================================

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Quick check: is the Go scanner alive?"""
        self._ensure_connected()
        try:
            resp = await self._stub.HealthCheck(
                pb.HealthCheckRequest(ping="wraith"),
                timeout=timeout,
            )
            healthy = resp.healthy
            logger.debug(
                f"Scanner health: {'OK' if healthy else 'UNHEALTHY'} "
                f"(v{resp.version}, {resp.active_attacks} active attacks)"
            )
            return healthy
        except grpc.aio.AioRpcError as e:
            logger.warning(f"Scanner health check failed: {e.code()} — {e.details()}")
            return False
        except Exception as e:
            logger.warning(f"Scanner health check failed: {e}")
            return False

    async def get_status(self, timeout: float = 5.0) -> ScannerStatus:
        """Get detailed scanner status and resource usage."""
        self._ensure_connected()
        try:
            resp = await self._stub.GetStatus(
                pb.StatusRequest(),
                timeout=timeout,
            )
            return ScannerStatus(
                ready=resp.ready,
                version=resp.version,
                uptime_seconds=resp.uptime_seconds,
                active_goroutines=resp.active_goroutines,
                memory_used_bytes=resp.memory_used_bytes,
                active_attacks=resp.active_attacks,
                total_requests_sent=resp.total_requests_sent,
                total_attacks_completed=resp.total_attacks_completed,
                max_concurrent_requests=resp.max_concurrent_requests,
                rate_limit_per_second=resp.rate_limit_per_second,
                request_timeout_seconds=resp.request_timeout_seconds,
            )
        except grpc.aio.AioRpcError as e:
            raise ScannerConnectionError(
                message="Failed to get scanner status",
                details=f"{e.code()} — {e.details()}",
            )

    # ===================================================================
    # Attack Execution
    # ===================================================================

    async def execute_attack(
        self,
        request: AttackRequest,
        timeout: float = 300.0,
    ) -> AttackResult:
        """
        Execute a complete attack (unary RPC).
        Sends all payloads and waits for the full result.
        Best for small payload sets (< 50 payloads).
        """
        self._ensure_connected()

        proto_req = attack_request_to_proto(request)

        logger.info(
            f"Executing attack: {request.attack_id} "
            f"({request.attack_type}, {len(request.payloads)} payloads)"
        )

        try:
            proto_result = await self._stub.ExecuteAttack(
                proto_req, timeout=timeout,
            )
            result = attack_result_from_proto(proto_result)
            logger.info(
                f"Attack complete: {result.attack_id} — "
                f"{result.status.value}, {result.total_requests} requests, "
                f"{result.duration_ms:.0f}ms"
            )
            return result
        except grpc.aio.AioRpcError as e:
            raise ScannerConnectionError(
                message=f"Attack execution failed: {request.attack_id}",
                details=f"{e.code()} — {e.details()}",
            )

    async def execute_attack_stream(
        self,
        request: AttackRequest,
        timeout: float = 300.0,
    ) -> AsyncGenerator[PayloadResult, None]:
        """
        Execute an attack with streaming results (server-streaming RPC).
        Yields each PayloadResult as it arrives from Go.
        Best for large payload sets where real-time feedback matters.
        """
        self._ensure_connected()

        proto_req = attack_request_to_proto(request)

        logger.info(
            f"Streaming attack: {request.attack_id} "
            f"({request.attack_type}, {len(request.payloads)} payloads)"
        )

        try:
            stream = self._stub.ExecuteAttackStream(
                proto_req, timeout=timeout,
            )
            count = 0
            async for proto_result in stream:
                count += 1
                yield payload_result_from_proto(proto_result)

            logger.info(
                f"Attack stream complete: {request.attack_id} — "
                f"{count} results received"
            )
        except grpc.aio.AioRpcError as e:
            raise ScannerConnectionError(
                message=f"Attack stream failed: {request.attack_id}",
                details=f"{e.code()} — {e.details()}",
            )

    # ===================================================================
    # Reconnaissance
    # ===================================================================

    async def crawl_target(
        self,
        target_url: str,
        max_depth: int = 3,
        max_pages: int = 100,
        timeout_seconds: int = 120,
        delay_between_ms: int = 100,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Crawl the target website (server-streaming RPC).
        Yields discovered page dicts as Go finds them.
        """
        self._ensure_connected()

        proto_req = build_crawl_request(
            target_url=target_url,
            max_depth=max_depth,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
            delay_between_ms=delay_between_ms,
            headers=headers,
            cookies=cookies,
        )

        logger.info(
            f"Crawling target: {target_url} "
            f"(depth={max_depth}, max_pages={max_pages})"
        )

        try:
            stream = self._stub.CrawlTarget(
                proto_req, timeout=float(timeout_seconds + 30),
            )
            count = 0
            async for proto_result in stream:
                count += 1
                yield crawl_result_to_dict(proto_result)

            logger.info(f"Crawl complete: {count} pages discovered")
        except grpc.aio.AioRpcError as e:
            raise ScannerConnectionError(
                message=f"Crawl failed for {target_url}",
                details=f"{e.code()} — {e.details()}",
            )

    async def fingerprint_target(
        self,
        target_url: str,
        timeout_seconds: int = 30,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> TechStack:
        """
        Fingerprint the target's technology stack (unary RPC).
        Returns a Pydantic TechStack populated from the probe results.
        """
        self._ensure_connected()

        proto_req = build_fingerprint_request(
            target_url=target_url,
            timeout_seconds=timeout_seconds,
            headers=headers,
            cookies=cookies,
        )

        logger.info(f"Fingerprinting target: {target_url}")

        try:
            proto_result = await self._stub.FingerprintTarget(
                proto_req, timeout=float(timeout_seconds + 15),
            )

            if proto_result.error:
                logger.warning(f"Fingerprint error: {proto_result.error}")

            tech = fingerprint_to_tech_stack(proto_result)
            logger.info(
                f"Fingerprint complete: "
                f"lang={tech.language}, framework={tech.framework}, "
                f"server={tech.web_server}, db={tech.database}"
            )
            return tech
        except grpc.aio.AioRpcError as e:
            raise ScannerConnectionError(
                message=f"Fingerprint failed for {target_url}",
                details=f"{e.code()} — {e.details()}",
            )

    async def send_baseline(
        self,
        url: str,
        method: str = "GET",
        body: str = "",
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        timeout_seconds: int = 30,
    ) -> dict:
        """
        Send a baseline request to establish the "normal" response.
        Used for comparison during attack result analysis.
        """
        self._ensure_connected()

        proto_req = build_baseline_request(
            url=url, method=method, body=body,
            headers=headers, cookies=cookies,
            timeout_seconds=timeout_seconds,
        )

        logger.debug(f"Sending baseline: {method} {url}")

        try:
            proto_result = await self._stub.SendBaselineRequest(
                proto_req, timeout=float(timeout_seconds + 10),
            )
            result = baseline_response_to_dict(proto_result)

            if result["error"]:
                logger.warning(f"Baseline error: {result['error']}")
            else:
                logger.debug(
                    f"Baseline received: {result['status_code']} "
                    f"({result['content_length']} bytes, "
                    f"{result['response_time_ms']:.0f}ms)"
                )
            return result
        except grpc.aio.AioRpcError as e:
            raise ScannerConnectionError(
                message=f"Baseline request failed for {url}",
                details=f"{e.code()} — {e.details()}",
            )

    # ===================================================================
    # Utility
    # ===================================================================

    async def send_raw_request(
        self,
        url: str,
        method: str = "GET",
        body: str = "",
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        timeout_seconds: int = 30,
        follow_redirects: bool = True,
    ) -> dict:
        """Send a raw HTTP request through the Go scanner."""
        self._ensure_connected()

        proto_req = build_raw_http_request(
            url=url, method=method, body=body,
            headers=headers, cookies=cookies,
            timeout_seconds=timeout_seconds,
            follow_redirects=follow_redirects,
        )

        logger.debug(f"Raw request: {method} {url}")

        try:
            proto_result = await self._stub.SendRawRequest(
                proto_req, timeout=float(timeout_seconds + 10),
            )
            return raw_response_to_dict(proto_result)
        except grpc.aio.AioRpcError as e:
            raise ScannerConnectionError(
                message=f"Raw request failed: {method} {url}",
                details=f"{e.code()} — {e.details()}",
            )

    async def abort_all(self, reason: str = "user request", force: bool = False) -> dict:
        """Stop all running attacks immediately."""
        self._ensure_connected()

        logger.warning(f"Aborting all attacks: {reason} (force={force})")

        try:
            resp = await self._stub.AbortAll(
                pb.AbortRequest(reason=reason, force=force),
                timeout=10.0,
            )
            result = {
                "success": resp.success,
                "attacks_aborted": resp.attacks_aborted,
                "requests_cancelled": resp.requests_cancelled,
                "message": resp.message,
            }
            logger.info(
                f"Abort complete: {resp.attacks_aborted} attacks stopped, "
                f"{resp.requests_cancelled} requests cancelled"
            )
            return result
        except grpc.aio.AioRpcError as e:
            raise ScannerConnectionError(
                message="Failed to abort attacks",
                details=f"{e.code()} — {e.details()}",
            )

    # ===================================================================
    # Properties
    # ===================================================================

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def address(self) -> str:
        return self._address
