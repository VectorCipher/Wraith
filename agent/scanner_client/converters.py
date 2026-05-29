"""
WRAITH Proto ↔ Pydantic Converters

Translates between gRPC protobuf messages and WRAITH's internal
Pydantic data models. This is the single translation layer — no
other module should manually construct proto objects.

Directions:
    Pydantic → Proto  (outbound: Python sends to Go)
    Proto → Pydantic  (inbound:  Go sends back to Python)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure scanner_client package is importable for the generated stubs
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scanner_pb2 as pb  # noqa: E402

from models.attack_result import (
    AttackRequest as PydanticAttackRequest,
    AttackResult as PydanticAttackResult,
    AttackStatus,
    PayloadResult as PydanticPayloadResult,
)
from models.target import TechStack


# ===================================================================
# Outbound: Pydantic → Proto  (Python → Go)
# ===================================================================

def attack_request_to_proto(req: PydanticAttackRequest) -> pb.AttackRequest:
    """Convert a Pydantic AttackRequest to a proto AttackRequest."""
    return pb.AttackRequest(
        attack_id=req.attack_id,
        attack_type=req.attack_type,
        target_url=req.target_url,
        method=req.method,
        payloads=req.payloads,
        injection_point=req.injection_point,
        parameter_name=req.parameter_name,
        headers=req.headers,
        body=req.body or "",
        cookies=req.cookies,
        timeout_seconds=req.timeout_seconds,
        follow_redirects=req.follow_redirects,
        baseline_response_body=req.baseline_response or "",
    )


def build_crawl_request(
    target_url: str,
    max_depth: int = 3,
    max_pages: int = 100,
    timeout_seconds: int = 120,
    delay_between_ms: int = 100,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> pb.CrawlRequest:
    """Build a proto CrawlRequest from keyword args."""
    return pb.CrawlRequest(
        target_url=target_url,
        max_depth=max_depth,
        max_pages=max_pages,
        timeout_seconds=timeout_seconds,
        delay_between_ms=delay_between_ms,
        headers=headers or {},
        cookies=cookies or {},
    )


def build_fingerprint_request(
    target_url: str,
    timeout_seconds: int = 30,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> pb.FingerprintRequest:
    """Build a proto FingerprintRequest from keyword args."""
    return pb.FingerprintRequest(
        target_url=target_url,
        headers=headers or {},
        cookies=cookies or {},
        timeout_seconds=timeout_seconds,
    )


def build_baseline_request(
    url: str,
    method: str = "GET",
    body: str = "",
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    timeout_seconds: int = 30,
) -> pb.BaselineRequest:
    """Build a proto BaselineRequest from keyword args."""
    return pb.BaselineRequest(
        url=url,
        method=method,
        body=body,
        headers=headers or {},
        cookies=cookies or {},
        timeout_seconds=timeout_seconds,
    )


def build_raw_http_request(
    url: str,
    method: str = "GET",
    body: str = "",
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    timeout_seconds: int = 30,
    follow_redirects: bool = True,
) -> pb.RawHTTPRequest:
    """Build a proto RawHTTPRequest from keyword args."""
    return pb.RawHTTPRequest(
        url=url,
        method=method,
        body=body,
        headers=headers or {},
        cookies=cookies or {},
        timeout_seconds=timeout_seconds,
        follow_redirects=follow_redirects,
    )


# ===================================================================
# Inbound: Proto → Pydantic  (Go → Python)
# ===================================================================

def payload_result_from_proto(pr: pb.PayloadResult) -> PydanticPayloadResult:
    """Convert a proto PayloadResult to a Pydantic PayloadResult."""
    return PydanticPayloadResult(
        payload=pr.payload,
        status_code=pr.status_code,
        response_body=pr.response_body,
        response_headers=dict(pr.response_headers),
        response_time_ms=pr.response_time_ms,
        content_length=pr.content_length,
        error=pr.error if pr.error else None,
    )


def attack_result_from_proto(ar: pb.AttackResult) -> PydanticAttackResult:
    """Convert a proto AttackResult to a Pydantic AttackResult."""
    status_map = {
        "success": AttackStatus.SUCCESS,
        "failed": AttackStatus.FAILED,
        "timeout": AttackStatus.TIMEOUT,
        "blocked": AttackStatus.BLOCKED,
        "error": AttackStatus.ERROR,
    }

    return PydanticAttackResult(
        attack_id=ar.attack_id,
        attack_type=ar.attack_type,
        status=status_map.get(ar.status, AttackStatus.ERROR),
        payload_results=[
            payload_result_from_proto(pr) for pr in ar.payload_results
        ],
        total_requests=ar.total_requests,
        duration_ms=ar.duration_ms,
        error=ar.error if ar.error else None,
    )


def fingerprint_to_tech_stack(fp: pb.FingerprintResult) -> TechStack:
    """Convert a proto FingerprintResult to a Pydantic TechStack."""
    return TechStack(
        language=fp.language or None,
        framework=fp.framework or None,
        database=fp.database or None,
        web_server=fp.web_server or None,
        template_engine=fp.template_engine or None,
        other=list(fp.other_tech),
    )


def crawl_result_to_dict(cr: pb.CrawlResult) -> dict:
    """Convert a proto CrawlResult to a plain dict for memory storage."""
    return {
        "url": cr.url,
        "method": cr.method,
        "status_code": cr.status_code,
        "depth": cr.depth,
        "forms": [
            {
                "action": f.action,
                "method": f.method,
                "enctype": f.enctype,
                "fields": [
                    {
                        "name": ff.name,
                        "type": ff.field_type,
                        "value": ff.value,
                        "required": ff.required,
                    }
                    for ff in f.fields
                ],
            }
            for f in cr.forms
        ],
        "links": list(cr.links),
        "scripts": list(cr.scripts),
        "api_endpoints": list(cr.api_endpoints),
        "detected_tech": list(cr.detected_tech),
        "page_title": cr.page_title,
        "content_type": cr.content_type,
        "response_time_ms": cr.response_time_ms,
        "error": cr.error if cr.error else None,
    }


def baseline_response_to_dict(br: pb.BaselineResponse) -> dict:
    """Convert a proto BaselineResponse to a plain dict."""
    return {
        "status_code": br.status_code,
        "body": br.body,
        "headers": dict(br.headers),
        "content_length": br.content_length,
        "response_time_ms": br.response_time_ms,
        "content_type": br.content_type,
        "error": br.error if br.error else None,
    }


def raw_response_to_dict(rr: pb.RawHTTPResponse) -> dict:
    """Convert a proto RawHTTPResponse to a plain dict."""
    return {
        "status_code": rr.status_code,
        "body": rr.body,
        "headers": dict(rr.headers),
        "content_length": rr.content_length,
        "response_time_ms": rr.response_time_ms,
        "content_type": rr.content_type,
        "redirected": rr.redirected,
        "final_url": rr.final_url,
        "error": rr.error if rr.error else None,
    }
