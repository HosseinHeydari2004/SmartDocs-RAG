"""Thin HTTP client for the SmartDocs-RAG FastAPI backend (all errors -> ApiError)."""
from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_URL = "http://localhost:8000"


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 0, request_id: str | None = None, details: Any = None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.request_id, self.details = request_id, details


class ApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("SMARTDOCS_API_URL") or DEFAULT_URL).rstrip("/")

    def _request(self, method: str, path: str, *, timeout: float = 30, **kwargs) -> Any:
        try:
            resp = requests.request(method, self.base_url + path, timeout=timeout, **kwargs)
        except requests.ConnectionError as exc:
            raise ApiError("API_UNREACHABLE", "Cannot connect to the API.", details={"url": self.base_url}) from exc
        except requests.Timeout as exc:
            raise ApiError("API_TIMEOUT", "The API did not answer in time.", details={"url": self.base_url}) from exc
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:
                body = {}
            raise ApiError(
                body.get("error_code", f"HTTP_{resp.status_code}"),
                body.get("message", "Request failed."),
                resp.status_code,
                body.get("request_id"),
                body.get("details"),
            )
        return resp.json()

    # system
    def health(self): return self._request("GET", "/health", timeout=5)
    def stats(self): return self._request("GET", "/api/v1/stats", timeout=20)
    def settings_status(self): return self._request("GET", "/api/v1/settings/status", timeout=10)

    # documents / jobs
    def upload(self, scanned: bool, files: list[tuple[str, bytes, str]], on_duplicate: str = "skip", ocr_lang: str = "eng"):
        path = "/api/v1/documents/upload-scanned" if scanned else "/api/v1/documents/upload"
        data = {"on_duplicate": on_duplicate}
        if scanned:
            data["ocr_lang"] = ocr_lang
        return self._request("POST", path, timeout=180, data=data, files=[("files", f) for f in files])

    def job(self, job_id: str): return self._request("GET", f"/api/v1/jobs/{job_id}", timeout=10)
    def cancel_job(self, job_id: str): return self._request("POST", f"/api/v1/jobs/{job_id}/cancel", timeout=10)
    def documents(self): return self._request("GET", "/api/v1/documents", timeout=30)
    def delete_document(self, doc_id: str): return self._request("DELETE", f"/api/v1/documents/{doc_id}", timeout=30)
    def clear_documents(self): return self._request("DELETE", "/api/v1/documents", timeout=60, params={"confirm": "true"})

    # chat
    def chat(self, question: str, *, history=None, k=None, auto_filter=True, manual_filter=None):
        body = {"question": question, "history": history or [], "auto_filter": auto_filter}
        if k:
            body["k"] = k
        if manual_filter:
            body["manual_filter"] = manual_filter
        return self._request("POST", "/api/v1/chat", timeout=240, json=body)

    # settings
    def set_llm(self, api_key: str | None, model: str | None):
        body = {k: v for k, v in {"api_key": api_key, "model": model}.items() if v}
        return self._request("PUT", "/api/v1/settings/llm", timeout=10, json=body)

    def clear_key(self): return self._request("DELETE", "/api/v1/settings/llm/key", timeout=10)
    def test_llm(self): return self._request("POST", "/api/v1/settings/llm/test", timeout=90)
