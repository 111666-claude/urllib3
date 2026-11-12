"""
Offline, fake-based tests for the redirect machinery in PoolManager /
ProxyManager / HTTPConnectionPool / HTTPResponse.

No sockets and no public network are used: ConnectionPool.urlopen() is
replaced by an in-memory fake that returns scripted responses, and response
bodies are real urllib3 HTTPResponse objects backed by BytesIO streams.

Covered behaviour:
  * same-origin decision for 301/302/303/307/308 (scheme/host/port, relative
    locations, default ports, case folding);
  * Authorization / Cookie / Proxy-Authorization forwarding (stripped only
    cross-origin, honoring remove_headers_on_redirect);
  * ProxyManager Host / proxy header recomputation on every hop;
  * request-body replay: 307/308 keep and rewind, 303 drops, 301/302 keep
    POST bodies, unrewindable bodies raise UnrewindableBodyError;
  * redirect budget: loops and chains are bounded by Retry(total/redirect),
    raise_on_redirect=False returns the last response;
  * short-read response bodies: drain_conn() must not raise and must be
    invoked on every hop, read() surfaces ProtocolError/IncompleteRead.
"""

from __future__ import annotations

import io
import unittest
from typing import Any
from urllib.parse import urlparse

from urllib3 import PoolManager, ProxyManager
from urllib3._collections import HTTPHeaderDict
from urllib3.exceptions import (
    IncompleteRead,
    MaxRetryError,
    ProtocolError,
    UnrewindableBodyError,
)
from urllib3.response import HTTPResponse
from urllib3.util.retry import Retry
from urllib3.util.request import set_file_position

REDIRECT_STATUSES = (301, 302, 303, 307, 308)


class BadTellObject(io.BytesIO):
    """File body whose tell() fails, so no rewind position can be recorded."""

    def tell(self) -> Any:
        raise OSError


class NoSeekObject(io.BytesIO):
    """File body that does not support seek()."""

    seek = None  # type: ignore[assignment]


class FakeResponse:
    """Minimal stand-in for urllib3's HTTPResponse on the redirect path."""

    def __init__(
        self,
        status: int,
        location: str | None = None,
        headers: dict[str, str] | None = None,
        retries: Retry | None = None,
        drain_raises: BaseException | None = None,
    ) -> None:
        self.status = status
        self.headers = HTTPHeaderDict(headers or {})
        if location is not None:
            self.headers["location"] = location
        self.retries = retries if retries is not None else Retry(3)
        self.drain_calls = 0
        self._drain_raises = drain_raises

    def get_redirect_location(self) -> str | None | bool:
        if self.status in REDIRECT_STATUSES:
            return self.headers.get("location")
        return False

    def drain_conn(self) -> None:
        self.drain_calls += 1
        if self._drain_raises is not None:
            raise self._drain_raises


class RecordedRequest:
    def __init__(self, pool: "FakePool", method: str, url: str, kw: dict[str, Any]):
        self.pool = pool
        self.method = method
        self.url = url
        self.kw = kw
        self.headers = HTTPHeaderDict(kw["headers"])
        # Mimic HTTPConnectionPool.urlopen(): merge pool-level proxy headers.
        if pool.proxy is not None and pool.proxy_headers:
            merged = HTTPHeaderDict(pool.proxy_headers)
            merged.update(self.headers)
            self.merged_headers = merged
        else:
            self.merged_headers = self.headers
        self.body = kw.get("body")
        self.body_pos = kw.get("body_pos")
        self.chunked = kw.get("chunked", False)
        # The bytes the fake connection actually put on the wire; populated by
        # FakePool after the stream is consumed.
        self.sent_body = b""

    def body_bytes(self) -> bytes:
        # Report what was actually transmitted, not a re-read of the (possibly
        # exhausted) body object.
        return self.sent_body


class FakePool:
    """In-memory replacement for HTTPConnectionPool/HTTPSConnectionPool.

    Faithfully models the request-body lifecycle of a real pool/connection:
    ``set_file_position`` rewinds to a supplied position (or records one on
    the first send), and the fake connection then *consumes* the stream just
    as a socket would. A later hop with no usable position therefore observes
    an exhausted (empty) body, matching real cross-pool redirect behaviour.
    """

    def __init__(self, host: str, port: int | None, scheme: str, **kw: Any) -> None:
        self.host = host
        self.port = port
        self.scheme = scheme
        self.proxy = kw.get("_proxy")
        self.proxy_headers = kw.get("_proxy_headers") or {}
        self.requests: list[RecordedRequest] = []
        self.handler: Any = None

    def urlopen(self, method: str, url: str, **kw: Any) -> Any:
        body = kw.get("body")
        # Mirror the real pool's body-position handling so replay semantics
        # (rewind / UnrewindableBodyError) are exercised faithfully. A real
        # pool raises during urlopen(), so let the exception escape instead
        # of recording the request.
        kw["body_pos"] = set_file_position(body, kw.get("body_pos"))
        recorded = RecordedRequest(self, method, url, kw)
        # Simulate the connection consuming the body off the wire.
        recorded.sent_body = _consume_body(body)
        self.requests.append(recorded)
        assert self.handler is not None, "no response handler installed"
        response = self.handler(recorded)
        # A real pool stamps the active Retry onto the response so that
        # PoolManager.urlopen() picks it up on the next redirect hop.
        if kw.get("retries") is not None:
            response.retries = kw["retries"]
        return response


def _consume_body(body: Any) -> bytes:
    """Read a request body the way a real connection sends it, leaving the
    stream positioned at EOF for file-like objects."""
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode()
    if hasattr(body, "read"):
        data = body.read()
        return data.encode() if isinstance(data, str) else data
    chunks = []
    for chunk in body:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(chunks)


class FakePoolManager(PoolManager):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.fake_pools: dict[tuple[str, str, int | None], FakePool] = {}
        self.router: Any = None

    def connection_from_host(
        self,
        host: str | None,
        port: int | None = None,
        scheme: str | None = "http",
        pool_kwargs: dict[str, Any] | None = None,
    ) -> FakePool:
        assert host
        scheme = scheme or "http"
        key = (scheme, host, port)
        pool = self.fake_pools.get(key)
        if pool is None:
            ctx = self._merge_pool_kwargs(pool_kwargs)
            pool = FakePool(host, port, scheme, **ctx)
            pool.handler = lambda recorded, mgr=self: mgr.router(recorded)
            self.fake_pools[key] = pool
        return pool


class FakeProxyManager(ProxyManager):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.fake_pools = {}
        self.router: Any = None

    def connection_from_host(
        self,
        host: str | None,
        port: int | None = None,
        scheme: str | None = "http",
        pool_kwargs: dict[str, Any] | None = None,
    ) -> FakePool:
        assert host
        scheme = scheme or "http"
        key = (scheme, host, port)
        pool = self.fake_pools.get(key)
        if pool is None:
            ctx = self._merge_pool_kwargs(pool_kwargs)
            pool = FakePool(host, port, scheme, **ctx)
            pool.handler = lambda recorded, mgr=self: mgr.router(recorded)
            self.fake_pools[key] = pool
        return pool


def make_manager(router: Any) -> FakePoolManager:
    manager = FakePoolManager()
    manager.router = router
    return manager


def redirect_router(routes: dict[str, tuple[int, str]], final_status: int = 200):
    """routes: request-URI path -> (status, absolute or relative location)."""

    def router(recorded: RecordedRequest) -> FakeResponse:
        parsed = urlparse(recorded.url if "://" in recorded.url else "http://x" + recorded.url)
        path = parsed.path
        if path in routes:
            status, location = routes[path]
            return FakeResponse(status, location=location)
        return FakeResponse(final_status)

    return router


class SameOriginTests(unittest.TestCase):
    def _manager_with_chain(self, first_location: str) -> FakePoolManager:
        def router(recorded: RecordedRequest) -> FakeResponse:
            if recorded.url in (
                "http://example.com/start",
                "/start",
            ) or recorded.url.startswith("http://example.com/start"):
                return FakeResponse(302, location=first_location)
            return FakeResponse(200)

        manager = make_manager(router)
        return manager

    def test_relative_location_keeps_credentials(self) -> None:
        manager = make_manager(redirect_router({"/start": (302, "/next")}))
        r = manager.urlopen(
            "GET",
            "http://example.com/start",
            headers=HTTPHeaderDict({"Authorization": "tok", "Cookie": "a=b"}),
        )
        self.assertEqual(r.status, 200)
        last = list(manager.fake_pools.values())[0].requests[-1]
        self.assertEqual(last.headers.get("Authorization"), "tok")
        self.assertEqual(last.headers.get("Cookie"), "a=b")

    def test_same_host_different_port_strips_credentials(self) -> None:
        manager = self._manager_with_chain("http://example.com:8080/next")
        manager.urlopen(
            "GET",
            "http://example.com/start",
            headers={"Authorization": "tok", "Cookie": "a=b", "Proxy-Authorization": "p"},
        )
        last = self._last_request(manager)
        self.assertIsNone(last.headers.get("Authorization"))
        self.assertIsNone(last.headers.get("Cookie"))
        self.assertIsNone(last.headers.get("Proxy-Authorization"))

    def test_scheme_change_strips_credentials(self) -> None:
        manager = self._manager_with_chain("https://example.com/next")
        manager.urlopen(
            "GET",
            "http://example.com/start",
            headers={"Authorization": "tok", "Cookie": "a=b"},
        )
        last = self._last_request(manager)
        self.assertIsNone(last.headers.get("Authorization"))
        self.assertIsNone(last.headers.get("Cookie"))

    def test_default_port_is_same_origin(self) -> None:
        manager = self._manager_with_chain("http://example.com:80/next")
        manager.urlopen(
            "GET",
            "http://example.com/start",
            headers={"Authorization": "tok"},
        )
        last = self._last_request(manager)
        self.assertEqual(last.headers.get("Authorization"), "tok")

    def test_case_folded_host_is_same_origin(self) -> None:
        manager = self._manager_with_chain("http://EXAMPLE.com/next")
        manager.urlopen(
            "GET",
            "http://example.com/start",
            headers={"Authorization": "tok"},
        )
        last = self._last_request(manager)
        self.assertEqual(last.headers.get("Authorization"), "tok")

    def test_cross_host_keeps_unrelated_headers(self) -> None:
        manager = self._manager_with_chain("http://other.example/next")
        manager.urlopen(
            "GET",
            "http://example.com/start",
            headers={"Authorization": "tok", "X-Trace-Id": "abc"},
        )
        last = self._last_request(manager)
        self.assertIsNone(last.headers.get("Authorization"))
        self.assertEqual(last.headers.get("X-Trace-Id"), "abc")

    def test_all_redirect_statuses_followed(self) -> None:
        for status in REDIRECT_STATUSES:
            manager = make_manager(
                redirect_router({"/start": (status, "http://example.com/next")})
            )
            r = manager.urlopen("GET", "http://example.com/start")
            self.assertEqual(r.status, 200, msg=f"status {status} not followed")
            self.assertEqual(
                len(list(manager.fake_pools.values())[0].requests), 2
            )

    @staticmethod
    def _last_request(manager: FakePoolManager) -> RecordedRequest:
        requests = []
        for pool in manager.fake_pools.values():
            requests.extend(pool.requests)
        return requests[-1]


class CredentialForwardingTests(unittest.TestCase):
    def test_credentials_kept_same_origin_all_statuses(self) -> None:
        for status in REDIRECT_STATUSES:
            manager = make_manager(
                redirect_router({"/start": (status, "/next")})
            )
            r = manager.urlopen(
                "POST",
                "http://example.com/start",
                headers={
                    "Authorization": "tok",
                    "Cookie": "a=b",
                    "Proxy-Authorization": "p",
                },
                body=b"data",
            )
            self.assertEqual(r.status, 200)
            last = list(manager.fake_pools.values())[0].requests[-1]
            self.assertEqual(last.headers.get("Authorization"), "tok", status)
            self.assertEqual(last.headers.get("Cookie"), "a=b", status)
            self.assertEqual(last.headers.get("Proxy-Authorization"), "p", status)

    def test_custom_remove_headers_on_redirect_honored(self) -> None:
        manager = make_manager(
            redirect_router({"/start": (302, "http://other.example/next")})
        )
        manager.urlopen(
            "GET",
            "http://example.com/start",
            headers={"Authorization": "tok", "X-API-Secret": "s"},
            retries=Retry(remove_headers_on_redirect=["X-API-Secret"]),
        )
        last = SameOriginTests._last_request(manager)
        self.assertEqual(last.headers.get("Authorization"), "tok")
        self.assertIsNone(last.headers.get("X-API-Secret"))

    def test_empty_remove_headers_keeps_everything(self) -> None:
        manager = make_manager(
            redirect_router({"/start": (302, "http://other.example/next")})
        )
        manager.urlopen(
            "GET",
            "http://example.com/start",
            headers={
                "Authorization": "tok",
                "Cookie": "a=b",
                "Proxy-Authorization": "p",
            },
            retries=Retry(remove_headers_on_redirect=[]),
        )
        last = SameOriginTests._last_request(manager)
        self.assertEqual(last.headers.get("Authorization"), "tok")
        self.assertEqual(last.headers.get("Cookie"), "a=b")
        self.assertEqual(last.headers.get("Proxy-Authorization"), "p")

    def test_none_headers_does_not_crash(self) -> None:
        manager = make_manager(
            redirect_router({"/start": (302, "http://other.example/next")})
        )
        r = manager.urlopen(
            "GET", "http://example.com/start", headers=None
        )
        self.assertEqual(r.status, 200)


class ProxyHeaderTests(unittest.TestCase):
    def test_host_recomputed_for_every_hop(self) -> None:
        def router(recorded: RecordedRequest) -> FakeResponse:
            host = recorded.pool.host
            if host == "a.example":
                return FakeResponse(302, location="http://b.example/final")
            return FakeResponse(200)

        manager = FakeProxyManager(
            "http://proxy.example:3128",
            proxy_headers={"Proxy-Authorization": "proxytok"},
        )
        manager.router = router
        manager.urlopen(
            "GET",
            "http://a.example/start",
            headers={"Host": "a.example"},
        )
        requests = [
            req
            for pool in manager.fake_pools.values()
            for req in pool.requests
        ]
        self.assertEqual(len(requests), 2)
        # ProxyManager._set_proxy_headers derives Host from each request URL.
        self.assertEqual(requests[0].headers.get("Host"), "a.example")
        self.assertEqual(requests[1].headers.get("Host"), "b.example")

    def test_stale_host_header_cannot_override_new_hop(self) -> None:
        def router(recorded: RecordedRequest) -> FakeResponse:
            if recorded.pool.host == "a.example":
                return FakeResponse(302, location="http://b.example/final")
            return FakeResponse(200)

        manager = FakeProxyManager("http://proxy.example:3128")
        manager.router = router
        manager.urlopen(
            "GET",
            "http://a.example/start",
            headers={"Host": "stale.example"},
        )
        requests = [
            req
            for pool in manager.fake_pools.values()
            for req in pool.requests
        ]
        self.assertEqual(requests[1].headers.get("Host"), "b.example")
        self.assertNotEqual(requests[1].headers.get("Host"), "stale.example")

    def test_proxy_authorization_stripped_cross_origin_but_pool_header_remains(
        self,
    ) -> None:
        def router(recorded: RecordedRequest) -> FakeResponse:
            if recorded.pool.host == "a.example":
                return FakeResponse(302, location="http://b.example/final")
            return FakeResponse(200)

        manager = FakeProxyManager(
            "http://proxy.example:3128",
            proxy_headers={"Proxy-Authorization": "pool-level"},
        )
        manager.router = router
        manager.urlopen(
            "GET",
            "http://a.example/start",
            headers={"Proxy-Authorization": "per-request"},
        )
        requests = [
            req
            for pool in manager.fake_pools.values()
            for req in pool.requests
        ]
        # The per-request credential is dropped cross-origin ...
        self.assertNotIn(
            "per-request", requests[1].headers.get("Proxy-Authorization", "")
        )
        # ... but the pool-level proxy credential is re-merged by the pool.
        self.assertEqual(
            requests[1].merged_headers.get("Proxy-Authorization"), "pool-level"
        )

    def test_accept_added_when_absent(self) -> None:
        manager = FakeProxyManager("http://proxy.example:3128")
        manager.router = lambda recorded: FakeResponse(200)
        manager.urlopen("GET", "http://a.example/start")
        req = next(iter(manager.fake_pools.values())).requests[0]
        self.assertEqual(req.headers.get("Accept"), "*/*")


class BodyReplayTests(unittest.TestCase):
    def test_307_and_308_replay_seekable_body(self) -> None:
        for status in (307, 308):
            manager = make_manager(
                redirect_router({"/start": (status, "/next")})
            )
            body = io.BytesIO(b"the request body")
            r = manager.urlopen(
                "PUT",
                "http://example.com/start",
                body=body,
                headers={"Content-Length": "17"},
            )
            self.assertEqual(r.status, 200)
            reqs = list(manager.fake_pools.values())[0].requests
            self.assertEqual(reqs[0].method, "PUT")
            self.assertEqual(reqs[1].method, "PUT")
            self.assertEqual(reqs[0].body_bytes(), b"the request body")
            self.assertEqual(reqs[1].body_bytes(), b"the request body")
            # The second hop was rewound to the recorded starting position
            # before its stream was consumed.
            self.assertEqual(reqs[1].body_pos, 0)
            # The fake connection consumed the body again, leaving it at EOF.
            self.assertEqual(reqs[1].body.tell(), len(b"the request body"))

    def test_307_replays_body_across_separate_pools(self) -> None:
        # PoolManager follows redirects through *separate* ConnectionPool
        # urlopen() calls that do not share a body-position local. The body
        # stream is exhausted after the first send, so PoolManager must record
        # a rewind position itself; otherwise the second host receives b"".
        def router(recorded: RecordedRequest) -> FakeResponse:
            if recorded.pool.host == "a.example":
                return FakeResponse(307, location="http://b.example/dest")
            return FakeResponse(200)

        manager = make_manager(router)
        payload = b"cross-pool-payload"
        body = io.BytesIO(payload)
        r = manager.urlopen(
            "PUT",
            "http://a.example/start",
            body=body,
            headers={"Content-Length": str(len(payload))},
        )
        self.assertEqual(r.status, 200)
        by_host = {
            pool.host: pool.requests for pool in manager.fake_pools.values()
        }
        self.assertEqual(len(by_host["a.example"]), 1)
        self.assertEqual(len(by_host["b.example"]), 1)
        self.assertEqual(by_host["a.example"][0].body_bytes(), payload)
        # The decisive assertion: the second pool got the full body, not b"".
        self.assertEqual(by_host["b.example"][0].body_bytes(), payload)
        self.assertEqual(by_host["b.example"][0].body_pos, 0)

    def test_301_302_keep_post_method_and_body_in_poolmanager(self) -> None:
        # PoolManager only changes the method on 303. On 301/302 the body is
        # replayed so a non-idempotent POST can be retried verbatim.
        for status in (301, 302):
            manager = make_manager(
                redirect_router({"/start": (status, "/next")})
            )
            r = manager.urlopen(
                "POST",
                "http://example.com/start",
                body=b"payload",
                headers={"Content-Length": "7"},
            )
            self.assertEqual(r.status, 200)
            reqs = list(manager.fake_pools.values())[0].requests
            self.assertEqual(reqs[1].method, "POST")
            self.assertEqual(reqs[1].body_bytes(), b"payload")

    def test_303_drops_body_and_switches_to_get(self) -> None:
        manager = make_manager(
            redirect_router({"/start": (303, "/next")})
        )
        r = manager.urlopen(
            "POST",
            "http://example.com/start",
            body=io.BytesIO(b"sensitive"),
            headers={
                "Content-Length": "9",
                "Content-Type": "application/octet-stream",
                "Transfer-Encoding": "chunked",
            },
            chunked=True,
            body_pos=0,
        )
        self.assertEqual(r.status, 200)
        reqs = list(manager.fake_pools.values())[0].requests
        redirected = reqs[1]
        self.assertEqual(redirected.method, "GET")
        self.assertIsNone(redirected.body)
        self.assertFalse(redirected.chunked)
        self.assertIsNone(redirected.body_pos)
        self.assertIsNone(redirected.headers.get("Content-Length"))
        self.assertIsNone(redirected.headers.get("Content-Type"))
        self.assertIsNone(redirected.headers.get("Transfer-Encoding"))

    def test_303_with_failed_tell_does_not_try_to_rewind(self) -> None:
        manager = make_manager(
            redirect_router({"/start": (303, "/next")})
        )
        # Must not raise UnrewindableBodyError: the body is dropped on 303.
        r = manager.urlopen(
            "PUT",
            "http://example.com/start",
            body=BadTellObject(b"the data"),
            headers={"Content-Length": "8"},
        )
        self.assertEqual(r.status, 200)

    def test_307_with_failed_tell_cannot_replay_body(self) -> None:
        # A body whose tell() raises cannot have a rewind position recorded.
        # PoolManager must not blow up on the first request or on a 303, and
        # on a 307 the body simply cannot be replayed (the connection pool keeps
        # retrying tell() rather than seeking, so no body is re-sent).
        manager = make_manager(
            redirect_router({"/start": (307, "/next")})
        )
        r = manager.urlopen(
            "PUT",
            "http://example.com/start",
            body=BadTellObject(b"the data"),
            headers={"Content-Length": "8"},
        )
        self.assertEqual(r.status, 200)
        reqs = list(manager.fake_pools.values())[0].requests
        self.assertEqual(reqs[0].body_bytes(), b"the data")
        self.assertEqual(reqs[1].body_bytes(), b"")

    def test_307_with_unseekable_body_and_known_pos_raises(self) -> None:
        # When a starting position *is* known but the body cannot seek to it,
        # the connection pool raises UnrewindableBodyError on the replay hop.
        manager = make_manager(
            redirect_router({"/start": (307, "/next")})
        )
        with self.assertRaises(UnrewindableBodyError):
            manager.urlopen(
                "PUT",
                "http://example.com/start",
                body=NoSeekObject(b"the data"),
                headers={"Content-Length": "8"},
                body_pos=0,
            )


class RedirectBudgetTests(unittest.TestCase):
    def test_redirect_loop_is_bounded(self) -> None:
        def router(recorded: RecordedRequest) -> FakeResponse:
            if recorded.url.endswith("/a"):
                return FakeResponse(302, location="http://example.com/b")
            return FakeResponse(302, location="http://example.com/a")

        manager = make_manager(router)
        with self.assertRaises(MaxRetryError):
            manager.urlopen(
                "GET",
                "http://example.com/a",
                retries=Retry(redirect=3, total=None),
            )
        # 1 initial request + 3 redirects, then the 4th increment fails.
        reqs = [
            req
            for pool in manager.fake_pools.values()
            for req in pool.requests
        ]
        self.assertEqual(len(reqs), 4)

    def test_chain_budget(self) -> None:
        manager = make_manager(
            redirect_router(
                {
                    "/r0": (302, "/r1"),
                    "/r1": (302, "/r2"),
                    "/r2": (302, "/r3"),
                }
            )
        )
        with self.assertRaises(MaxRetryError):
            manager.urlopen(
                "GET", "http://example.com/r0", retries=Retry(redirect=2)
            )

    def test_raise_on_redirect_false_returns_last_response(self) -> None:
        manager = make_manager(redirect_router({"/a": (302, "/b")}))
        r = manager.urlopen(
            "GET",
            "http://example.com/a",
            retries=Retry(redirect=0, raise_on_redirect=False),
        )
        self.assertEqual(r.status, 302)

    def test_integer_retries_budget(self) -> None:
        manager = make_manager(redirect_router({"/a": (307, "/b")}))
        with self.assertRaises(MaxRetryError):
            manager.urlopen("GET", "http://example.com/a", retries=0)

    def test_history_records_every_hop(self) -> None:
        manager = make_manager(
            redirect_router({"/a": (302, "/b"), "/b": (302, "/c")})
        )
        with self.assertRaises(MaxRetryError):
            manager.urlopen(
                "GET",
                "http://example.com/a",
                retries=Retry(redirect=1, total=None),
            )
        # The Retry object carried by each response accumulates one
        # RequestHistory entry per *completed* redirect hop. With
        # redirect=1 the first increment succeeds (1 history entry) and the
        # second one raises MaxRetryError.
        reqs = [
            req
            for pool in manager.fake_pools.values()
            for req in pool.requests
        ]
        last_retry = reqs[-1].kw["retries"]
        self.assertEqual(len(last_retry.history), 1)
        self.assertEqual(last_retry.history[0].status, 302)
        self.assertEqual(last_retry.history[0].redirect_location, "/b")
        self.assertEqual(last_retry.redirect, 0)


class DrainAndShortReadTests(unittest.TestCase):
    def test_every_redirect_hop_drains_the_response(self) -> None:
        responses = []

        def router(recorded: RecordedRequest) -> FakeResponse:
            if recorded.url.endswith("/a"):
                resp = FakeResponse(302, location="/b")
            elif recorded.url.endswith("/b"):
                resp = FakeResponse(302, location="/c")
            else:
                return FakeResponse(200)
            responses.append(resp)
            return resp

        manager = make_manager(router)
        manager.urlopen("GET", "http://example.com/a")
        self.assertEqual([r.drain_calls for r in responses], [1, 1])

    def test_drain_swallows_short_read_on_redirect(self) -> None:
        # A real HTTPResponse with a truncated body: drain_conn() must absorb
        # the IncompleteRead so the redirect can proceed.
        short_fp = io.BytesIO(b"abc")
        short_response = HTTPResponse(
            body=short_fp,
            headers={"Content-Length": "10"},
            status=302,
            preload_content=False,
            enforce_content_length=True,
            request_method="GET",
        )
        short_response.headers["location"] = "/next"

        def router(recorded: RecordedRequest) -> HTTPResponse | FakeResponse:
            if recorded.url.endswith("/start"):
                return short_response
            return FakeResponse(200)

        manager = make_manager(router)
        r = manager.urlopen("GET", "http://example.com/start")
        self.assertEqual(r.status, 200)

    def test_short_read_surfaces_on_final_read(self) -> None:
        resp = HTTPResponse(
            body=io.BytesIO(b"abc"),
            headers={"Content-Length": "10"},
            status=200,
            preload_content=False,
            enforce_content_length=True,
            request_method="GET",
        )
        # A bounded read runs the Content-Length check in _raw_read().
        with self.assertRaises(ProtocolError) as ctx:
            resp.read(10)
        self.assertIsInstance(ctx.exception.args[1], IncompleteRead)

    def test_drain_conn_on_short_read_does_not_raise(self) -> None:
        resp = HTTPResponse(
            body=io.BytesIO(b"abc"),
            headers={"Content-Length": "10"},
            status=200,
            preload_content=False,
            enforce_content_length=True,
            request_method="GET",
        )
        # Must be a no-op wrt exceptions even though the body is truncated.
        resp.drain_conn()
        # And the underlying fp got closed.
        self.assertTrue(resp.isclosed())


if __name__ == "__main__":
    unittest.main()
