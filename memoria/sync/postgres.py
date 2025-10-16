"""PostgreSQL-based synchronization backend using LISTEN/NOTIFY."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from queue import Empty
from typing import Any

from loguru import logger

from .base import NullSubscription, SyncBackend, SyncEvent, SyncSubscription

try:  # pragma: no cover - optional dependency resolution
    import psycopg  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - handled during backend creation
    psycopg = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency resolution
    import psycopg2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - handled during backend creation
    psycopg2 = None  # type: ignore[assignment]


def _is_closed(connection: Any) -> bool:
    closed = getattr(connection, "closed", False)
    if isinstance(closed, bool):
        return closed
    if isinstance(closed, int):
        return closed != 0
    return False


def _set_autocommit(connection: Any) -> None:
    try:
        connection.autocommit = True
    except Exception:  # pragma: no cover - defensive
        try:
            connection.set_session(autocommit=True)  # type: ignore[attr-defined]
        except Exception:
            logger.opt(exception=True).debug(
                "PostgresSyncBackend could not enable autocommit on connection",
            )


class _PostgresSubscription(SyncSubscription):
    """LISTEN to a PostgreSQL channel and feed messages to a handler."""

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        channel: str,
        handler: Callable[[SyncEvent], None],
    ) -> None:
        self._connection_factory = connection_factory
        self._channel = channel
        self._handler = handler
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread_started = False
        self._connection: Any | None = None
        self._cursor: Any | None = None

    def start(self) -> None:
        try:
            connection = self._connection_factory()
        except Exception:
            logger.opt(exception=True).warning(
                "PostgresSyncBackend failed to create LISTEN connection",
            )
            raise
        self._connection = connection
        try:
            cursor = connection.cursor()
            cursor.execute(f"LISTEN {self._channel};")
        except Exception:
            logger.opt(exception=True).warning(
                "PostgresSyncBackend failed to LISTEN on channel '%s'",
                self._channel,
            )
            try:
                connection.close()
            except Exception:
                logger.opt(exception=True).debug(
                    "Error closing PostgreSQL listener connection",
                )
            raise
        self._cursor = cursor
        self._thread_started = True
        self._thread.start()

    def _iter_notifications(self) -> Iterable[Any]:
        connection = self._connection
        if connection is None:
            return []
        notifies = getattr(connection, "notifies", None)
        if hasattr(notifies, "get"):
            while not self._closed.is_set():
                try:
                    notify = notifies.get(timeout=1.0)
                except Empty:
                    continue
                except Exception:
                    logger.opt(exception=True).debug(
                        "Postgres notification queue raised an exception",
                    )
                    break
                if notify is not None:
                    yield notify
            return []

        try:
            import select

            while not self._closed.is_set():
                try:
                    ready, _, _ = select.select([connection], [], [], 1.0)
                except Exception:
                    logger.opt(exception=True).debug(
                        "Postgres select() call failed while waiting for notifications",
                    )
                    break
                if not ready:
                    continue
                try:
                    poll = getattr(connection, "poll", None)
                    if callable(poll):
                        poll()
                except Exception:
                    logger.opt(exception=True).debug(
                        "Postgres connection poll() failed",
                    )
                    break
                notifications = getattr(connection, "notifies", None)
                if isinstance(notifications, list):
                    while notifications:
                        yield notifications.pop(0)
        except ImportError:  # pragma: no cover - select is part of stdlib
            logger.debug(
                "select module missing; cannot wait for PostgreSQL notifications"
            )

        return []

    def _run(self) -> None:
        for notify in self._iter_notifications():
            if self._closed.is_set():
                break
            payload = getattr(notify, "payload", None)
            if payload is None:
                continue
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", "ignore")
            if not isinstance(payload, str):
                continue
            try:
                decoded = json.loads(payload)
                event = SyncEvent.from_dict(decoded)
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to decode PostgreSQL sync payload",
                )
                continue
            try:
                self._handler(event)
            except Exception:  # pragma: no cover - defensive subscriber handling
                logger.opt(exception=True).warning("Sync handler raised an exception")

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._thread_started:
            self._thread.join(timeout=1.0)
        if self._cursor is not None:
            try:
                self._cursor.close()
            except Exception:
                logger.opt(exception=True).debug("Error closing PostgreSQL cursor")
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                logger.opt(exception=True).debug(
                    "Error closing PostgreSQL listener connection",
                )
        self._cursor = None
        self._connection = None


class PostgresSyncBackend(SyncBackend):
    """Publish/subscribe backend backed by PostgreSQL NOTIFY."""

    def __init__(
        self,
        dsn: str,
        *,
        channel: str = "memoria_sync",
        table: str | None = "memoria_sync_events",
        connect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if (
            psycopg is None and psycopg2 is None
        ):  # pragma: no cover - optional dependency
            raise RuntimeError(
                "psycopg (v3) or psycopg2 is required to use PostgresSyncBackend."
            )
        if not dsn:
            raise ValueError("PostgresSyncBackend requires a PostgreSQL DSN")
        self._dsn = dsn
        self._connect_kwargs = dict(connect_kwargs or {})
        self._channel = self._validate_channel(channel or "memoria_sync")
        self._table = self._validate_table(table)
        self._subscriptions: list[_PostgresSubscription] = []
        self._lock = threading.Lock()
        self._driver = psycopg if psycopg is not None else psycopg2
        self._publisher_conn = self._create_connection()
        self._ensure_table()

    @staticmethod
    def _validate_channel(channel: str) -> str:
        if not channel:
            raise ValueError("PostgresSyncBackend requires a channel name")
        if not all(part.isidentifier() for part in channel.split(".")):
            raise ValueError("Postgres channel names must be valid identifiers")
        return channel

    @staticmethod
    def _validate_table(table: str | None) -> str | None:
        if table is None:
            return None
        segments = table.split(".")
        if not all(
            segment and segment.replace("_", "").isalnum() for segment in segments
        ):
            raise ValueError(
                "Postgres table names must be alphanumeric with optional underscores"
            )
        return table

    def _create_connection(self) -> Any:
        try:
            connection = self._driver.connect(self._dsn, **self._connect_kwargs)
        except Exception as exc:  # pragma: no cover - connection failures
            raise RuntimeError(f"Failed to connect to PostgreSQL: {exc}") from exc
        _set_autocommit(connection)
        return connection

    def _get_publisher_connection(self) -> Any:
        if self._publisher_conn is None or _is_closed(self._publisher_conn):
            self._publisher_conn = self._create_connection()
        return self._publisher_conn

    def _ensure_table(self) -> None:
        if not self._table:
            return
        try:
            with self._publisher_conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload JSONB NOT NULL
                    )
                    """
                )
        except Exception:  # pragma: no cover - table creation guard
            logger.opt(exception=True).warning(
                "Failed to ensure existence of PostgreSQL sync table '%s'",
                self._table,
            )

    def publish(self, event: SyncEvent) -> None:
        payload = json.dumps(event.to_dict())
        connection: Any | None = None
        try:
            connection = self._get_publisher_connection()
            with connection.cursor() as cursor:
                if self._table:
                    cursor.execute(
                        f"INSERT INTO {self._table} (payload) VALUES (%s)",
                        (payload,),
                    )
                cursor.execute(
                    f"NOTIFY {self._channel}, %s",
                    (payload,),
                )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to publish sync event via PostgreSQL",
            )
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    logger.opt(exception=True).debug(
                        "Error rolling back failed PostgreSQL publish",
                    )

    def subscribe(self, handler: Callable[[SyncEvent], None]) -> SyncSubscription:
        def factory() -> Any:
            return self._create_connection()

        subscription = _PostgresSubscription(factory, self._channel, handler)
        try:
            subscription.start()
        except Exception:
            return NullSubscription()
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def close(self) -> None:
        with self._lock:
            subscriptions = list(self._subscriptions)
            self._subscriptions.clear()
        for subscription in subscriptions:
            try:
                subscription.close()
            except Exception:  # pragma: no cover - shutdown guard
                logger.opt(exception=True).debug(
                    "Error while closing PostgreSQL subscription",
                )
        if self._publisher_conn is not None:
            try:
                self._publisher_conn.close()
            except Exception:  # pragma: no cover - shutdown guard
                logger.opt(exception=True).debug(
                    "Error while closing PostgreSQL publisher connection",
                )
        self._publisher_conn = None
