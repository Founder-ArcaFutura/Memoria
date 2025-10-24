"""SQLAlchemy-based database manager for the Memoria 0.9 alpha (0.9.0a0) release.

Provides cross-database compatibility for SQLite, PostgreSQL, and MySQL
along with utilities for backing up and restoring databases. The manager
also exposes lightweight diagnostic helpers like :func:`get_db_size` and
``has_tables`` to detect obvious corruption (e.g., zero-byte files or
missing tables).
"""

import importlib.util
import json
import shutil
import ssl
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from loguru import logger
from sqlalchemy import JSON, and_, create_engine, func, inspect, or_, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from ..config.manager import ConfigManager
from ..schemas import PersonalMemoryDocument
from ..utils.embeddings import generate_embedding, vector_search_enabled
from ..utils.exceptions import DatabaseError
from ..utils.pydantic_compat import model_dump, model_dump_json_safe, model_validate
from ..utils.pydantic_models import (
    ProcessedLongTermMemory,
)
from . import migration
from .auto_creator import DatabaseAutoCreator
from .models import (
    Agent,
    Base,
    ChatHistory,
    LinkMemoryThread,
    LongTermMemory,
    MemoryAccessEvent,
    ShortTermMemory,
)
from .query_translator import QueryParameterTranslator
from .search_service import SearchService


class SQLAlchemyDatabaseManager:
    """SQLAlchemy-based database manager with cross-database support"""

    @staticmethod
    def _get_settings():
        """Return the current configuration settings."""
        return ConfigManager.get_instance().get_settings()

    def __init__(
        self,
        database_connect: str,
        template: str = "basic",
        schema_init: bool = True,
        enable_short_term: bool = True,
    ):
        self.database_connect = database_connect
        self.template = template
        self.schema_init = schema_init
        self.enable_short_term = enable_short_term

        # Initialize database auto-creator
        self.auto_creator = DatabaseAutoCreator(schema_init)
        self.enable_auto_creation = bool(self.auto_creator.schema_init)

        # Ensure database exists (create if necessary)
        self.database_connect = self.auto_creator.ensure_database_exists(
            database_connect
        )

        # Parse connection string and create engine
        self.engine = self._create_engine(self.database_connect)
        self.database_type = self.engine.dialect.name

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize search service
        self._search_service = None

        # Initialize query parameter translator for cross-database compatibility
        self.query_translator = QueryParameterTranslator(self.database_type)

        logger.info(f"Initialized SQLAlchemy database manager for {self.database_type}")

        # Scheduler is optional and may not be available in all deployments
        self._scheduler = None

        # Schedule automatic backups if enabled
        settings = self._get_settings()
        if settings.database.backup_enabled:
            try:
                from apscheduler.schedulers.background import BackgroundScheduler

                backup_dir = Path("backups")
                backup_dir.mkdir(parents=True, exist_ok=True)

                def _scheduled_backup():
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    destination = backup_dir / f"backup_{timestamp}.sql"
                    self.backup_database(destination)

                self._scheduler = BackgroundScheduler()
                self._scheduler.add_job(
                    _scheduled_backup,
                    "interval",
                    hours=settings.database.backup_interval_hours,
                )
                self._scheduler.start()
                logger.info(
                    "Scheduled database backups every "
                    f"{settings.database.backup_interval_hours} hours"
                )
            except ModuleNotFoundError:
                logger.warning(
                    "Automatic database backups are disabled because APScheduler "
                    "is not installed. Install it with 'pip install apscheduler' "
                    "or set `database.backup_enabled` to false to disable backups."
                )
            except Exception as e:
                logger.error(
                    "Failed to schedule database backup due to an unexpected "
                    f"error: {e}"
                )

    def _validate_database_dependencies(self, database_connect: str):
        """Validate that required database drivers are installed"""
        if database_connect.startswith("mysql:") or database_connect.startswith(
            "mysql+"
        ):
            # Check for MySQL drivers
            mysql_drivers = []

            if (
                "mysqlconnector" in database_connect
                or "mysql+mysqlconnector" in database_connect
            ):
                if importlib.util.find_spec("mysql.connector") is not None:
                    mysql_drivers.append("mysql-connector-python")

            if "pymysql" in database_connect:
                if importlib.util.find_spec("pymysql") is not None:
                    mysql_drivers.append("PyMySQL")

            # If using generic mysql:// try both drivers
            if database_connect.startswith("mysql://"):
                if importlib.util.find_spec("mysql.connector") is not None:
                    mysql_drivers.append("mysql-connector-python")
                if importlib.util.find_spec("pymysql") is not None:
                    mysql_drivers.append("PyMySQL")

            if not mysql_drivers:
                error_msg = (
                    "❌ No MySQL driver found. Install one of the following:\n\n"
                    "Option 1 (Recommended): pip install mysql-connector-python\n"
                    "Option 2: pip install PyMySQL\n"
                    'Option 3: pip install -e ".[mysql]"  # run inside the Memoria repo\n\n'
                    "Then update your connection string:\n"
                    "- For mysql-connector-python: mysql+mysqlconnector://user:pass@host:port/db\n"
                    "- For PyMySQL: mysql+pymysql://user:pass@host:port/db"
                )
                raise DatabaseError(error_msg)

        elif database_connect.startswith("postgresql:") or database_connect.startswith(
            "postgresql+"
        ):
            # Check for PostgreSQL drivers, supporting psycopg (v3), psycopg2, or asyncpg
            postgres_drivers = ("psycopg", "psycopg2", "asyncpg")
            if not any(importlib.util.find_spec(driver) for driver in postgres_drivers):
                error_msg = (
                    "❌ No PostgreSQL driver found. Install one of the following:\n\n"
                    "Option 1 (Recommended): pip install psycopg[binary] (psycopg v3)\n"
                    "Option 2: pip install psycopg2-binary\n"
                    'Option 3: pip install -e ".[postgres]"  # run inside the Memoria repo\n\n'
                    "Then use connection string: postgresql://user:pass@host:port/db"
                )
                raise DatabaseError(error_msg)

    def _create_engine(self, database_connect: str):
        """Create SQLAlchemy engine with appropriate configuration"""
        try:
            # Validate database driver dependencies first
            self._validate_database_dependencies(database_connect)
            # Parse connection string
            if database_connect.startswith("sqlite:"):
                # Ensure directory exists for SQLite
                if ":///" in database_connect:
                    db_path = database_connect.replace("sqlite:///", "")
                    db_dir = Path(db_path).parent
                    db_dir.mkdir(parents=True, exist_ok=True)

                # SQLite-specific configuration
                engine = create_engine(
                    database_connect,
                    json_serializer=json.dumps,
                    json_deserializer=json.loads,
                    echo=False,
                    # SQLite-specific options
                    connect_args={
                        "check_same_thread": False,  # Allow multiple threads
                    },
                )

            elif database_connect.startswith("mysql:") or database_connect.startswith(
                "mysql+"
            ):
                # MySQL-specific configuration
                connect_args = {"charset": "utf8mb4"}

                # Parse URL for SSL parameters
                parsed = urlparse(database_connect)
                if parsed.query:
                    query_params = parse_qs(parsed.query)

                    # Handle SSL parameters for PyMySQL - enforce secure transport
                    if any(key in query_params for key in ["ssl", "ssl_disabled"]):
                        if query_params.get("ssl", ["false"])[0].lower() == "true":
                            # Enable SSL with secure configuration for required secure transport
                            connect_args["ssl"] = {
                                "ssl_disabled": False,
                                "check_hostname": False,
                                "verify_mode": ssl.CERT_NONE,
                            }
                            # Also add ssl_disabled=False for PyMySQL
                            connect_args["ssl_disabled"] = False
                        elif (
                            query_params.get("ssl_disabled", ["true"])[0].lower()
                            == "false"
                        ):
                            # Enable SSL with secure configuration for required secure transport
                            connect_args["ssl"] = {
                                "ssl_disabled": False,
                                "check_hostname": False,
                                "verify_mode": ssl.CERT_NONE,
                            }
                            # Also add ssl_disabled=False for PyMySQL
                            connect_args["ssl_disabled"] = False

                # Different args for different MySQL drivers
                if "pymysql" in database_connect:
                    # PyMySQL-specific arguments
                    connect_args.update(
                        {
                            "charset": "utf8mb4",
                            "autocommit": False,
                        }
                    )
                elif (
                    "mysqlconnector" in database_connect
                    or "mysql+mysqlconnector" in database_connect
                ):
                    # MySQL Connector/Python-specific arguments
                    connect_args.update(
                        {
                            "charset": "utf8mb4",
                            "use_pure": True,
                        }
                    )

                engine = create_engine(
                    database_connect,
                    json_serializer=json.dumps,
                    json_deserializer=json.loads,
                    echo=False,
                    connect_args=connect_args,
                    pool_pre_ping=True,  # Validate connections
                    pool_recycle=3600,  # Recycle connections every hour
                )

            elif database_connect.startswith(
                "postgresql:"
            ) or database_connect.startswith("postgresql+"):
                # PostgreSQL-specific configuration
                engine = create_engine(
                    database_connect,
                    json_serializer=json.dumps,
                    json_deserializer=json.loads,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                )

            else:
                raise DatabaseError(f"Unsupported database type: {database_connect}")

            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            return engine

        except DatabaseError:
            # Re-raise our custom database errors with helpful messages
            raise
        except ModuleNotFoundError as e:
            if "mysql" in str(e).lower():
                error_msg = (
                    "❌ MySQL driver not found. Install one of the following:\n\n"
                    "Option 1 (Recommended): pip install mysql-connector-python\n"
                    "Option 2: pip install PyMySQL\n"
                    'Option 3: pip install -e ".[mysql]"  # run inside the Memoria repo\n\n'
                    f"Original error: {e}"
                )
                raise DatabaseError(error_msg)
            elif "psycopg" in str(e).lower() or "postgresql" in str(e).lower():
                error_msg = (
                    "❌ PostgreSQL driver not found. Install one of the following:\n\n"
                    "Option 1 (Recommended): pip install psycopg[binary]\n"
                    "Option 2: pip install psycopg2-binary\n"
                    'Option 3: pip install -e ".[postgres]"  # run inside the Memoria repo\n\n'
                    f"Original error: {e}"
                )
                raise DatabaseError(error_msg)
            else:
                raise DatabaseError(f"Missing required dependency: {e}")
        except SQLAlchemyError as e:
            error_msg = f"Database connection failed: {e}\n\nCheck your connection string and ensure the database server is running."
            raise DatabaseError(error_msg)
        except Exception as e:
            raise DatabaseError(f"Failed to create database engine: {e}")

    def initialize_schema(self):
        """Initialize database schema and repair missing columns."""
        try:

            # Create all tables if they don't exist. `checkfirst=True` handles
            # race conditions between multiple workers on startup.
            Base.metadata.create_all(bind=self.engine, checkfirst=True)

            # Setup database-specific features
            self._setup_database_features()

            logger.info(
                f"Database schema initialized successfully for {self.database_type}"
            )

            # Refresh inspector to ensure new tables/columns are visible
            inspector = inspect(self.engine)

            repaired_columns = self._repair_missing_columns(inspector)
            if repaired_columns:
                logger.info(
                    "Added missing columns during schema initialization: "
                    + ", ".join(repaired_columns)
                )
                # Refresh inspector after repairs
                inspector = inspect(self.engine)

            repaired_columns = self._repair_missing_columns(inspector)
            if repaired_columns:
                logger.info(
                    "Added missing columns during schema initialization: "
                    + ", ".join(repaired_columns)
                )
                # Refresh inspector after repairs
                inspector = inspect(self.engine)

            team_changes = migration.ensure_team_support(self.engine)
            if any(team_changes.values()):
                logger.info(
                    "Ensured team schema support: "
                    + ", ".join(
                        f"{category}={len(items)}"
                        for category, items in team_changes.items()
                        if items
                    )
                )
                inspector = inspect(self.engine)

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise DatabaseError(f"Failed to initialize schema: {e}")

    def _repair_missing_columns(self, inspector) -> list[str]:
        """Add any columns missing from existing tables."""

        repaired: list[str] = []
        existing_tables = set(inspector.get_table_names())

        for table_name, table in Base.metadata.tables.items():
            if table_name not in existing_tables:
                continue

            existing_columns = {
                column_info["name"] for column_info in inspector.get_columns(table_name)
            }

            for column in table.columns:
                if column.name in existing_columns:
                    continue

                try:
                    self._add_missing_column(table_name, column)
                    repaired.append(f"{table_name}.{column.name}")
                    existing_columns.add(column.name)
                except Exception as exc:
                    logger.error(
                        f"Failed to add column {column.name} to {table_name}: {exc}"
                    )
                    raise

        return repaired

    def _normalize_column_type_for_dialect(self, column) -> str:
        """Return a dialect-appropriate column type for ALTER TABLE statements."""

        def _resolve_visit_name(sql_type) -> str:
            """Walk type decorators/variants to find the underlying visit name."""

            visited: set[object] = set()

            while sql_type is not None and sql_type not in visited:
                visited.add(sql_type)

                visit_name = getattr(sql_type, "__visit_name__", "")
                if visit_name:
                    return visit_name.upper()

                sql_type = getattr(sql_type, "impl", None) or getattr(
                    sql_type, "adapted", None
                )

            return ""

        visit_name = _resolve_visit_name(column.type)

        if isinstance(column.type, JSON) or visit_name in {"JSON", "JSONB"}:
            if self.database_type == "sqlite":
                return "TEXT"
            if self.database_type == "postgresql":
                return "JSONB"
            return "JSON"

        return column.type.compile(self.engine.dialect)

    def _add_missing_column(self, table_name: str, column):
        """Issue an ALTER TABLE statement to add a missing column."""

        column_type = self._normalize_column_type_for_dialect(column)
        default_clause = self._render_default_clause(column)

        nullable_clause = ""
        if not column.nullable and not column.primary_key:
            if default_clause:
                nullable_clause = " NOT NULL"
            else:
                logger.warning(
                    f"Adding column {table_name}.{column.name} without server default; "
                    "leaving as nullable"
                )

        statement = (
            f"ALTER TABLE {self._format_identifier(table_name)} ADD COLUMN "
            f"{self._format_identifier(column.name)} {column_type}"
        )

        if default_clause:
            statement += f" DEFAULT {default_clause}"

        statement += nullable_clause

        with self.engine.begin() as conn:
            conn.execute(text(statement))

    def _render_default_clause(self, column) -> str | None:
        """Render a DEFAULT clause for a column if possible."""

        if column.server_default is not None:
            default = column.server_default.arg
            if hasattr(default, "text"):
                rendered = default.text
            else:
                rendered = str(default)
            return rendered

        default = getattr(column, "default", None)
        if default is None:
            return None

        default_value = getattr(default, "arg", default)
        if callable(default_value):
            return None

        rendered = self._render_literal(default_value)
        if rendered.upper() == "NULL":
            return None

        return rendered

    def _render_literal(self, value: Any) -> str:
        """Render a literal value appropriate for the current dialect."""

        if isinstance(value, str):
            escaped = value.replace("'", "''")
            return f"'{escaped}'"

        if isinstance(value, bool):
            if self.database_type == "postgresql":
                return "TRUE" if value else "FALSE"
            return "1" if value else "0"

        if value is None:
            return "NULL"

        return str(value)

    def _format_identifier(self, name: str) -> str:
        """Quote an identifier for the current database dialect."""

        preparer = self.engine.dialect.identifier_preparer
        return preparer.quote(name)

    def has_tables(self) -> bool:
        """Check if any tables exist in the database"""
        try:
            inspector = inspect(self.engine)
            return bool(inspector.get_table_names())
        except Exception:  # pragma: no cover - inspection failures
            return False

    def get_db_size(self) -> int:
        """Return size of the database in bytes"""
        backend = self.database_type
        try:
            with self.engine.connect() as conn:
                if backend == "sqlite":
                    database_name = self.engine.url.database
                    if not database_name:
                        logger.warning(
                            "Database size check skipped: SQLite database is in-memory"
                        )
                        return 0

                    db_path = Path(database_name)
                    return db_path.stat().st_size
                if backend == "postgresql":
                    return int(
                        conn.execute(
                            text("SELECT pg_database_size(current_database())")
                        ).scalar()
                    )
                if backend == "mysql":
                    return int(
                        conn.execute(
                            text(
                                "SELECT SUM(data_length + index_length) "
                                "FROM information_schema.tables "
                                "WHERE table_schema = DATABASE()"
                            )
                        ).scalar()
                    )
        except Exception as exc:  # pragma: no cover - depends on backend
            logger.warning(f"Failed to determine database size: {exc}")
        return 0

    def _setup_database_features(self):
        """Setup database-specific features like full-text search"""
        try:
            with self.engine.connect() as conn:
                if self.database_type == "sqlite":
                    self._setup_sqlite_fts(conn)
                elif self.database_type == "mysql":
                    self._setup_mysql_fulltext(conn)
                elif self.database_type == "postgresql":
                    self._setup_postgresql_fts(conn)

                conn.commit()

        except Exception as e:
            logger.warning(f"Failed to setup database-specific features: {e}")

    def _setup_sqlite_fts(self, conn):
        """Setup SQLite FTS5"""
        try:
            # Create FTS5 virtual table
            conn.execute(
                text(
                    """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_fts USING fts5(
                    memory_id,
                    memory_type,
                    namespace,
                    searchable_content,
                    summary,
                    category_primary,
                    content='',
                    contentless_delete=1
                )
            """
                )
            )

            # Create triggers
            if self.enable_short_term:
                conn.execute(
                    text(
                        """
                CREATE TRIGGER IF NOT EXISTS short_term_memory_fts_insert AFTER INSERT ON short_term_memory
                BEGIN
                    INSERT INTO memory_search_fts(memory_id, memory_type, namespace, searchable_content, summary, category_primary)
                    VALUES (NEW.memory_id, 'short_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary);
                END
            """
                    )
                )

            conn.execute(
                text(
                    """
                CREATE TRIGGER IF NOT EXISTS long_term_memory_fts_insert AFTER INSERT ON long_term_memory
                BEGIN
                    INSERT INTO memory_search_fts(memory_id, memory_type, namespace, searchable_content, summary, category_primary)
                    VALUES (NEW.memory_id, 'long_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary);
                END
            """
                )
            )

            logger.info("SQLite FTS5 setup completed")

        except Exception as e:
            logger.warning(f"SQLite FTS5 setup failed: {e}")

    def _setup_mysql_fulltext(self, conn):
        """Setup MySQL FULLTEXT indexes"""
        try:
            # Create FULLTEXT indexes
            if self.enable_short_term:
                conn.execute(
                    text(
                        "ALTER TABLE short_term_memory ADD FULLTEXT INDEX ft_short_term_search (searchable_content, summary)"
                    )
                )
            conn.execute(
                text(
                    "ALTER TABLE long_term_memory ADD FULLTEXT INDEX ft_long_term_search (searchable_content, summary)"
                )
            )

            logger.info("MySQL FULLTEXT indexes setup completed")

        except Exception as e:
            logger.warning(
                f"MySQL FULLTEXT setup failed (indexes may already exist): {e}"
            )

    def _setup_postgresql_fts(self, conn):
        """Setup PostgreSQL full-text search"""
        try:
            # Add tsvector columns
            if self.enable_short_term:
                conn.execute(
                    text(
                        "ALTER TABLE short_term_memory ADD COLUMN IF NOT EXISTS search_vector tsvector"
                    )
                )
            conn.execute(
                text(
                    "ALTER TABLE long_term_memory ADD COLUMN IF NOT EXISTS search_vector tsvector"
                )
            )

            # Create GIN indexes
            if self.enable_short_term:
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_short_term_search_vector ON short_term_memory USING GIN(search_vector)"
                    )
                )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_long_term_search_vector ON long_term_memory USING GIN(search_vector)"
                )
            )

            # Create update functions and triggers
            if self.enable_short_term:
                conn.execute(
                    text(
                        """
                CREATE OR REPLACE FUNCTION update_short_term_search_vector() RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector := to_tsvector('english', COALESCE(NEW.searchable_content, '') || ' ' || COALESCE(NEW.summary, ''));
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql;
            """
                    )
                )

                conn.execute(
                    text(
                        """
                DROP TRIGGER IF EXISTS update_short_term_search_vector_trigger ON short_term_memory;
                CREATE TRIGGER update_short_term_search_vector_trigger
                BEFORE INSERT OR UPDATE ON short_term_memory
                FOR EACH ROW EXECUTE FUNCTION update_short_term_search_vector();
            """
                    )
                )

            logger.info("PostgreSQL FTS setup completed")

        except Exception as e:
            logger.warning(f"PostgreSQL FTS setup failed: {e}")

    def _get_search_service(
        self, *, team_id: str | None = None, workspace_id: str | None = None
    ) -> SearchService:
        """Get search service instance with fresh session"""
        # Always create a new session to avoid stale connections
        session = self.SessionLocal()
        return SearchService(
            session,
            self.database_type,
            vector_search_enabled=vector_search_enabled(),
            team_id=team_id,
            workspace_id=workspace_id,
        )

    def store_chat_history(
        self,
        chat_id: str,
        user_input: str,
        ai_output: str,
        timestamp: datetime,
        session_id: str,
        model: str | None = None,
        namespace: str = "default",
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
        *,
        team_id: str | None = None,
        workspace_id: str | None = None,
        edited_by_model: str | None = None,
    ):
        """Store chat history.

        Args:
            chat_id: Conversation identifier.
            user_input: User message.
            ai_output: AI response text.
            timestamp: Time of conversation.
            session_id: Associated session identifier.
            model: Optional model override. Defaults to the configured
                default agent model.
            namespace: Memory namespace.
            tokens_used: Token count for the interaction.
            metadata: Additional metadata.
        """
        if model is None:
            model = self._get_settings().agents.default_model
        with self.SessionLocal() as session:
            try:
                resolved_editor = edited_by_model or model or "human"

                chat_history = ChatHistory(
                    chat_id=chat_id,
                    user_input=user_input,
                    ai_output=ai_output,
                    timestamp=timestamp,
                    session_id=session_id,
                    model=model,
                    namespace=namespace,
                    team_id=team_id,
                    workspace_id=workspace_id,
                    tokens_used=tokens_used,
                    metadata=metadata or {},
                    last_edited_by_model=resolved_editor,
                )

                session.merge(chat_history)  # Use merge for INSERT OR REPLACE behavior
                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                raise DatabaseError(f"Failed to store chat history: {e}")

    # ------------------------------------------------------------------
    # Agent registry helpers
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        role: str | None = None,
        preferred_model: str | None = None,
        is_agent: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or update an agent registry entry."""

        if not agent_id or not str(agent_id).strip():
            raise DatabaseError("Agent identifier must be provided")

        with self.SessionLocal() as session:
            try:
                instance = session.get(Agent, agent_id)
                if instance is None:
                    instance = Agent(agent_id=str(agent_id).strip(), name=name or str(agent_id))
                if name:
                    instance.name = name
                elif not instance.name:
                    instance.name = str(agent_id)
                instance.role = role
                instance.preferred_model = preferred_model
                instance.is_agent = bool(is_agent)
                instance.metadata = dict(metadata or {})  # type: ignore[attr-defined]
                session.add(instance)
                session.commit()
                session.refresh(instance)
            except SQLAlchemyError as exc:
                session.rollback()
                raise DatabaseError(f"Failed to register agent: {exc}")

        return self._serialise_agent(instance)

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Return a serialised agent record when present."""

        if not agent_id or not str(agent_id).strip():
            return None

        with self.SessionLocal() as session:
            instance = session.get(Agent, str(agent_id).strip())
            return None if instance is None else self._serialise_agent(instance)

    def list_agents(self) -> list[dict[str, Any]]:
        """Return all registered agent profiles."""

        with self.SessionLocal() as session:
            records = session.query(Agent).all()
            return [self._serialise_agent(record) for record in records]

    @staticmethod
    def _serialise_agent(agent: Agent) -> dict[str, Any]:
        """Serialise an :class:`Agent` ORM instance into a dictionary."""

        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "role": agent.role,
            "preferred_model": agent.preferred_model,
            "is_agent": agent.is_agent,
            "metadata": getattr(agent, "metadata", {}) or {},
            "created_at": agent.created_at,
            "updated_at": agent.updated_at,
        }

    def get_chat_history(
        self,
        namespace: str = "default",
        session_id: str | None = None,
        limit: int = 10,
        *,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Get chat history with optional session filtering"""
        with self.SessionLocal() as session:
            try:
                query = session.query(ChatHistory).filter(
                    ChatHistory.namespace == namespace
                )

                if team_id is not None:
                    query = query.filter(ChatHistory.team_id == team_id)

                if workspace_id is not None:
                    query = query.filter(ChatHistory.workspace_id == workspace_id)

                if session_id:
                    query = query.filter(ChatHistory.session_id == session_id)

                results = (
                    query.order_by(ChatHistory.timestamp.desc()).limit(limit).all()
                )

                # Convert to dictionaries
                return [
                    {
                        "chat_id": result.chat_id,
                        "user_input": result.user_input,
                        "ai_output": result.ai_output,
                        "model": result.model,
                        "last_edited_by_model": result.last_edited_by_model,
                        "timestamp": result.timestamp,
                        "session_id": result.session_id,
                        "namespace": result.namespace,
                        "tokens_used": result.tokens_used,
                        "metadata": result.metadata or {},
                    }
                    for result in results
                ]

            except SQLAlchemyError as e:
                raise DatabaseError(f"Failed to get chat history: {e}")

    def record_memory_touches(
        self,
        namespace: str,
        touches: list[dict[str, Any]],
        *,
        access_type: str = "retrieval",
        metadata: dict[str, Any] | None = None,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        """Increment access counters and log touch metadata for memories."""

        if not touches:
            return

        now = datetime.utcnow()
        metadata = metadata or {}

        with self.SessionLocal() as session:
            try:
                for touch in touches:
                    memory_id = touch.get("memory_id")
                    if not memory_id:
                        continue

                    table_hint = touch.get("table")
                    event_source = touch.get("event_source")
                    extra_meta = touch.get("metadata") or {}

                    models: list[type] = []
                    if table_hint == "short_term" and self.enable_short_term:
                        models.append(ShortTermMemory)
                    elif table_hint == "long_term":
                        models.append(LongTermMemory)
                    else:
                        if self.enable_short_term:
                            models.append(ShortTermMemory)
                        models.append(LongTermMemory)

                    updated = False
                    for model in models:
                        query = session.query(model).filter(
                            model.memory_id == memory_id,
                            model.namespace == namespace,
                        )
                        if team_id is not None and hasattr(model, "team_id"):
                            query = query.filter(model.team_id == team_id)
                        if workspace_id is not None and hasattr(model, "workspace_id"):
                            query = query.filter(model.workspace_id == workspace_id)

                        record = query.one_or_none()
                        if record is None:
                            continue
                        record.access_count = (record.access_count or 0) + 1
                        record.last_accessed = now
                        updated = True

                    if updated:
                        session.add(
                            MemoryAccessEvent(
                                memory_id=memory_id,
                                namespace=namespace,
                                team_id=team_id,
                                workspace_id=workspace_id,
                                accessed_at=now,
                                access_type=access_type,
                                source=event_source,
                                metadata_json={**metadata, **extra_meta},
                            )
                        )

                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                logger.error(f"Failed to record memory touches: {exc}")
                raise DatabaseError(f"Failed to record memory touches: {exc}")

    def refresh_memory_last_access(
        self,
        namespace: str,
        memory_ids: Sequence[str],
        *,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        """Update ``last_accessed`` for the provided memories without touching counters."""

        normalized = [
            memory_id for memory_id in dict.fromkeys(memory_ids or []) if memory_id
        ]
        if not normalized:
            return

        now = datetime.utcnow()

        with self.SessionLocal() as session:
            try:
                long_term_query = session.query(LongTermMemory).filter(
                    LongTermMemory.namespace == namespace,
                    LongTermMemory.memory_id.in_(normalized),
                )
                if team_id is not None:
                    long_term_query = long_term_query.filter(
                        LongTermMemory.team_id == team_id
                    )
                if workspace_id is not None:
                    long_term_query = long_term_query.filter(
                        LongTermMemory.workspace_id == workspace_id
                    )

                long_term_query.update(
                    {LongTermMemory.last_accessed: now}, synchronize_session=False
                )

                if self.enable_short_term:
                    short_term_query = session.query(ShortTermMemory).filter(
                        ShortTermMemory.namespace == namespace,
                        ShortTermMemory.memory_id.in_(normalized),
                    )
                    if team_id is not None:
                        short_term_query = short_term_query.filter(
                            ShortTermMemory.team_id == team_id
                        )
                    if workspace_id is not None:
                        short_term_query = short_term_query.filter(
                            ShortTermMemory.workspace_id == workspace_id
                        )

                    short_term_query.update(
                        {ShortTermMemory.last_accessed: now},
                        synchronize_session=False,
                    )

                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                logger.error(f"Failed to refresh memory access timestamps: {exc}")
                raise DatabaseError(
                    f"Failed to refresh memory access timestamps: {exc}"
                )

    def store_long_term_memory_enhanced(
        self,
        memory: ProcessedLongTermMemory,
        chat_id: str,
        namespace: str = "default",
        *,
        team_id: str | None = None,
        workspace_id: str | None = None,
        memory_id: str | None = None,
        edited_by_model: str | None = None,
    ) -> str:
        """Store a ProcessedLongTermMemory with enhanced schema"""
        memory_id = memory_id or str(uuid.uuid4())

        embedding = list(memory.embedding or []) if memory.embedding else None
        if vector_search_enabled() and not embedding:
            summary_text = memory.summary or memory.content
            if summary_text:
                embedding = generate_embedding(summary_text)
                memory.embedding = embedding

        documents_payload: list[dict[str, Any]] | None = None
        if getattr(memory, "documents", None):
            documents_payload = []
            for document in memory.documents or []:
                if document is None:
                    continue
                if isinstance(document, PersonalMemoryDocument):
                    serialised = model_dump(document, mode="python")
                else:
                    try:
                        normalised = model_validate(PersonalMemoryDocument, document)
                    except Exception:
                        if isinstance(document, Mapping):
                            try:
                                normalised = PersonalMemoryDocument(**dict(document))
                            except Exception:
                                continue
                        else:
                            continue
                    serialised = model_dump(normalised, mode="python")
                documents_payload.append(serialised)
            if not documents_payload:
                documents_payload = None

        with self.SessionLocal() as session:
            try:
                long_term_memory = LongTermMemory(
                    memory_id=memory_id,
                    original_chat_id=chat_id,
                    processed_data=model_dump_json_safe(memory),
                    importance_score=memory.importance_score,
                    category_primary=memory.classification.value,
                    retention_type="long_term",
                    namespace=namespace,
                    team_id=team_id,
                    workspace_id=workspace_id,
                    created_at=datetime.now(),
                    searchable_content=memory.content,
                    summary=memory.summary,
                    novelty_score=0.5,
                    relevance_score=0.5,
                    actionability_score=0.5,
                    x_coord=memory.x_coord,
                    y_coord=memory.y_coord,
                    z_coord=memory.z_coord,
                    symbolic_anchors=memory.symbolic_anchors,
                    embedding=embedding,
                    classification=memory.classification.value,
                    memory_importance=memory.importance.value,
                    topic=memory.topic,
                    entities_json=memory.entities,
                    keywords_json=memory.keywords,
                    is_user_context=memory.is_user_context,
                    is_preference=memory.is_preference,
                    is_skill_knowledge=memory.is_skill_knowledge,
                    is_current_project=memory.is_current_project,
                    promotion_eligible=memory.promotion_eligible,
                    duplicate_of=memory.duplicate_of,
                    supersedes_json=memory.supersedes,
                    related_memories_json=memory.related_memories,
                    confidence_score=memory.confidence_score,
                    extraction_timestamp=memory.extraction_timestamp,
                    classification_reason=memory.classification_reason,
                    processed_for_duplicates=False,
                    conscious_processed=False,
                    documents_json=documents_payload,
                    last_edited_by_model=edited_by_model or "human",
                )

                session.add(long_term_memory)
                session.commit()

                logger.debug(f"Stored enhanced long-term memory {memory_id}")
                return memory_id

            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to store enhanced long-term memory: {e}")
                raise DatabaseError(f"Failed to store enhanced long-term memory: {e}")

    def store_direct_long_term_memory(
        self,
        memory: ProcessedLongTermMemory,
        chat_id: str,
        namespace: str = "default",
        *,
        team_id: str | None = None,
        workspace_id: str | None = None,
        documents: (
            PersonalMemoryDocument
            | Mapping[str, Any]
            | Sequence[PersonalMemoryDocument | Mapping[str, Any]]
            | None
        ) = None,
        memory_id: str | None = None,
        edited_by_model: str | None = None,
    ) -> str:
        """Persist long-term memory while allowing direct document payload injection."""

        def _coerce_document(
            payload: PersonalMemoryDocument | Mapping[str, Any],
        ) -> PersonalMemoryDocument:
            if isinstance(payload, PersonalMemoryDocument):
                return payload
            if isinstance(payload, Mapping):
                return model_validate(PersonalMemoryDocument, payload)
            raise TypeError("Unsupported document payload type")

        doc_inputs: list[PersonalMemoryDocument | Mapping[str, Any]] = []

        if getattr(memory, "documents", None):
            doc_inputs.extend(list(memory.documents or []))

        if documents is not None:
            if isinstance(documents, (PersonalMemoryDocument, Mapping)):
                doc_inputs.append(documents)
            else:
                doc_inputs.extend(list(documents))

        normalized_documents: list[PersonalMemoryDocument] | None = None
        if doc_inputs:
            normalized_documents = []
            for item in doc_inputs:
                normalized_documents.append(_coerce_document(item))

        payload = ProcessedLongTermMemory(**model_dump(memory, mode="python"))
        if normalized_documents is not None:
            payload.documents = normalized_documents

        return self.store_long_term_memory_enhanced(
            payload,
            chat_id,
            namespace,
            team_id=team_id,
            workspace_id=workspace_id,
            memory_id=memory_id,
            edited_by_model=edited_by_model,
        )

    def store_memory_links(
        self,
        source_memory_id: str,
        related_ids: Sequence[str],
        *,
        relation: str = "related",
    ) -> None:
        """Persist bidirectional relationship edges between memories."""

        if not source_memory_id or not related_ids:
            return

        targets = [
            value for value in related_ids if value and value != source_memory_id
        ]
        if not targets:
            return

        with self.SessionLocal() as session:
            try:
                filters = []
                for target_id in targets:
                    filters.append(
                        and_(
                            LinkMemoryThread.source_memory_id == source_memory_id,
                            LinkMemoryThread.target_memory_id == target_id,
                        )
                    )
                    filters.append(
                        and_(
                            LinkMemoryThread.source_memory_id == target_id,
                            LinkMemoryThread.target_memory_id == source_memory_id,
                        )
                    )

                if filters:
                    session.query(LinkMemoryThread).filter(or_(*filters)).delete(
                        synchronize_session=False
                    )

                for target_id in targets:
                    session.add(
                        LinkMemoryThread(
                            source_memory_id=source_memory_id,
                            target_memory_id=target_id,
                            relation=relation,
                        )
                    )
                    session.add(
                        LinkMemoryThread(
                            source_memory_id=target_id,
                            target_memory_id=source_memory_id,
                            relation=relation,
                        )
                    )

                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                logger.debug(f"Failed to persist memory links: {exc}")

    def store_short_term_memory(
        self,
        memory: ProcessedLongTermMemory,
        chat_id: str,
        namespace: str = "default",
        *,
        team_id: str | None = None,
        workspace_id: str | None = None,
        edited_by_model: str | None = None,
    ) -> str:
        """Store a short-term memory row based on processed long-term memory data."""

        memory_id = str(uuid.uuid4())
        created_at = datetime.now()
        expires_at = created_at + timedelta(days=7)
        processed_payload = model_dump_json_safe(memory)

        embedding = list(memory.embedding or []) if memory.embedding else None
        if vector_search_enabled() and not embedding:
            summary_text = memory.summary or memory.content
            if summary_text:
                embedding = generate_embedding(summary_text)
                memory.embedding = embedding

        classification_value = getattr(
            memory.classification, "value", str(memory.classification)
        )

        if memory.is_user_context:
            category = "essential_user_profile"
        elif memory.is_preference:
            category = "essential_preference"
        elif memory.is_skill_knowledge:
            category = "essential_skill"
        elif memory.is_current_project:
            category = "essential_project"
        elif isinstance(classification_value, str) and classification_value.startswith(
            "essential_"
        ):
            category = classification_value
        elif classification_value == "essential":
            category = "essential_conversation"
        elif classification_value == "conscious-info":
            category = "essential_conscious"
        else:
            category = "essential_context"

        with self.SessionLocal() as session:
            try:
                short_term_memory = ShortTermMemory(
                    memory_id=memory_id,
                    chat_id=chat_id,
                    processed_data=processed_payload,
                    importance_score=memory.importance_score,
                    category_primary=category,
                    retention_type="short_term",
                    namespace=namespace,
                    team_id=team_id,
                    workspace_id=workspace_id,
                    created_at=created_at,
                    expires_at=expires_at,
                    access_count=0,
                    searchable_content=memory.content,
                    summary=memory.summary,
                    x_coord=memory.x_coord,
                    y_coord=memory.y_coord,
                    z_coord=memory.z_coord,
                    symbolic_anchors=memory.symbolic_anchors,
                    embedding=embedding,
                    last_edited_by_model=edited_by_model or "human",
                )

                session.add(short_term_memory)
                session.commit()
                logger.debug(f"Stored short-term memory {memory_id}")
                return memory_id

            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to store short-term memory: {e}")
                raise DatabaseError(f"Failed to store short-term memory: {e}")

    def store_manual_short_term_memory(
        self,
        *,
        chat_id: str,
        namespace: str,
        processed_payload: dict[str, Any] | None = None,
        summary: str,
        importance_score: float = 0.5,
        retention_type: str = "short_term",
        x_coord: float | None = None,
        y_coord: float | None = None,
        z_coord: float | None = None,
        symbolic_anchors: list[str] | None = None,
        expires_in_days: int = 7,
        team_id: str | None = None,
        workspace_id: str | None = None,
        edited_by_model: str | None = None,
    ) -> str:
        """Store a manual short-term memory without requiring processing metadata."""

        memory_id = str(uuid.uuid4())
        created_at = datetime.now()
        expires_at = created_at + timedelta(days=expires_in_days)

        payload = processed_payload or {}
        anchors = symbolic_anchors or []
        embedding = None
        if vector_search_enabled() and summary:
            embedding = generate_embedding(summary)

        with self.SessionLocal() as session:
            try:
                short_term_memory = ShortTermMemory(
                    memory_id=memory_id,
                    chat_id=chat_id,
                    processed_data=payload,
                    importance_score=importance_score,
                    category_primary="manual_staged",
                    retention_type=retention_type,
                    namespace=namespace,
                    team_id=team_id,
                    workspace_id=workspace_id,
                    created_at=created_at,
                    expires_at=expires_at,
                    access_count=0,
                    searchable_content=summary,
                    summary=summary,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    z_coord=z_coord,
                    symbolic_anchors=anchors,
                    embedding=embedding,
                    last_edited_by_model=edited_by_model or "human",
                )

                session.add(short_term_memory)
                session.commit()
                logger.debug(f"Stored manual short-term memory {memory_id}")
                return memory_id
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to store manual short-term memory: {e}")
                raise DatabaseError(f"Failed to store manual short-term memory: {e}")

    def search_memories(
        self,
        query: str,
        namespace: str = "default",
        category_filter: list[str] | None = None,
        limit: int = 10,
        memory_types: list[str] | None = None,
        use_anchor: bool = True,
        use_fuzzy: bool = False,
        fuzzy_min_similarity: int = 50,
        fuzzy_max_results: int = 5,
        keywords: list[str] | None = None,
        adaptive_min_similarity: bool = True,
        rank_weights: dict[str, float] | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        max_distance: float | None = None,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        anchors: list[str] | None = None,
        query_embedding: Sequence[float] | None = None,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Search memories using the cross-database search service.

        If the initial search fails or returns no results, a second attempt is made
        using fuzzy search with a lowered similarity threshold. When both attempts
        fail, a structured response with a hint is returned.
        """

        search_service = self._get_search_service(
            team_id=team_id, workspace_id=workspace_id
        )

        try:

            def _normalize_response_structure(
                response: list[dict[str, Any]] | dict[str, Any],
            ) -> dict[str, Any]:
                if isinstance(response, dict):
                    raw_items = response.get("results") or []
                    if isinstance(raw_items, Sequence) and not isinstance(
                        raw_items, (str, bytes)
                    ):
                        items = list(raw_items)
                    elif raw_items:
                        items = [raw_items]
                    else:
                        items = []
                    response["results"] = items
                    response.setdefault("attempted", [])
                    return response

                raw_items = response or []
                if isinstance(raw_items, Sequence) and not isinstance(
                    raw_items, (str, bytes)
                ):
                    items = list(raw_items)
                elif raw_items:
                    items = [raw_items]
                else:
                    items = []
                return {"results": items, "attempted": []}

            try:
                results = search_service.search_memories(
                    query,
                    namespace,
                    category_filter,
                    limit,
                    memory_types,
                    use_anchor=use_anchor,
                    use_fuzzy=use_fuzzy,
                    fuzzy_min_similarity=fuzzy_min_similarity,
                    fuzzy_max_results=fuzzy_max_results,
                    keywords=keywords,
                    adaptive_min_similarity=adaptive_min_similarity,
                    rank_weights=rank_weights,
                    x=x,
                    y=y,
                    z=z,
                    max_distance=max_distance,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    min_importance=min_importance,
                    max_importance=max_importance,
                    anchors=anchors,
                    query_embedding=query_embedding,
                )

                results = _normalize_response_structure(results)
                result_items = (
                    results["results"] if isinstance(results, dict) else results
                )

            except Exception as e:
                logger.warning(
                    f"Primary search failed for query '{query}': {e}. Retrying with fuzzy search"
                )
                results = {"results": [], "attempted": []}
                result_items = []

            if not result_items:
                try:
                    results = search_service.search_memories(
                        query,
                        namespace,
                        category_filter,
                        limit,
                        memory_types,
                        use_anchor=use_anchor,
                        use_fuzzy=True,
                        fuzzy_min_similarity=min(fuzzy_min_similarity, 40),
                        fuzzy_max_results=fuzzy_max_results,
                        keywords=keywords,
                        adaptive_min_similarity=adaptive_min_similarity,
                        rank_weights=rank_weights,
                        x=x,
                        y=y,
                        z=z,
                        max_distance=max_distance,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        min_importance=min_importance,
                        max_importance=max_importance,
                        anchors=anchors,
                        query_embedding=query_embedding,
                    )

                    results = _normalize_response_structure(results)
                    result_items = (
                        results["results"] if isinstance(results, dict) else results
                    )
                except Exception as e:
                    logger.error(f"Fuzzy search failed for query '{query}': {e}")
                    return {
                        "results": [],
                        "attempted": (
                            results.get("attempted", [])
                            if isinstance(results, dict)
                            else []
                        ),
                        "hint": "Expected query format: keywords=['foo']",
                        "error": str(e),
                    }

                if not result_items:
                    return {
                        "results": [],
                        "attempted": (
                            results.get("attempted", [])
                            if isinstance(results, dict)
                            else []
                        ),
                        "hint": "Expected query format: keywords=['foo']",
                        "error": "no_matches",
                    }

            if isinstance(result_items, Sequence) and not isinstance(
                result_items, (str, bytes)
            ):
                result_count = len(result_items)
            else:
                result_count = 0

            logger.debug(f"Search for '{query}' returned {result_count} results")
            return results

        finally:
            # Ensure session is properly closed
            search_service.session.close()

    def find_related_memory_candidates(
        self,
        *,
        namespace: str,
        symbolic_anchors: Sequence[str] | None = None,
        keywords: Sequence[str] | None = None,
        topic: str | None = None,
        exclude_ids: Sequence[str] | None = None,
        limit: int = 5,
        privacy_floor: float | None = -10.0,
        team_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidate memories likely related via shared metadata."""

        search_service = self._get_search_service(team_id=team_id)
        try:
            return search_service.find_related_memory_candidates(
                namespace=namespace,
                symbolic_anchors=symbolic_anchors,
                keywords=keywords,
                topic=topic,
                exclude_ids=exclude_ids,
                limit=limit,
                min_privacy=privacy_floor,
            )
        finally:
            search_service.session.close()

    def get_related_memories(
        self,
        *,
        memory_id: str,
        namespace: str,
        limit: int = 10,
        relation_types: Sequence[str] | None = None,
        privacy_floor: float | None = -10.0,
        team_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return persisted related memories for ``memory_id``."""

        search_service = self._get_search_service(team_id=team_id)
        try:
            return search_service.get_related_memories(
                memory_id=memory_id,
                namespace=namespace,
                limit=limit,
                relation_types=relation_types,
                min_privacy=privacy_floor,
            )
        finally:
            search_service.session.close()

    def get_memory_stats(
        self,
        namespace: str = "default",
        *,
        team_id: str | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive memory statistics"""
        with self.SessionLocal() as session:
            try:
                stats = {}

                # Basic counts
                chat_query = session.query(ChatHistory).filter(
                    ChatHistory.namespace == namespace
                )
                if team_id is not None:
                    chat_query = chat_query.filter(ChatHistory.team_id == team_id)

                stats["chat_history_count"] = chat_query.count()

                if self.enable_short_term:
                    short_query = session.query(ShortTermMemory).filter(
                        ShortTermMemory.namespace == namespace
                    )
                    if team_id is not None:
                        short_query = short_query.filter(
                            ShortTermMemory.team_id == team_id
                        )

                    stats["short_term_count"] = short_query.count()
                else:
                    stats["short_term_count"] = 0

                long_query = session.query(LongTermMemory).filter(
                    LongTermMemory.namespace == namespace
                )
                if team_id is not None:
                    long_query = long_query.filter(LongTermMemory.team_id == team_id)

                stats["long_term_count"] = long_query.count()

                # Category breakdown
                categories = {}

                # Short-term categories
                if self.enable_short_term:
                    short_query = session.query(
                        ShortTermMemory.category_primary,
                        func.count(ShortTermMemory.memory_id).label("count"),
                    ).filter(ShortTermMemory.namespace == namespace)

                    if team_id is not None:
                        short_query = short_query.filter(
                            ShortTermMemory.team_id == team_id
                        )

                    short_categories = short_query.group_by(
                        ShortTermMemory.category_primary
                    ).all()

                    for cat, count in short_categories:
                        categories[cat] = categories.get(cat, 0) + count

                # Long-term categories
                long_query = session.query(
                    LongTermMemory.category_primary,
                    func.count(LongTermMemory.memory_id).label("count"),
                ).filter(LongTermMemory.namespace == namespace)

                if team_id is not None:
                    long_query = long_query.filter(LongTermMemory.team_id == team_id)

                long_categories = long_query.group_by(
                    LongTermMemory.category_primary
                ).all()

                for cat, count in long_categories:
                    categories[cat] = categories.get(cat, 0) + count

                stats["memories_by_category"] = categories

                # Average importance
                if self.enable_short_term:
                    short_avg_query = session.query(
                        func.avg(ShortTermMemory.importance_score)
                    ).filter(ShortTermMemory.namespace == namespace)
                    if team_id is not None:
                        short_avg_query = short_avg_query.filter(
                            ShortTermMemory.team_id == team_id
                        )

                    short_avg = short_avg_query.scalar() or 0
                else:
                    short_avg = 0

                long_avg_query = session.query(
                    func.avg(LongTermMemory.importance_score)
                ).filter(LongTermMemory.namespace == namespace)
                if team_id is not None:
                    long_avg_query = long_avg_query.filter(
                        LongTermMemory.team_id == team_id
                    )

                long_avg = long_avg_query.scalar() or 0

                total_memories = stats["short_term_count"] + stats["long_term_count"]
                if total_memories > 0:
                    # Weight averages by count
                    total_avg = (
                        (short_avg * stats["short_term_count"])
                        + (long_avg * stats["long_term_count"])
                    ) / total_memories
                    stats["average_importance"] = float(total_avg) if total_avg else 0.0
                else:
                    stats["average_importance"] = 0.0

                # Database info
                stats["database_type"] = self.database_type
                stats["database_url"] = (
                    self.database_connect.split("@")[-1]
                    if "@" in self.database_connect
                    else self.database_connect
                )

                return stats

            except SQLAlchemyError as e:
                raise DatabaseError(f"Failed to get memory stats: {e}")

    def clear_memory(
        self,
        namespace: str = "default",
        memory_type: str | None = None,
        *,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ):
        """Clear memory data"""
        with self.SessionLocal() as session:
            try:
                if memory_type == "short_term":
                    if not self.enable_short_term:
                        return
                    short_query = session.query(ShortTermMemory).filter(
                        ShortTermMemory.namespace == namespace
                    )
                    if team_id is not None:
                        short_query = short_query.filter(
                            ShortTermMemory.team_id == team_id
                        )
                    if workspace_id is not None:
                        short_query = short_query.filter(
                            ShortTermMemory.workspace_id == workspace_id
                        )
                    short_query.delete()
                elif memory_type == "long_term":
                    long_query = session.query(LongTermMemory).filter(
                        LongTermMemory.namespace == namespace
                    )
                    if team_id is not None:
                        long_query = long_query.filter(
                            LongTermMemory.team_id == team_id
                        )
                    if workspace_id is not None:
                        long_query = long_query.filter(
                            LongTermMemory.workspace_id == workspace_id
                        )
                    long_query.delete()
                elif memory_type == "chat_history":
                    chat_query = session.query(ChatHistory).filter(
                        ChatHistory.namespace == namespace
                    )
                    if team_id is not None:
                        chat_query = chat_query.filter(ChatHistory.team_id == team_id)
                    if workspace_id is not None:
                        chat_query = chat_query.filter(
                            ChatHistory.workspace_id == workspace_id
                        )
                    chat_query.delete()
                else:  # Clear all
                    if self.enable_short_term:
                        short_query = session.query(ShortTermMemory).filter(
                            ShortTermMemory.namespace == namespace
                        )
                        if team_id is not None:
                            short_query = short_query.filter(
                                ShortTermMemory.team_id == team_id
                            )
                        if workspace_id is not None:
                            short_query = short_query.filter(
                                ShortTermMemory.workspace_id == workspace_id
                            )
                        short_query.delete()

                    long_query = session.query(LongTermMemory).filter(
                        LongTermMemory.namespace == namespace
                    )
                    if team_id is not None:
                        long_query = long_query.filter(
                            LongTermMemory.team_id == team_id
                        )
                    if workspace_id is not None:
                        long_query = long_query.filter(
                            LongTermMemory.workspace_id == workspace_id
                        )
                    long_query.delete()

                    chat_query = session.query(ChatHistory).filter(
                        ChatHistory.namespace == namespace
                    )
                    if team_id is not None:
                        chat_query = chat_query.filter(ChatHistory.team_id == team_id)
                    if workspace_id is not None:
                        chat_query = chat_query.filter(
                            ChatHistory.workspace_id == workspace_id
                        )
                    chat_query.delete()

                session.commit()

            except SQLAlchemyError as e:
                session.rollback()
                raise DatabaseError(f"Failed to clear memory: {e}")

    def execute_with_translation(self, query: str, parameters: dict[str, Any] = None):
        """
        Execute a query with automatic parameter translation for cross-database compatibility.

        Args:
            query: SQL query string
            parameters: Query parameters

        Returns:
            Query result
        """
        if parameters:
            translated_params = self.query_translator.translate_parameters(parameters)
        else:
            translated_params = {}

        with self.engine.connect() as conn:
            result = conn.execute(text(query), translated_params)
            conn.commit()
            return result

    def get_connection(self):
        """Retained for backward compatibility with components like ConsciousAgent and StorageService."""
        return self._get_connection()

    def _get_connection(self):
        """
        Compatibility method for legacy code that expects raw database connections.

        Returns a context manager that provides a SQLAlchemy connection with
        automatic parameter translation support.

        This is used by memory.py for direct SQL queries.
        """
        from contextlib import contextmanager

        @contextmanager
        def connection_context():
            # "TranslatingConnection" maintains compatibility with older modules
            # that still expect a raw DB-API connection. Legacy code that issues
            # ad-hoc SQL should use this wrapper so parameters are translated to
            # the active database dialect. New code should prefer SQLAlchemy's
            # higher-level abstractions instead of this shim.
            class TranslatingConnection:
                """Wrapper that adds parameter translation to SQLAlchemy connections"""

                def __init__(self, conn, translator):
                    self._conn = conn
                    self._translator = translator

                def execute(self, query, parameters=None):
                    """Execute query with automatic parameter translation"""
                    if parameters:
                        # Translate any DB-API style placeholders before delegation
                        if hasattr(query, "text"):
                            # SQLAlchemy text() object
                            translated_params = self._translator.translate_parameters(
                                parameters
                            )
                            return self._conn.execute(query, translated_params)
                        else:
                            # Raw string query
                            translated_params = self._translator.translate_parameters(
                                parameters
                            )
                            return self._conn.execute(
                                text(str(query)), translated_params
                            )
                    else:
                        return self._conn.execute(query)

                def commit(self):
                    """Commit transaction"""
                    return self._conn.commit()

                def rollback(self):
                    """Rollback transaction"""
                    return self._conn.rollback()

                def close(self):
                    """Close connection"""
                    return self._conn.close()

                def fetchall(self):
                    """Compatibility method for cursor-like usage"""
                    # This is for backwards compatibility with code that expects cursor.fetchall()
                    return []

                def scalar(self):
                    """Compatibility method for cursor-like usage"""
                    return None

                def __getattr__(self, name):
                    """Delegate unknown attributes to the underlying connection"""
                    # Pass through any attributes we don't explicitly override.
                    return getattr(self._conn, name)

            conn = self.engine.connect()
            try:
                yield TranslatingConnection(conn, self.query_translator)
            finally:
                conn.close()

        return connection_context()

    def backup_database(self, destination: Path) -> None:
        """Create a database backup at the given destination path"""
        try:
            destination = Path(destination)
            backend = self.database_type

            if backend == "sqlite":
                database_name = self.engine.url.database
                if not database_name:
                    logger.warning(
                        "Skipping SQLite backup: database is using in-memory storage"
                    )
                    return

                db_path = Path(database_name)
                shutil.copy2(db_path, destination)
            elif backend == "postgresql":
                cmd = ["pg_dump", self.database_connect]
                with destination.open("wb") as f:
                    result = subprocess.run(cmd, stdout=f, capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode())
            elif backend == "mysql":
                mysql_url = make_url(self.database_connect)
                cmd = ["mysqldump"]

                if mysql_url.host:
                    cmd.append(f"--host={mysql_url.host}")
                if mysql_url.port:
                    cmd.append(f"--port={mysql_url.port}")
                if mysql_url.username:
                    cmd.append(f"--user={mysql_url.username}")
                if mysql_url.password is not None:
                    cmd.append(f"--password={mysql_url.password}")

                if not mysql_url.database:
                    raise DatabaseError(
                        "MySQL connection string must include a database name"
                    )

                cmd.append(mysql_url.database)
                with destination.open("wb") as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode())
            else:
                raise DatabaseError(f"Unsupported database type: {backend}")

            logger.info(f"Database backup created at {destination}")
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise

    def restore_database(self, backup_path: Path) -> None:
        """Restore the database from a backup file"""
        try:
            backup_path = Path(backup_path)
            backend = self.database_type

            if backend == "sqlite":
                database_name = self.engine.url.database
                if not database_name:
                    logger.warning(
                        "Skipping SQLite restore: database is using in-memory storage"
                    )
                    return

                db_path = Path(database_name)
                shutil.copy2(backup_path, db_path)
            elif backend == "postgresql":
                cmd = ["psql", self.database_connect]
                with backup_path.open("rb") as f:

                    result = subprocess.run(cmd, stdin=f, capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode())
            elif backend == "mysql":
                mysql_url = make_url(self.database_connect)
                cmd = ["mysql"]

                if mysql_url.host:
                    cmd.append(f"--host={mysql_url.host}")
                if mysql_url.port:
                    cmd.append(f"--port={mysql_url.port}")
                if mysql_url.username:
                    cmd.append(f"--user={mysql_url.username}")
                if mysql_url.password is not None:
                    cmd.append(f"--password={mysql_url.password}")

                if not mysql_url.database:
                    raise DatabaseError(
                        "MySQL connection string must include a database name"
                    )

                cmd.append(mysql_url.database)

                with backup_path.open("rb") as f:

                    result = subprocess.run(
                        cmd, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode())
            else:
                raise DatabaseError(f"Unsupported database type: {backend}")

            logger.info(f"Database restored from {backup_path}")

        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            raise

    def export_dataset(
        self,
        destination: str | Path | None = None,
        *,
        format: str = "json",
        tables: Sequence[str] | None = None,
    ):
        """Export database tables to JSON or NDJSON."""

        result = migration.export_database(
            self.SessionLocal,
            format=format,
            destination=destination,
            tables=tables,
        )
        exported_rows = sum(t["row_count"] for t in result.metadata.get("tables", []))
        target = destination or "memory"
        logger.info(
            "Exported %s rows from %s tables to %s",
            exported_rows,
            len(result.metadata.get("tables", [])),
            target,
        )
        return result

    def import_dataset(
        self,
        source: str | Path | dict[str, Any],
        *,
        format: str | None = None,
        tables: Sequence[str] | None = None,
        truncate: bool = True,
    ) -> dict[str, Any]:
        """Import database tables from a JSON or NDJSON payload."""

        metadata = migration.import_database(
            self.SessionLocal,
            source,
            format=format,
            tables=tables,
            truncate=truncate,
        )
        logger.info(
            "Imported dataset for %s tables (truncate=%s)",
            (
                len(metadata.get("tables", []))
                if isinstance(metadata, dict)
                else "unknown"
            ),
            bool(truncate),
        )
        return metadata

    @contextmanager
    def backup_guard(self, backup_path: Path):
        """Context manager that creates a backup and restores on error"""
        backup_path = Path(backup_path)
        self.backup_database(backup_path)
        try:
            yield
        except Exception:
            self.restore_database(backup_path)
            raise
        finally:
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except Exception:
                    pass

    def close(self):
        """Close database connections"""
        if self._search_service and hasattr(self._search_service, "session"):
            self._search_service.session.close()

        scheduler = getattr(self, "_scheduler", None)
        if scheduler:
            try:
                scheduler.shutdown(wait=False)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to shut down backup scheduler cleanly: %s", exc)
            finally:
                self._scheduler = None

        if hasattr(self, "engine"):
            self.engine.dispose()

    def get_database_info(self) -> dict[str, Any]:
        """Get database information and capabilities"""
        base_info = {
            "database_type": self.database_type,
            "database_url": (
                self.database_connect.split("@")[-1]
                if "@" in self.database_connect
                else self.database_connect
            ),
            "driver": self.engine.dialect.driver,
            "server_version": getattr(self.engine.dialect, "server_version_info", None),
            "supports_fulltext": True,  # Assume true for SQLAlchemy managed connections
            "auto_creation_enabled": getattr(
                self,
                "enable_auto_creation",
                getattr(self.auto_creator, "schema_init", False),
            ),
        }

        # Add auto-creation specific information
        if hasattr(self, "auto_creator"):
            creation_info = self.auto_creator.get_database_info(self.database_connect)
            base_info.update(creation_info)

        return base_info
