"""
Transaction management utilities for Memoria
Provides robust transaction handling with proper error recovery
"""

import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger

from .exceptions import DatabaseError, ValidationError


class ExecutionResult(list):
    """Container for query execution results with rowcount metadata."""

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        rowcount: int | None = None,
        rowcount_source: str = "unknown",
    ) -> None:
        super().__init__(rows or [])
        self.rowcount = rowcount
        self.rowcount_source = rowcount_source


class TransactionState(str, Enum):
    """Transaction states"""

    PENDING = "pending"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class IsolationLevel(str, Enum):
    """Database isolation levels"""

    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class TransactionOperation:
    """Represents a single database operation within a transaction"""

    query: str
    params: list[Any] | None
    operation_type: str  # 'select', 'insert', 'update', 'delete'
    table: str | None = None
    expected_rows: int | None = None  # For validation
    rollback_query: str | None = None  # Compensation query if needed


@dataclass
class TransactionResult:
    """Result of a transaction execution"""

    success: bool
    state: TransactionState
    operations_completed: int
    total_operations: int
    error_message: str | None = None
    execution_time: float | None = None
    rollback_performed: bool = False


class TransactionManager:
    """Robust transaction manager with error recovery"""

    def __init__(self, connector, max_retries: int = 3, retry_delay: float = 0.1):
        self.connector = connector
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.current_transaction = None

    @contextmanager
    def transaction(
        self,
        isolation_level: IsolationLevel | None = None,
        timeout: float | None = 30.0,
        readonly: bool = False,
    ):
        """Context manager for database transactions with proper error handling"""

        transaction_id = f"txn_{int(time.time()*1000)}"
        start_time = time.time()

        conn = None
        connection_acquired = False

        try:
            with ExitStack() as stack:
                # Get connection and start transaction
                raw_connection = self.connector.get_connection()
                conn = self._enter_connection_context(raw_connection, stack)
                if conn is None:
                    raise DatabaseError("Failed to acquire database connection.")

                connection_acquired = True

                # Set isolation level if specified
                if isolation_level:
                    self._set_isolation_level(conn, isolation_level)

                # Set readonly mode if specified
                if readonly:
                    self._set_readonly(conn, True)

                # Begin transaction
                self._begin_transaction(conn)

                logger.debug(f"Started transaction {transaction_id}")

                # Store transaction context
                self.current_transaction = {
                    "id": transaction_id,
                    "connection": conn,
                    "start_time": start_time,
                    "operations": [],
                    "state": TransactionState.ACTIVE,
                }

                try:
                    yield TransactionContext(self, conn, transaction_id)

                    # Check timeout
                    if timeout and (time.time() - start_time) > timeout:
                        raise DatabaseError(
                            f"Transaction {transaction_id} timed out after {timeout}s"
                        )

                    # Commit transaction
                    if conn is not None:
                        conn.commit()
                        self.current_transaction["state"] = TransactionState.COMMITTED
                        logger.debug(f"Committed transaction {transaction_id}")

                except Exception as e:
                    # Rollback on any error
                    if conn is not None:
                        try:
                            conn.rollback()
                            self.current_transaction["state"] = (
                                TransactionState.ROLLED_BACK
                            )
                            logger.warning(
                                f"Rolled back transaction {transaction_id}: {e}"
                            )
                        except Exception as rollback_error:
                            self.current_transaction["state"] = TransactionState.FAILED
                            logger.error(
                                f"Failed to rollback transaction {transaction_id}: {rollback_error}"
                            )

                    raise e

        except Exception as e:
            logger.error(f"Transaction {transaction_id} failed: {e}")
            if not connection_acquired:
                raise
            raise DatabaseError(f"Transaction failed: {e}") from e

        finally:
            # Cleanup
            if self.current_transaction:
                execution_time = time.time() - start_time
                logger.debug(
                    f"Transaction {transaction_id} completed in {execution_time:.3f}s"
                )
                self.current_transaction = None

    def execute_atomic_operations(
        self,
        operations: list[TransactionOperation],
        isolation_level: IsolationLevel | None = None,
    ) -> TransactionResult:
        """Execute multiple operations atomically with validation"""

        start_time = time.time()
        completed_ops = 0

        try:
            with self.transaction(isolation_level=isolation_level) as tx:
                for i, operation in enumerate(operations):
                    try:
                        # Validate operation parameters
                        self._validate_operation(operation)

                        # Execute operation
                        result = tx.execute(operation.query, operation.params)

                        # Validate result if expected rows specified
                        if operation.expected_rows is not None:
                            rowcount = self._resolve_rowcount(result, operation, i)
                            if rowcount is None:
                                continue

                            if rowcount != operation.expected_rows:
                                raise DatabaseError(
                                    f"Operation {i} affected {rowcount} rows, expected {operation.expected_rows}"
                                )

                        completed_ops += 1

                    except Exception as e:
                        logger.error(f"Operation {i} failed: {e}")
                        raise DatabaseError(
                            f"Operation {i} ({operation.operation_type}) failed: {e}"
                        )

                return TransactionResult(
                    success=True,
                    state=TransactionState.COMMITTED,
                    operations_completed=completed_ops,
                    total_operations=len(operations),
                    execution_time=time.time() - start_time,
                )

        except Exception as e:
            return TransactionResult(
                success=False,
                state=TransactionState.ROLLED_BACK,
                operations_completed=completed_ops,
                total_operations=len(operations),
                error_message=str(e),
                execution_time=time.time() - start_time,
                rollback_performed=True,
            )

    def execute_with_retry(
        self,
        operation: Callable[[], Any],
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ) -> Any:
        """Execute operation with automatic retry on transient failures"""

        retries = max_retries or self.max_retries
        delay = retry_delay or self.retry_delay
        last_error = None

        for attempt in range(retries + 1):
            try:
                return operation()
            except Exception as e:
                last_error = e

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.debug(f"Non-retryable error: {e}")
                    break

                if attempt < retries:
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{retries + 1}), retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Operation failed after {retries + 1} attempts: {e}")

        raise DatabaseError(
            f"Operation failed after {retries + 1} attempts: {last_error}"
        )

    def _validate_operation(self, operation: TransactionOperation):
        """Validate transaction operation parameters"""
        if not operation.query or not operation.query.strip():
            raise ValidationError("Query cannot be empty")

        if operation.params is not None and not isinstance(operation.params, list):
            raise ValidationError("Parameters must be a list or None")

        # Basic SQL injection detection
        query_lower = operation.query.lower().strip()
        dangerous_patterns = [
            ";--",
            "; --",
            "/*",
            "*/",
            "xp_",
            "sp_execute",
            "union select",
            "drop table",
            "truncate table",
        ]

        for pattern in dangerous_patterns:
            if pattern in query_lower:
                raise ValidationError(
                    f"Potentially dangerous SQL pattern detected: {pattern}"
                )

    def _resolve_rowcount(
        self,
        result: Any,
        operation: TransactionOperation,
        index: int,
    ) -> int | None:
        """Determine the affected row count for an operation result."""

        rowcount = getattr(result, "rowcount", None)
        rowcount_source = getattr(result, "rowcount_source", "unknown")

        if isinstance(rowcount, int) and rowcount >= 0:
            if rowcount_source != "cursor":
                logger.warning(
                    "Backend did not directly provide affected rows for operation %s (%s on %s); using inferred count %s.",
                    index,
                    operation.operation_type,
                    operation.table or "<unknown table>",
                    rowcount,
                )
            return rowcount

        fallback_rowcount: int | None = None

        if isinstance(result, list) and result:
            first_entry = result[0]
            if isinstance(first_entry, dict) and "affected_rows" in first_entry:
                candidate = first_entry.get("affected_rows")
                if isinstance(candidate, int) and candidate >= 0:
                    fallback_rowcount = candidate

        if (
            fallback_rowcount is None
            and getattr(result, "rowcount_source", None) != "payload"
            and hasattr(result, "__len__")
        ):
            try:
                fallback_rowcount = len(result)
            except TypeError:
                fallback_rowcount = None

        if fallback_rowcount is not None:
            logger.warning(
                "Backend could not report affected rows for operation %s (%s on %s); falling back to inferred count %s.",
                index,
                operation.operation_type,
                operation.table or "<unknown table>",
                fallback_rowcount,
            )
            return fallback_rowcount

        logger.warning(
            "Unable to determine affected rows for operation %s (%s on %s) using result type %s.",
            index,
            operation.operation_type,
            operation.table or "<unknown table>",
            type(result).__name__,
        )
        return None

    def _set_isolation_level(self, conn, isolation_level: IsolationLevel):
        """Set transaction isolation level (database-specific)"""
        try:
            if hasattr(conn, "set_isolation_level"):
                # PostgreSQL
                if isolation_level == IsolationLevel.READ_UNCOMMITTED:
                    conn.set_isolation_level(1)
                elif isolation_level == IsolationLevel.READ_COMMITTED:
                    conn.set_isolation_level(2)
                elif isolation_level == IsolationLevel.REPEATABLE_READ:
                    conn.set_isolation_level(3)
                elif isolation_level == IsolationLevel.SERIALIZABLE:
                    conn.set_isolation_level(4)
            else:
                # SQLite/MySQL - use SQL commands
                cursor = conn.cursor()
                if isolation_level != IsolationLevel.READ_COMMITTED:  # SQLite default
                    cursor.execute(
                        f"PRAGMA read_uncommitted = {'ON' if isolation_level == IsolationLevel.READ_UNCOMMITTED else 'OFF'}"
                    )

        except Exception as e:
            logger.warning(f"Could not set isolation level: {e}")

    def _set_readonly(self, conn, readonly: bool):
        """Set transaction to readonly mode"""
        try:
            cursor = conn.cursor()
            if readonly:
                # Database-specific readonly settings
                cursor.execute("SET TRANSACTION READ ONLY")
        except Exception as e:
            logger.debug(f"Could not set readonly mode: {e}")

    def _begin_transaction(self, conn):
        """Begin transaction (database-specific)"""
        try:
            if hasattr(conn, "autocommit"):
                # Ensure autocommit is off
                conn.autocommit = False

            # Explicitly begin transaction
            cursor = conn.cursor()
            cursor.execute("BEGIN")
        except Exception as e:
            logger.debug(f"Could not explicitly begin transaction: {e}")

    def _enter_connection_context(self, connection_candidate, stack: ExitStack):
        """Normalize connection acquisition for raw connections and context managers."""

        if connection_candidate is None:
            logger.debug(
                "Connector returned no connection; skipping context management registration."
            )
            return None

        enter = getattr(connection_candidate, "__enter__", None)
        exit_ = getattr(connection_candidate, "__exit__", None)

        if callable(enter) and callable(exit_):
            try:
                return stack.enter_context(connection_candidate)
            except TypeError:
                logger.debug(
                    "Connection object %s advertises context manager methods but"
                    " cannot be entered directly; falling back to manual close.",
                    type(connection_candidate).__name__,
                )

        close = getattr(connection_candidate, "close", None)
        if callable(close):
            stack.callback(close)

        return connection_candidate

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable"""
        error_str = str(error).lower()

        # Common retryable error patterns
        retryable_patterns = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "busy",
            "lock",
            "deadlock",
            "serialization",
        ]

        # Non-retryable error patterns
        non_retryable_patterns = [
            "constraint",
            "unique",
            "foreign key",
            "not null",
            "syntax error",
            "permission",
            "access denied",
        ]

        # Check non-retryable first
        for pattern in non_retryable_patterns:
            if pattern in error_str:
                return False

        # Check retryable patterns
        for pattern in retryable_patterns:
            if pattern in error_str:
                return True

        # Default to non-retryable for unknown errors
        return False


class TransactionContext:
    """Context for operations within a transaction"""

    def __init__(self, manager: TransactionManager, connection, transaction_id: str):
        self.manager = manager
        self.connection = connection
        self.transaction_id = transaction_id
        self.operations_count = 0

    def execute(self, query: str, params: list[Any] | None = None) -> ExecutionResult:
        """Execute query within the transaction context"""
        try:
            cursor = self.connection.cursor()

            # Execute query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            normalized_query = query.strip().upper()

            # Get results for SELECT queries
            if normalized_query.startswith("SELECT"):
                rows: list[dict[str, Any]] = []
                for row in cursor.fetchall():
                    if hasattr(row, "keys"):
                        # Dictionary-like row
                        rows.append(dict(row))
                    else:
                        # Tuple row - convert to dict with column names
                        column_names = (
                            [desc[0] for desc in cursor.description]
                            if cursor.description
                            else []
                        )
                        rows.append(dict(zip(column_names, row, strict=False)))

                cursor_rowcount = getattr(cursor, "rowcount", None)
                if isinstance(cursor_rowcount, int) and cursor_rowcount >= 0:
                    rowcount = cursor_rowcount
                    rowcount_source = "cursor"
                else:
                    rowcount = len(rows)
                    rowcount_source = "inferred"

                return ExecutionResult(
                    rows,
                    rowcount=rowcount,
                    rowcount_source=rowcount_source,
                )

            # For non-SELECT queries, return affected row count metadata
            raw_rowcount = getattr(cursor, "rowcount", None)
            if isinstance(raw_rowcount, int) and raw_rowcount >= 0:
                rowcount = raw_rowcount
                rowcount_source = "cursor"
            else:
                rowcount = None
                rowcount_source = "payload"

            payload: list[dict[str, Any]] = (
                [{"affected_rows": raw_rowcount}] if raw_rowcount is not None else []
            )

            return ExecutionResult(
                payload,
                rowcount=rowcount,
                rowcount_source=rowcount_source,
            )

        except Exception as e:
            logger.error(
                f"Query execution failed in transaction {self.transaction_id}: {e}"
            )
            raise DatabaseError(f"Query execution failed: {e}")
        finally:
            self.operations_count += 1

    def execute_many(self, query: str, params_list: list[list[Any]]) -> int:
        """Execute query with multiple parameter sets"""
        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount
        except Exception as e:
            logger.error(
                f"Batch execution failed in transaction {self.transaction_id}: {e}"
            )
            raise DatabaseError(f"Batch execution failed: {e}")
        finally:
            self.operations_count += 1

    def execute_script(self, script: str):
        """Execute SQL script (SQLite specific)"""
        try:
            cursor = self.connection.cursor()
            if hasattr(cursor, "executescript"):
                cursor.executescript(script)
            else:
                # Fallback for other databases - split and execute individually
                statements = script.split(";")
                for statement in statements:
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)
        except Exception as e:
            logger.error(
                f"Script execution failed in transaction {self.transaction_id}: {e}"
            )
            raise DatabaseError(f"Script execution failed: {e}")
        finally:
            self.operations_count += 1


class SavepointManager:
    """Manage savepoints within transactions for fine-grained rollback control"""

    def __init__(self, transaction_context: TransactionContext):
        self.tx_context = transaction_context
        self.savepoint_counter = 0

    @contextmanager
    def savepoint(self, name: str | None = None):
        """Create a savepoint within the current transaction"""
        if not name:
            name = f"sp_{self.savepoint_counter}"
            self.savepoint_counter += 1

        try:
            # Create savepoint
            self.tx_context.execute(f"SAVEPOINT {name}")
            logger.debug(f"Created savepoint {name}")

            yield name

        except Exception as e:
            # Rollback to savepoint
            try:
                self.tx_context.execute(f"ROLLBACK TO SAVEPOINT {name}")
                logger.warning(f"Rolled back to savepoint {name}: {e}")
            except Exception as rollback_error:
                logger.error(
                    f"Failed to rollback to savepoint {name}: {rollback_error}"
                )

            raise e

        finally:
            # Release savepoint
            try:
                self.tx_context.execute(f"RELEASE SAVEPOINT {name}")
                logger.debug(f"Released savepoint {name}")
            except Exception as e:
                logger.warning(f"Failed to release savepoint {name}: {e}")


# Convenience functions for common transaction patterns
def atomic_operation(connector):
    """Decorator for atomic database operations"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            tm = TransactionManager(connector)

            def operation():
                return func(*args, **kwargs)

            return tm.execute_with_retry(operation)

        return wrapper

    return decorator


def bulk_insert_transaction(
    connector, table: str, data: list[dict[str, Any]], batch_size: int = 1000
) -> TransactionResult:
    """Perform bulk insert with proper transaction management"""
    from .input_validator import DatabaseInputValidator

    tm = TransactionManager(connector)
    operations = []

    # Validate and prepare operations
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        # Validate batch data
        for row in batch:
            validated_row = DatabaseInputValidator.validate_insert_params(table, row)

            # Create insert operation
            columns = list(validated_row.keys())
            database_type = connector.database_type
            if isinstance(database_type, Enum):
                database_type = database_type.value
            else:
                database_type = getattr(database_type, "value", database_type)
            database_type = str(database_type).lower()

            placeholders = ",".join(
                ["?" if database_type == "sqlite" else "%s"] * len(columns)
            )

            query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
            params = list(validated_row.values())

            operations.append(
                TransactionOperation(
                    query=query,
                    params=params,
                    operation_type="insert",
                    table=table,
                    expected_rows=1,
                )
            )

    return tm.execute_atomic_operations(operations)
