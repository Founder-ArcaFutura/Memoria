#!/usr/bin/env python3
"""
MySQL Support Test Suite for Memoria 0.9 alpha
Tests cross-database compatibility with MySQL using SQLAlchemy
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:  # pragma: no cover - optional dependency when running as a script
    import pytest
except ImportError:  # pragma: no cover - allow running without pytest installed
    pytest = None

# Add the memoria package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils.test_utils import get_test_model


def check_mysql_availability():
    """Check if MySQL is available and accessible"""
    try:
        import mysql.connector

        # Test connection to MySQL server
        conn = mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            password="",  # Adjust as needed
        )

        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        cursor.fetchall()

        # Check if test database exists, create if not
        cursor.execute("CREATE DATABASE IF NOT EXISTS memoria_test")
        cursor.execute("USE memoria_test")

        conn.close()
        return True, "MySQL server is available"

    except ImportError:
        return (
            False,
            "mysql-connector-python not installed. Install with: pip install mysql-connector-python",
        )
    except Exception as e:
        return False, f"MySQL server not accessible: {e}"


def run_mysql_test_scenario(test_name, provider_config, test_inputs, database_url):
    """Run a test scenario with MySQL backend"""
    print(f"🧪 Running MySQL test: {test_name}")

    try:
        from memoria import Memoria

        # Initialize Memoria with MySQL database
        memory = Memoria(database_connect=database_url, template="basic", verbose=True)

        # Test database initialization
        print("   📋 Testing database initialization...")
        db_info = memory.db_manager.get_database_info()
        assert (
            db_info["database_type"] == "mysql"
        ), f"Expected MySQL, got {db_info['database_type']}"
        print(f"   ✅ Connected to MySQL: {db_info}")

        # Test memory storage and retrieval
        print("   💾 Testing memory operations...")

        session_id = f"test_session_{int(time.time())}"

        # Store chat history
        chat_id = f"chat_{int(time.time())}"
        memory.db_manager.store_chat_history(
            chat_id=chat_id,
            user_input="Hello, can you help me with MySQL testing?",
            ai_output="Of course! I'll help you test MySQL integration with Memoria.",
            timestamp=datetime.now(),
            session_id=session_id,
            model=get_test_model(),
            namespace="mysql_test",
            tokens_used=75,
            metadata={"test": "mysql_integration", "version": "0.9.0a0"},
        )

        # Retrieve chat history
        history = memory.db_manager.get_chat_history(
            namespace="mysql_test", session_id=session_id
        )
        assert len(history) > 0, "No chat history retrieved"
        assert history[0]["chat_id"] == chat_id, "Chat ID mismatch"
        print(f"   ✅ Stored and retrieved {len(history)} chat records")

        # Test memory search functionality
        print("   🔍 Testing MySQL FULLTEXT search...")

        # Search should work even with empty results initially
        resp = memory.db_manager.search_memories(
            "MySQL testing", namespace="mysql_test"
        )
        results = resp.get("results", resp)
        print(
            f"   📊 Search returned {len(results)} results (expected 0 for new database)"
        )

        # Test memory statistics
        print("   📈 Testing memory statistics...")
        stats = memory.db_manager.get_memory_stats("mysql_test")
        expected_stats = [
            "chat_history_count",
            "short_term_count",
            "long_term_count",
            "database_type",
        ]
        for stat in expected_stats:
            assert stat in stats, f"Missing statistic: {stat}"

        assert (
            stats["database_type"] == "mysql"
        ), f"Database type mismatch: {stats['database_type']}"
        assert (
            stats["chat_history_count"] >= 1
        ), f"Chat history count incorrect: {stats['chat_history_count']}"
        print(
            f"   ✅ Statistics: {stats['chat_history_count']} chats, {stats['short_term_count']} short-term, {stats['long_term_count']} long-term"
        )

        # Test database cleanup
        print("   🧹 Testing database cleanup...")
        memory.db_manager.clear_memory(namespace="mysql_test")
        stats_after = memory.db_manager.get_memory_stats("mysql_test")
        assert stats_after["chat_history_count"] == 0, "Cleanup failed"
        print("   ✅ Database cleanup successful")

        # Close connections
        memory.db_manager.close()

        print(f"✅ MySQL test '{test_name}' PASSED")
        return True

    except Exception as e:
        print(f"❌ MySQL test '{test_name}' FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_mysql_anchor_route_test(database_url):
    """Validate that the /memory/anchor route works with MySQL JSON anchors."""

    print("🪝 Testing MySQL /memory/anchor route...")

    from memoria_server.api.app_factory import create_app

    original_db_url = os.environ.get("DATABASE_URL")
    app = None
    memoria_instance = None

    try:
        os.environ["DATABASE_URL"] = database_url
        app = create_app()
        app.testing = True
        client = app.test_client()

        memoria_instance = app.config["memoria"]
        anchor_label = "mysql_symbolic_anchor"
        anchor_text = "MySQL anchor route memory"

        memoria_instance.store_memory(
            anchor=anchor_label,
            text=anchor_text,
            tokens=10,
            x_coord=0.0,
            y=0.0,
            z=0.0,
            symbolic_anchors=[anchor_label],
        )

        response = client.get(
            "/memory/anchor",
            query_string={"anchor": anchor_label},
        )

        assert (
            response.status_code == 200
        ), f"Unexpected status {response.status_code}: {response.get_data(as_text=True)}"

        payload = response.get_json()
        texts = {item.get("text") for item in payload}
        assert (
            anchor_text in texts
        ), f"Stored anchor memory not returned by API: {payload}"

        print("   ✅ MySQL anchor route returned stored memory")
        return True

    except Exception as e:
        print(f"   ❌ MySQL anchor route test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            if memoria_instance is not None:
                try:
                    memoria_instance.disable()
                except Exception:
                    pass
                try:
                    memoria_instance.db_manager.clear_memory(
                        namespace=memoria_instance.namespace
                    )
                except Exception as cleanup_exc:  # pragma: no cover - diagnostic aid
                    print(f"   ⚠️  Cleanup warning: {cleanup_exc}")
                try:
                    memoria_instance.db_manager.close()
                except Exception:
                    pass
        finally:
            if original_db_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = original_db_url


def run_mysql_anchor_search_filter_test(database_url):
    """Ensure anchor-filtered searches work with MySQL JSON expressions."""

    print("🧠 Testing MySQL Anchor Search Filters")

    try:
        from memoria import Memoria
    except ImportError as exc:  # pragma: no cover - environment misconfiguration
        message = f"Memoria import failed: {exc}"
        if pytest is not None:
            pytest.skip(message)
        else:
            print(f"   ⚠️ {message}")
            return True

    namespace = "mysql_anchor_filter_regression"
    anchor_value = "mysql_json_filter_anchor"
    stored_text = "MySQL anchor filter regression memory"

    memory = None
    try:
        memory = Memoria(
            database_connect=database_url,
            namespace=namespace,
            conscious_ingest=False,
            auto_ingest=False,
            enable_short_term=False,
            verbose=True,
        )

        memory.db_manager.clear_memory(namespace)

        memory.store_memory(
            anchor="mysql_anchor_filter_regression",
            text=stored_text,
            tokens=10,
            x_coord=0.0,
            y=0.0,
            z=0.0,
            symbolic_anchors=[anchor_value, "shared_mysql_anchor_filter"],
        )

        response = memory.db_manager.search_memories(
            anchor_value,
            namespace=namespace,
            limit=5,
            anchors=[anchor_value],
            use_anchor=True,
            use_fuzzy=False,
        )
        results = response.get("results", response)
        print(
            f"   ✅ Retrieved {len(results)} memories for MySQL anchor-filtered search"
        )

        assert any(
            (res.get("text") or res.get("summary")) == stored_text for res in results
        ), "Anchor-filtered search should return the stored MySQL memory"

        print("   ✅ MySQL anchor search filter succeeded")
        return True

    except Exception as exc:
        print(f"   ❌ MySQL anchor search filter test failed: {exc}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if memory is not None:
            try:
                memory.db_manager.clear_memory(namespace)
            except Exception:
                pass
            memory.db_manager.close()


def run_mysql_fulltext_test(database_url):
    """Test MySQL FULLTEXT search capabilities specifically"""
    print("🔍 Testing MySQL FULLTEXT search capabilities...")

    try:
        from memoria import Memoria

        memory = Memoria(database_connect=database_url, template="basic", verbose=True)

        # Check if FULLTEXT indexes are properly created
        with memory.db_manager.SessionLocal() as session:
            # Test FULLTEXT search capabilities
            from sqlalchemy import text

            result = session.execute(
                text(
                    """
                SELECT COUNT(*) as index_count
                FROM information_schema.STATISTICS
                WHERE table_schema = DATABASE()
                AND index_type = 'FULLTEXT'
                AND table_name IN ('short_term_memory', 'long_term_memory')
            """
                )
            )

            index_count = result.fetchone()[0]
            print(f"   📊 Found {index_count} FULLTEXT indexes")

            if index_count >= 2:
                print("   ✅ MySQL FULLTEXT indexes properly configured")
                fulltext_available = True
            else:
                print("   ⚠️  FULLTEXT indexes not found, search will use fallback")
                fulltext_available = False

        memory.db_manager.close()
        return fulltext_available

    except Exception as e:
        print(f"   ❌ FULLTEXT test failed: {e}")
        return False


def run_mysql_performance_test(database_url):
    """Test MySQL performance with multiple operations"""
    print("⚡ Testing MySQL performance...")

    try:
        from memoria import Memoria

        memory = Memoria(
            database_connect=database_url,
            template="basic",
            verbose=False,  # Reduce verbosity for performance test
        )

        start_time = time.time()
        session_id = f"perf_test_{int(time.time())}"

        # Bulk insert test
        print("   💾 Testing bulk operations...")
        for i in range(10):
            memory.db_manager.store_chat_history(
                chat_id=f"perf_chat_{i}",
                user_input=f"Performance test message {i}",
                ai_output=f"Response to performance test {i}",
                timestamp=datetime.now(),
                session_id=session_id,
                model=get_test_model(),
                namespace="mysql_perf_test",
                tokens_used=50,
                metadata={"test": "performance", "iteration": i},
            )

        # Test retrieval performance
        history = memory.db_manager.get_chat_history(
            namespace="mysql_perf_test", limit=20
        )
        assert len(history) >= 10, f"Expected at least 10 records, got {len(history)}"

        # Test search performance
        resp = memory.db_manager.search_memories(
            "performance test", namespace="mysql_perf_test"
        )
        results = resp.get("results", resp)

        elapsed_time = time.time() - start_time
        print(f"   ✅ Performance test completed in {elapsed_time:.2f}s")
        print(
            f"   📊 Processed 10 inserts, retrieved {len(history)} records, found {len(results)} search results"
        )

        # Cleanup
        memory.db_manager.clear_memory(namespace="mysql_perf_test")
        memory.db_manager.close()

        return elapsed_time < 5.0  # Should complete within 5 seconds

    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False


def main():
    """Run all MySQL integration tests"""
    print("🚀 Memoria MySQL Integration Test Suite 0.9 alpha")
    print("=" * 60)

    # Check MySQL availability
    print("🔧 Checking MySQL availability...")
    mysql_available, mysql_message = check_mysql_availability()
    print(f"   {mysql_message}")

    if not mysql_available:
        print("\n❌ MySQL tests cannot run. Please ensure:")
        print("   1. MySQL server is running on localhost:3306")
        print(
            "   2. mysql-connector-python is installed: pip install mysql-connector-python"
        )
        print(
            "   3. MySQL user 'root' has access (adjust credentials in script if needed)"
        )
        return 1

    # MySQL connection URL - use mysql+mysqlconnector for the mysql-connector-python driver
    database_url = "mysql+mysqlconnector://root:@127.0.0.1:3306/memoria_test"

    print(f"\n📊 Testing with database: {database_url}")

    # Load test inputs if available
    test_inputs_path = Path(__file__).parent.parent / "test_inputs.json"
    test_inputs = {}
    if test_inputs_path.exists():
        with open(test_inputs_path) as f:
            test_inputs = json.load(f)

    # Test scenarios
    test_scenarios = [
        {
            "name": "MySQL Basic Integration",
            "provider": {"type": "test"},
        },
        {
            "name": "MySQL Cross-Database Compatibility",
            "provider": {"type": "test"},
        },
    ]

    passed_tests = 0
    total_tests = (
        len(test_scenarios) + 4
    )  # + anchor route, anchor search, performance, and fulltext tests

    # Run basic integration tests
    for scenario in test_scenarios:
        if run_mysql_test_scenario(
            test_name=scenario["name"],
            provider_config=scenario["provider"],
            test_inputs=test_inputs,
            database_url=database_url,
        ):
            passed_tests += 1

    # Run anchor route test
    print(f"\n{'-' * 60}")
    if run_mysql_anchor_route_test(database_url):
        passed_tests += 1

    # Run anchor-filtered search test
    print(f"\n{'-' * 60}")
    if run_mysql_anchor_search_filter_test(database_url):
        passed_tests += 1

    # Run FULLTEXT search test
    print(f"\n{'-' * 60}")
    if run_mysql_fulltext_test(database_url):
        passed_tests += 1

    # Run performance test
    print(f"\n{'-' * 60}")
    if run_mysql_performance_test(database_url):
        passed_tests += 1

    # Final results
    print(f"\n{'=' * 60}")
    print(f"🏁 MySQL Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print(
            "🎉 All MySQL tests passed! Cross-database compatibility is working correctly."
        )
        return 0
    else:
        print("⚠️  Some MySQL tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
