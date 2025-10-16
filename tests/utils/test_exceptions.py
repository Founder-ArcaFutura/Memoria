from loguru import logger as loguru_logger

from memoria.utils.exceptions import ExceptionHandler, MemoriaError


def test_log_exception_includes_structured_fields():
    error = MemoriaError(
        message="Test error",
        error_code="TEST_ERROR",
        context={"key": "value"},
    )

    records = []

    def sink(message):
        records.append(message.record)

    handler_id = loguru_logger.add(sink)
    try:
        ExceptionHandler.log_exception(error, logger=loguru_logger)
    finally:
        loguru_logger.remove(handler_id)

    assert records, "No log records were captured"

    record = records[-1]
    assert record["extra"].get("exception_data") == error.to_dict()
    assert record["extra"].get("error_type") == "MemoriaError"
