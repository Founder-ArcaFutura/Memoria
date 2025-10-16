from __future__ import annotations

import json
from flask import Blueprint, current_app, render_template_string, request
from loguru import logger
from pydantic import ValidationError

from memoria.schemas import validate_memory_entry
from .spatial_utils import upsert_spatial_metadata
from .request_parsers import (
    CoordinateParsingError,
    parse_optional_coordinate,
    timestamp_from_x,
)


manual_entry_bp = Blueprint("manual_entry", __name__)


_FORM_HTML = """<!doctype html>
<title>Manual Memory Entry</title>
<h1>Manual Memory Entry</h1>
<form method=post>
  <label>Anchor: <input type=text name=anchor required></label><br>
  <label>Text:<br><textarea name=text rows=4 cols=40 required></textarea></label><br>
  <label>X Coord: <input type=number step=any name=x_coord></label><br>
  <label>Y Coord: <input type=number step=any name=y_coord></label><br>
  <label>Z Coord: <input type=number step=any name=z_coord></label><br>
  <label>Symbolic Anchors (comma separated): <input type=text name=symbolic_anchors></label><br>
  <input type=submit value="Store">
</form>
{% if message %}<p>{{ message }}</p>{% endif %}
"""


@manual_entry_bp.route("/manual", methods=["GET", "POST"])
def manual_entry():
    """Render a simple form for storing memories manually."""
    if request.method == "POST":
        try:
            memoria = current_app.config["memoria"]
            db_path = current_app.config.get("DB_PATH")
            db_manager = getattr(memoria, "db_manager", None)
            namespace = getattr(memoria, "namespace", "default") or "default"
            data = {
                "anchor": request.form.get("anchor", "").strip(),
                "text": request.form.get("text", "").strip(),
            }
            data["tokens"] = len(data["text"].split())
            try:
                for field in ["x_coord", "y_coord", "z_coord"]:
                    data[field] = parse_optional_coordinate(
                        request.form.get(field), field
                    )
            except CoordinateParsingError as err:
                return (
                    render_template_string(_FORM_HTML, message=str(err)),
                    400,
                )
            anchors_raw = request.form.get("symbolic_anchors", "")
            anchors_list = [a.strip() for a in anchors_raw.split(",") if a.strip()]
            data["symbolic_anchors"] = anchors_list or None

            ts_str = request.form.get("timestamp")
            if ts_str in (None, ""):
                derived_ts = timestamp_from_x(data.get("x_coord"))
                if derived_ts is not None:
                    data["timestamp"] = derived_ts

            entry = validate_memory_entry(data)
            memory_id = memoria.store_memory(
                anchor=entry.anchor,
                text=entry.text,
                tokens=entry.tokens,
                timestamp=entry.timestamp,
                x_coord=entry.x_coord,
                y=entry.y_coord,
                z=entry.z_coord,
                symbolic_anchors=entry.symbolic_anchors,
            )

            if any(
                v is not None
                for v in [
                    entry.x_coord,
                    entry.y_coord,
                    entry.z_coord,
                    entry.symbolic_anchors,
                ]
            ):
                upsert_spatial_metadata(
                    memory_id=memory_id,
                    namespace=namespace,
                    db_path=db_path,
                    db_manager=db_manager,
                    timestamp=entry.timestamp,
                    x=entry.x_coord,
                    y=entry.y_coord,
                    z=entry.z_coord,
                    symbolic_anchors=entry.symbolic_anchors,
                    ensure_timestamp=True,
                )

            msg = f"Stored memory {memory_id}"
            return render_template_string(_FORM_HTML, message=msg)
        except ValidationError as e:  # pragma: no cover - validated elsewhere
            err = json.loads(e.json())
            return render_template_string(_FORM_HTML, message=str(err)), 422
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"[ERROR] {e}")
            return render_template_string(_FORM_HTML, message=str(e)), 500
    return render_template_string(_FORM_HTML)
