from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    """Metadata for a coordinate axis."""

    description: str
    min: float | None = None
    max: float | None = None


X_AXIS = Axis(
    description="Temporal offset in days relative to the present",
)
Y_AXIS = Axis(
    description="-15 = private; +15 = public",
    min=-15.0,
    max=15.0,
)
Z_AXIS = Axis(
    description="-15 = sensory/physical; +15 = abstract/intellectual",
    min=-15.0,
    max=15.0,
)

__all__ = ["Axis", "X_AXIS", "Y_AXIS", "Z_AXIS"]
