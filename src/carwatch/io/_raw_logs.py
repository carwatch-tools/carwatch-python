"""Load raw log files exported by the CARWatch app."""

from __future__ import annotations

import json
import re
import warnings
import zipfile
from collections.abc import Sequence
from io import TextIOWrapper
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd

from carwatch.exceptions import LogParseError

PathLike: TypeAlias = str | Path
ErrorHandling: TypeAlias = Literal["raise", "warn", "ignore"]

_ENTRY_START = re.compile(r"^\d+;")
_ACTION = re.compile(r"^[a-z][a-z0-9_]*$")
_DATE_TOKEN = re.compile(r"^\d{8}$")
_COLUMNS = [
    "participant",
    "date",
    "timestamp",
    "timestamp_ms",
    "action",
    "payload",
    "source_file",
]


def load_raw_logs(
    path: PathLike | Sequence[PathLike],
    *,
    tz: str = "Europe/Berlin",
    errors: ErrorHandling = "raise",
) -> pd.DataFrame:
    """Load raw CARWatch log events.

    Parameters
    ----------
    path
        CSV file, ZIP archive, directory, or sequence of such paths. ZIP
        archives are read in memory and are not extracted.
    tz
        Timezone used to represent the Unix timestamps in the log.
    errors
        How invalid JSON payloads and non-monotonic timestamps are handled.

    Returns
    -------
    pandas.DataFrame
        One row per raw log event, indexed by ``participant``, ``date``, and
        timezone-aware ``timestamp``. The ``payload`` column contains parsed
        dictionaries.

    """
    _validate_errors(errors)
    paths = _normalize_paths(path)
    frames: list[pd.DataFrame] = []
    for current_path in paths:
        frames.extend(_load_path(current_path, tz=tz, errors=errors))

    if not frames:
        return _empty_raw_logs()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(
        ["participant", "timestamp", "source_file"],
        kind="stable",
        na_position="last",
    )
    result = result.set_index(["participant", "date", "timestamp"])
    result = result[["action", "payload", "timestamp_ms", "source_file"]]
    return result


def _empty_raw_logs() -> pd.DataFrame:
    result = pd.DataFrame(columns=_COLUMNS)
    result = result.set_index(["participant", "date", "timestamp"])
    return result[["action", "payload", "timestamp_ms", "source_file"]]


def _normalize_paths(path: PathLike | Sequence[PathLike]) -> list[Path]:
    if isinstance(path, (str, Path)):
        paths = [Path(path)]
    else:
        paths = [Path(item) for item in path]
    if not paths:
        raise ValueError("At least one log path must be supplied.")
    return paths


def _load_path(path: Path, *, tz: str, errors: ErrorHandling) -> list[pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Log path does not exist: {path}")
    if path.is_dir():
        files = [
            item
            for item in sorted(path.iterdir())
            if item.is_file()
            and item.suffix.lower() in {".csv", ".zip"}
            and not _is_hidden(item.name)
        ]
        if not files:
            raise FileNotFoundError(f"No CSV or ZIP log files found in: {path}")
        frames: list[pd.DataFrame] = []
        for file_path in files:
            frames.extend(_load_path(file_path, tz=tz, errors=errors))
        return frames
    if path.suffix.lower() == ".csv":
        return [
            _load_text(
                path.read_text(encoding="utf-8-sig"), path.name, tz=tz, errors=errors
            )
        ]
    if path.suffix.lower() == ".zip":
        return _load_zip(path, tz=tz, errors=errors)
    raise ValueError(f"Unsupported log file extension: {path.suffix}")


def _load_zip(path: Path, *, tz: str, errors: ErrorHandling) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        entries = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir()
            and entry.filename.lower().endswith(".csv")
            and not _is_hidden(entry.filename)
        ]
        for entry in sorted(entries, key=lambda item: item.filename):
            with archive.open(entry) as raw_file:
                with TextIOWrapper(raw_file, encoding="utf-8-sig") as text_file:
                    text = text_file.read()
            frames.append(
                _load_text(text, f"{path.name}!{entry.filename}", tz=tz, errors=errors)
            )
    if not frames:
        raise FileNotFoundError(f"No CSV log files found in ZIP archive: {path}")
    return frames


def _load_text(
    text: str, source_file: str, *, tz: str, errors: ErrorHandling
) -> pd.DataFrame:
    raw_entries = _split_entries(text)
    rows = [
        _parse_entry(
            entry, source_file=source_file, event_index=index, tz=tz, errors=errors
        )
        for index, entry in enumerate(raw_entries)
    ]
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=_COLUMNS)

    if not result["timestamp_ms"].is_monotonic_increasing:
        _handle_error(
            f"Timestamps are not monotonically increasing in {source_file}.",
            errors=errors,
        )

    participant, file_date = _metadata_from_filename(source_file)
    result.insert(0, "participant", participant)
    if file_date is None:
        file_date = result["timestamp"].iloc[0].normalize()
    else:
        file_date = pd.Timestamp(file_date, tz=tz)
    result.insert(1, "date", file_date)
    return result[_COLUMNS]


def _split_entries(text: str) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _ENTRY_START.match(line):
            if current:
                entries.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append("\n".join(current))
    return entries


def _parse_entry(
    entry: str,
    *,
    source_file: str,
    event_index: int,
    tz: str,
    errors: ErrorHandling,
) -> dict:
    parts = entry.split(";", maxsplit=3)
    if len(parts) < 3:
        raise LogParseError(f"Invalid log entry in {source_file}: {entry!r}")

    timestamp_raw = parts[0]
    if len(parts) == 4:
        _, action, payload_raw = parts[1:]
    elif _ACTION.fullmatch(parts[1]):
        action, payload_raw = parts[1:]
    else:
        _, action = parts[1:]
        payload_raw = "{}"

    try:
        timestamp_ms = int(timestamp_raw)
    except ValueError as exc:
        raise LogParseError(
            f"Invalid Unix timestamp in {source_file}: {timestamp_raw!r}"
        ) from exc

    payload = _parse_payload(
        payload_raw, source_file=source_file, event_index=event_index, errors=errors
    )
    timestamp = pd.to_datetime(timestamp_ms, unit="ms", utc=True).tz_convert(tz)
    return {
        "timestamp": timestamp,
        "timestamp_ms": timestamp_ms,
        "action": action,
        "payload": payload,
        "source_file": source_file,
    }


def _parse_payload(
    payload_raw: str,
    *,
    source_file: str,
    event_index: int,
    errors: ErrorHandling,
) -> dict | None:
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
        _handle_error(
            f"Invalid JSON payload in {source_file}, event {event_index}: {exc.msg}",
            errors=errors,
            exception=exc,
        )
        return None
    if not isinstance(payload, dict):
        _handle_error(
            f"JSON payload in {source_file}, event {event_index} is not an object.",
            errors=errors,
        )
        return None
    return payload


def _metadata_from_filename(
    source_file: str,
) -> tuple[str | None, str | None]:
    file_name = Path(source_file.split("!", maxsplit=1)[-1]).name
    stem = re.sub(r"\.csv$", "", file_name, flags=re.IGNORECASE)
    stem = re.sub(r"^carwatch_", "", stem, flags=re.IGNORECASE)
    parts = stem.split("_")
    has_date = bool(parts and _DATE_TOKEN.fullmatch(parts[-1]))
    date = _date_token_to_iso(parts[-1]) if has_date else None
    content = parts[:-1] if has_date else parts
    if len(content) >= 2:
        return "_".join(content[1:]), date
    if content:
        return content[0], date
    return None, date


def _date_token_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _is_hidden(file_name: str) -> bool:
    parts = Path(file_name.replace("\\", "/")).parts
    return any(part.startswith(".") or part == "__MACOSX" for part in parts)


def _validate_errors(errors: str) -> None:
    if errors not in {"raise", "warn", "ignore"}:
        raise ValueError("'errors' must be one of {'raise', 'warn', 'ignore'}.")


def _handle_error(
    message: str,
    *,
    errors: ErrorHandling,
    exception: Exception | None = None,
) -> None:
    if errors == "raise":
        raise LogParseError(message) from exception
    if errors == "warn":
        warnings.warn(message, UserWarning, stacklevel=3)
