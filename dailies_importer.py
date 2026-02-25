"""Dailies Importer

Streamlit app to combine Avid TAB-delimited text exports and ALE files into a single CSV.

Key behaviors (as requested):
- Always stamps each record with the originating uploaded filename in a first column: SourceFile.
- SourceFile is ALWAYS the first column in the output CSV (even when using a template).
- Camera Format is sourced ONLY from Input Color Space (no fallback to Format).
- Shoot Day is normalized to PREFIX_<2-digit day> by default (configurable width).
- Shoot Date is normalized to mm/dd/yyyy (handles yyyymmdd like 20251003).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------

def _normalize_header(s: str) -> str:
    """Normalize header names (strip non-printables + normalize whitespace)."""
    if s is None:
        return ""
    cleaned = "".join(ch for ch in str(s) if ch.isprintable())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _norm_key(s: str) -> str:
    """Stronger normalization for matching incoming column headers."""
    s = _normalize_header(s)
    s = s.replace("\uFFFD", "")  # occasional replacement char
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s




def _fmt_shoot_day(value: object, *, sep: str = "_", width: int = 2) -> str:
    """Normalize shoot day strings.

    Examples (width=2):
      MU030   -> MU_30
      MU_030  -> MU_30
      2U001   -> 2U_01
      2U014   -> 2U_14

    Rule: extract PREFIX and trailing digits; output PREFIX<sep><digits> where digits are:
    - right-truncated to width if longer than width
    - zero-padded to width if shorter than width
    """

    s = str(value).strip()
    if not s:
        return ""

    m = re.match(r"^(.+?)\s*[_-]?\s*(\d+)$", s)
    if not m:
        return s

    prefix, num = m.group(1).strip(), m.group(2).strip()
    if not prefix:
        return s

    try:
        w = int(width)
    except Exception:
        w = 2

    if w > 0:
        if len(num) > w:
            num = num[-w:]
        else:
            num = num.zfill(w)

    return f"{prefix}{sep}{num}"


def _fmt_shoot_date(value: object) -> str:
    """Force Shoot Date to mm/dd/yyyy.

    Handles:
      - yyyymmdd (20251003)
      - yyyy-mm-dd / yyyy/mm/dd
      - mm/dd/yyyy (normalizes zero padding)
    """

    s = str(value).strip()
    if not s:
        return ""

    # yyyymmdd
    if re.fullmatch(r"\d{8}", s):
        try:
            dt = datetime.strptime(s, "%Y%m%d")
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return s

    # yyyy-mm-dd or yyyy/mm/dd
    if re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", s):
        try:
            dt = datetime.strptime(s.replace("/", "-"), "%Y-%m-%d")
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return s

    # mm/dd/yyyy (already)
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s):
        try:
            dt = datetime.strptime(s, "%m/%d/%Y")
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return s

    return s


# -----------------------------
# Reading inputs
# -----------------------------

@dataclass(frozen=True)
class ReadOptions:
    encoding: str = "utf-8"
    errors: str = "strict"  # strict|replace|ignore
    delimiter: str = "\t"
    source_column_name: str = "SourceFile"


def _read_text_bytes(uploaded_file, *, opts: ReadOptions) -> str:
    raw = uploaded_file.getvalue()
    return raw.decode(opts.encoding, errors=opts.errors)


def _read_tsv(uploaded_file, *, opts: ReadOptions) -> pd.DataFrame:
    text = _read_text_bytes(uploaded_file, opts=opts)
    df = pd.read_csv(io.StringIO(text), sep=opts.delimiter, dtype=str, keep_default_na=False)
    df = df.fillna("")
    df = df.replace({"nan": "", "NaN": "", "NAN": ""})

    # Stamp SourceFile (always)
    src = _normalize_header(opts.source_column_name or "SourceFile")
    if src in df.columns:
        df[src] = uploaded_file.name
    else:
        df.insert(0, src, uploaded_file.name)

    return df


def _read_ale(uploaded_file, *, opts: ReadOptions) -> pd.DataFrame:
    """Very small ALE reader.

    ALE structure (common):
      - Header lines
      - 'Column' line
      - next line: tab-separated headers
      - 'Data' line
      - subsequent lines: tab-separated rows

    This parser is tolerant: it searches for 'Column' and 'Data' markers.
    """

    text = _read_text_bytes(uploaded_file, opts=opts)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    # Find markers
    col_idx = None
    data_idx = None
    for i, ln in enumerate(lines):
        if col_idx is None and ln.strip().lower() == "column":
            col_idx = i
        if ln.strip().lower() == "data":
            data_idx = i
            break

    if col_idx is None or data_idx is None or col_idx + 1 >= len(lines) or data_idx + 1 > len(lines):
        raise ValueError(f"Could not parse ALE structure in {uploaded_file.name}")

    headers = lines[col_idx + 1].split("\t")
    rows = []
    for ln in lines[data_idx + 1 :]:
        if not ln.strip():
            continue
        rows.append(ln.split("\t"))

    df = pd.DataFrame(rows, columns=headers)
    df = df.fillna("")
    df = df.astype(str)
    # Guard against literal 'nan' strings that can appear from upstream
    df = df.replace({"nan": "", "NaN": "", "NAN": ""})

    # Stamp SourceFile (always)
    src = _normalize_header(opts.source_column_name or "SourceFile")
    if src in df.columns:
        df[src] = uploaded_file.name
    else:
        df.insert(0, src, uploaded_file.name)

    return df


def _read_any(uploaded_file, *, opts: ReadOptions) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".ale"):
        return _read_ale(uploaded_file, opts=opts)
    # treat everything else as tab-delimited text
    return _read_tsv(uploaded_file, opts=opts)


# -----------------------------
# Output building
# -----------------------------

# Target -> list of possible incoming headers
COPY_MAP: dict[str, tuple[str, ...]] = {
    "SourceFile": ("SourceFile",),
    "Tape": ("Tape", "Reel", "Reel Name", "Tape Name", "tape", "reel"),
    "ASC_SAT": ("ASC_SAT", "ASC SAT"),
    "ASC_SOP": ("ASC_SOP", "ASC SOP"),
    # Camera Format ONLY from Input Color Space (no fallbacks)
    "Camera Format": ("Input Color Space", "Input color spac", "colorspaceinput", "ColorSpaceInput"),
    "Camera Letter": (
        "Camera Letter",
        "Camera",
        "CameraLetter",
        "Cam",
        "Cam Letter",
        "CamLetter",
    ),
    "Camera Roll Angle": ("Camera Roll Angle", "CameraRollAngle"),
    "Shutter Angle": ("Shutter Angle", "ShutterAngle"),
    "Camera Tilt Angle": ("Camera Tilt Angle", "CameraTiltAngle"),
    "Camera Model": (
        "Camera Model",
        "Camera model",
        "CameraModel",
        "UniqueCameraMode",
        "Uniquecameramode",
        "Camera Type",
        "CameraType",
    ),
    "ColorTemp": (
        "ColorTemp",
        "Color Temp",
        "Color Temperature",
        "Kelvin",
        "White Balance",
        "WhiteBalance",
        "WB",
    ),
    "ISO": (
        "ISO",
        "ISO.1",
        "EI",
        "Exposure Index",
        "ExposureIndex",
        "ASA",
        "ISOSpeed",
        "ISO Speed",
    ),
    "Camera Resolution": ("Resolution", "Camera Resolution", "CameraResolution"),
    "FPS": ("FPS", "Framerate", "Frame Rate"),
    "Comments": ("Comments", "Comment"),
    "Slate": ("Slate", "Scene"),
    "Clip Name": ("Take Name", "TakeName", "Name"),
    "Take Number": ("Take Number", "TakeNumber", "Take"),
    "Shoot Date": ("Shoot Date", "ShootDate"),
    "Shoot Day": ("Shoot Day", "Shoot day", "ShootDay"),
    # Removed "Detected" on purpose; keep these if present
    "Detected Secondaries": ("Detected Secondaries", "Detected seconda", "Detected Seconda"),
    "Detected Dynamics": ("Detected Dynamics", "Detected dynamic", "Detected Dynamic"),
    "Grade Info": ("Grade Info", "Grade info", "GradeInfo"),
}


DEFAULT_OUTPUT_COLUMNS: list[str] = [
    "SourceFile",
    "Tape",
    "ASC_SAT",
    "ASC_SOP",
    "Camera Format",
    "Camera Letter",
    "Camera Roll Angle",
    "Shutter Angle",
    "Camera Tilt Angle",
    "Camera Model",
    "ColorTemp",
    "ISO",
    "Camera Resolution",
    "FPS",
    "Comments",
    "Slate",
    "Clip Name",
    "Take Number",
    "Shoot Date",
    "Shoot Day",
    "Detected Secondaries",
    "Detected Dynamics",
    "Grade Info",
]


def _dedupe_join(values: Iterable[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        vv = str(v).strip()
        if not vv:
            continue
        if vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return "; ".join(out)

def _coalesce_first_nonempty(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return a Series with the first non-empty value across cols (left-to-right)."""
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    block = df[cols].astype(str)
    block = block.replace(r"^\s*$", pd.NA, regex=True)
    return block.bfill(axis=1).iloc[:, 0].fillna("")

def _build_output(
    combined_sources: pd.DataFrame,
    *,
    output_columns: list[str],
    shoot_day_separator: str = "_",
    shoot_day_number_width: int = 2,
) -> pd.DataFrame:
    """Build final output dataframe in the requested schema."""

    # Normalize incoming columns for matching
    src_df = combined_sources.copy()
    src_df.columns = [_normalize_header(c) for c in src_df.columns]
    src_df = src_df.fillna("")
    src_df = src_df.replace({"nan": "", "NaN": "", "NAN": ""})

    # Build a lookup from normalized key -> list of matching actual column names (preserve order)
    key_to_cols: dict[str, list[str]] = {}
    for c in src_df.columns:
        key_to_cols.setdefault(_norm_key(c), []).append(c)

    def pick_cols(*aliases: str) -> list[str]:
        """Return all matching columns for the alias list, in alias order, preserving source order."""
        picked: list[str] = []
        seen: set[str] = set()
        for a in aliases:
            cols = key_to_cols.get(_norm_key(a), [])
            for col in cols:
                if col in seen:
                    continue
                seen.add(col)
                picked.append(col)
        return picked

    out = pd.DataFrame(index=src_df.index)

    # Always ensure SourceFile exists in source and is first in output schema
    out_cols = [_normalize_header(c) for c in output_columns]
    src_col = _normalize_header("SourceFile")
    out_cols = [c for c in out_cols if c != src_col]
    out_cols.insert(0, src_col)

    # Comments: combine + dedupe if multiple comment-like columns exist
    comments_key = _normalize_header("Comments")
    if comments_key in out_cols:
        comment_cols = pick_cols("Comments", "Comment", "Notes", "Note")
        if comment_cols:
            out[comments_key] = src_df[comment_cols].astype(str).apply(lambda r: _dedupe_join(r.tolist()), axis=1)
            out[comments_key] = out[comments_key].replace({"nan": "", "NaN": "", "NAN": ""})
        else:
            out[comments_key] = ""

    # Populate requested columns (coalesce across all matching candidates)
    for target in out_cols:
        if target == comments_key:
            continue
        aliases = COPY_MAP.get(target, (target,))
        cols = pick_cols(*aliases)
        if not cols:
            out[target] = ""
        else:
            out[target] = _coalesce_first_nonempty(src_df, cols).replace({"nan": "", "NaN": "", "NAN": ""})

    # Legacy conform rules (from previous version):
    # - Slate defaults to Scene
    # - Take Name defaults to Name
    # - Take Number defaults to Take
    slate_col = _normalize_header("Slate")
    clip_name_col = _normalize_header("Clip Name")
    take_num_col = _normalize_header("Take Number")

    def _blank(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().eq("")

    # If Slate came through as blank, try Scene from the source df
    if slate_col in out.columns:
        scene_cols = pick_cols("Scene")
        if scene_cols:
            mask = _blank(out[slate_col])
            out.loc[mask, slate_col] = _coalesce_first_nonempty(src_df.loc[mask], scene_cols)

    if clip_name_col in out.columns:
        name_cols = pick_cols("Name")
        if name_cols:
            mask = _blank(out[clip_name_col])
            out.loc[mask, clip_name_col] = _coalesce_first_nonempty(src_df.loc[mask], name_cols)

    if take_num_col in out.columns:
        take_cols = pick_cols("Take")
        if take_cols:
            mask = _blank(out[take_num_col])
            out.loc[mask, take_num_col] = _coalesce_first_nonempty(src_df.loc[mask], take_cols)

    # Clean any newly introduced 'nan' strings
    out = out.replace({"nan": "", "NaN": "", "NAN": ""})

    # Shoot Day formatting
    sd_col = _normalize_header("Shoot Day")
    if sd_col in out.columns:
        out[sd_col] = out[sd_col].map(lambda v: _fmt_shoot_day(v, sep=shoot_day_separator, width=shoot_day_number_width))

    # Shoot Date formatting
    sdate_col = _normalize_header("Shoot Date")
    if sdate_col in out.columns:
        out[sdate_col] = out[sdate_col].map(_fmt_shoot_date)

    # Reorder
    out = out.loc[:, out_cols]

    return out


def _load_template_columns(uploaded_template) -> list[str]:
    """Load headers from a template CSV (first row only)."""
    raw = uploaded_template.getvalue()
    # Try utf-8-sig first (common for Excel)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            df = pd.read_csv(io.StringIO(text), nrows=0)
            cols = [_normalize_header(c) for c in df.columns.tolist()]
            if cols:
                return cols
        except Exception:
            continue
    raise ValueError("Could not read template CSV headers")


# -----------------------------
# Streamlit UI
# -----------------------------

def run_app() -> None:
    st.set_page_config(page_title="Dailies Importer", layout="wide")
    st.title("Dailies Importer")

    with st.sidebar:
        # Allow clearing the file uploader by changing its key
        if "uploaded_files_key" not in st.session_state:
            st.session_state["uploaded_files_key"] = 0
        st.header("Inputs")
        if st.button("Clear uploaded files"):
            st.session_state["uploaded_files_key"] += 1
        uploaded_files = st.file_uploader(
            "Upload ALE and/or TAB-delimited TXT files",
            type=["ale", "txt", "tab", "tsv", "csv"],
            accept_multiple_files=True,
            key=f"uploaded_files_{st.session_state['uploaded_files_key']}",
        )
        file_count = len(uploaded_files) if uploaded_files else 0
        st.caption(f"Files selected: **{file_count}**")

        template = st.file_uploader(
            "Optional: Output template CSV (header order)",
            type=["csv"],
            accept_multiple_files=False,
        )

        st.header("Options")
        encoding = st.selectbox(
            "Encoding",
            options=["utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman"],
            index=0,
        )
        errors = st.selectbox("Encoding errors", options=["strict", "replace", "ignore"], index=0)

        st.subheader("Shoot Day")
        shoot_day_separator = st.text_input("Separator", value="_")
        shoot_day_number_width = st.number_input(
            "Number digits (zero-pad)",
            min_value=0,
            max_value=6,
            value=2,
            step=1,
            help="Width of the numeric portion. If source has more digits, keeps rightmost digits (e.g. 030 -> 30 for width=2).",
        )

    if file_count == 0:
        st.info("Upload at least one ALE or tab-delimited text file to begin.")
        return

    opts = ReadOptions(encoding=encoding.strip() or "utf-8", errors=errors)

    # Read and combine
    frames: list[pd.DataFrame] = []
    read_errors: list[str] = []

    for f in uploaded_files:
        try:
            frames.append(_read_any(f, opts=opts))
        except Exception as e:
            read_errors.append(f"{f.name}: {e}")

    st.caption(f"Files read successfully: **{len(frames)}** / **{file_count}**")

    if read_errors:
        st.error("Some files could not be read:")
        for msg in read_errors:
            st.write(f"- {msg}")
        if not frames:
            return

    combined = pd.concat(frames, ignore_index=True)

    # Determine output schema
    if template is not None:
        try:
            output_columns = _load_template_columns(template)
        except Exception as e:
            st.error(f"Template could not be read: {e}")
            output_columns = DEFAULT_OUTPUT_COLUMNS.copy()
    else:
        output_columns = DEFAULT_OUTPUT_COLUMNS.copy()

    # Always include these critical identifiers even when a template is used
    must_have = ["SourceFile"]
    norm_cols = [_normalize_header(c) for c in output_columns]
    for c in must_have:
        cc = _normalize_header(c)
        if cc not in norm_cols:
            norm_cols.insert(0, cc)
    output_columns = norm_cols

    output_df = _build_output(
        combined,
        output_columns=output_columns,
        shoot_day_separator=shoot_day_separator,
        shoot_day_number_width=int(shoot_day_number_width),
    )

    # Warn on duplicate Tape names (we do NOT remove/merge rows)
    tape_col = _normalize_header("Tape")
    if tape_col in output_df.columns:
        tape_series = output_df[tape_col].astype(str).str.strip()

        # consider non-empty only
        nonempty = tape_series.ne("")
        dup_mask = nonempty & tape_series.duplicated(keep=False)
        dup_rows = int(dup_mask.sum())

        if dup_rows > 0:
            # number of distinct duplicate tape values
            dup_groups = int(tape_series[dup_mask].drop_duplicates().shape[0])
            st.warning(
                f"⚠️ Duplicate Tape names detected: **{dup_rows}** rows across **{dup_groups}** duplicate Tape values. "
                "No rows were removed."
            )

            with st.expander("Show duplicate Tape examples"):
                # Show counts per Tape
                counts = (
                    tape_series[dup_mask]
                    .value_counts()
                    .rename_axis(tape_col)
                    .reset_index(name="Count")
                    .sort_values("Count", ascending=False)
                    .head(50)
                )
                st.dataframe(counts, use_container_width=True)

                st.caption("Sample rows (first 50) with duplicate Tape names")
                sample_cols = [
                    c
                    for c in [
                        tape_col,
                        _normalize_header("Clip Name"),
                        _normalize_header("Take Number"),
                        _normalize_header("Shoot Date"),
                        _normalize_header("Shoot Day"),
                        _normalize_header("SourceFile"),
                    ]
                    if c in output_df.columns
                ]
                sample = (
                    output_df.loc[dup_mask, sample_cols]
                    .sort_values(sample_cols)
                    .head(50)
                    .reset_index(drop=True)
                )
                st.dataframe(sample, use_container_width=True)

    st.subheader("Preview")
    st.dataframe(output_df, use_container_width=True, height=420)

    # Export
    csv_bytes = output_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download combined CSV",
        data=csv_bytes,
        file_name="dailies_combined.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    run_app()