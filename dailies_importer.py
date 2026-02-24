from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


def _normalize_header(name: str) -> str:
    """Normalize header names (strip non-printables + trim whitespace)."""
    if name is None:
        return ""
    return "".join(ch for ch in str(name) if ch.isprintable()).strip()


def _norm_key(name: str) -> str:
    """Aggressive key normalization for header alias matching (case/space/punct-insensitive)."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    return "".join(ch for ch in s if ch.isalnum())


def _coalesce_first_nonempty(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return a Series with the first non-empty value across cols (left-to-right)."""
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    block = df[cols].astype(str)
    block = block.replace(r"^\s*$", pd.NA, regex=True)
    return block.bfill(axis=1).iloc[:, 0].fillna("")


def _dedupe_join(values: List[str], sep: str = " | ") -> str:
    """Join unique, non-empty strings preserving first-seen order."""
    seen = set()
    out: list[str] = []
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return sep.join(out)


def _sanitize_template_headers(headers: list[str]) -> list[str]:
    """
    Apply user-defined corrections to the template schema.

    Drops mistaken columns:
      - Source Clip Versions
      - Date Created
      - CameraFormat (duplicate)
      - ISO.1 (duplicate)

    Keeps only one Camera Format and one ISO.
    """
    drop = {
        _normalize_header("Source Clip Versions"),
        _normalize_header("Date Created"),
        _normalize_header("CameraFormat"),
        _normalize_header("ISO.1"),
    }

    cleaned: list[str] = []
    for h in headers:
        nh = _normalize_header(h)
        if not nh or nh in drop:
            continue
        cleaned.append(nh)

    # If only 'CameraFormat' exists, normalize it to 'Camera Format'
    if _normalize_header("Camera Format") not in cleaned and _normalize_header("CameraFormat") in [
        _normalize_header(x) for x in headers
    ]:
        cleaned = [
            _normalize_header("Camera Format") if c == _normalize_header("CameraFormat") else c
            for c in cleaned
        ]

    # If only 'ISO.1' exists, normalize it back to ISO
    if _normalize_header("ISO") not in cleaned and _normalize_header("ISO.1") in [
        _normalize_header(x) for x in headers
    ]:
        cleaned = [
            _normalize_header("ISO") if c == _normalize_header("ISO.1") else c
            for c in cleaned
        ]

    return cleaned


def _load_template_columns(path: Path, encoding: str = "utf-8") -> list[str]:
    """Load template CSV headers (ShotGrid import schema) and sanitize them."""
    try:
        tpl = pd.read_csv(path, nrows=0, dtype=str, encoding=encoding, keep_default_na=False)
    except Exception:
        tpl = pd.read_csv(
            path,
            nrows=0,
            dtype=str,
            encoding="utf-8",
            encoding_errors="replace",
            keep_default_na=False,
        )

    headers = [str(c) for c in tpl.columns]
    return _sanitize_template_headers(headers)


@dataclass(frozen=True)
class CombineOptions:
    delimiter: str = "\t"
    encoding: str = "utf-8"
    keep_as_text: bool = True
    add_source_column: bool = True
    source_column_name: str = "SourceFile"
    errors: str = "strict"  # strict|replace|ignore


class TabCombineError(Exception):
    pass


def _read_tsv(path: Path, opts: CombineOptions) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            sep=opts.delimiter,
            encoding=opts.encoding,
            encoding_errors=opts.errors,
            dtype=str if opts.keep_as_text else None,
            na_filter=False,
            keep_default_na=False,
            engine="python",
        )
    except Exception as e:
        raise TabCombineError(f"Failed to read '{path}': {e}") from e

    df.columns = [_normalize_header(c) for c in df.columns]

    if opts.add_source_column:
        source_name = _normalize_header(opts.source_column_name)
        df.insert(0, source_name, path.name)

    return df


def _read_ale(path: Path, opts: CombineOptions) -> pd.DataFrame:
    """Read an Avid ALE file into a DataFrame."""
    try:
        raw = path.read_text(encoding=opts.encoding, errors=opts.errors)
    except Exception as e:
        raise TabCombineError(f"Failed to read '{path}': {e}") from e

    lines = raw.splitlines()

    def _find_line_index(label: str) -> int:
        for i, line in enumerate(lines):
            if line.strip().lower() == label.lower():
                return i
        return -1

    col_idx = _find_line_index("Column")
    data_idx = _find_line_index("Data")
    if col_idx < 0 or data_idx < 0 or data_idx <= col_idx + 1:
        raise TabCombineError(
            f"File '{path.name}' does not look like a valid ALE (missing Column/Data sections)."
        )

    header_line = lines[col_idx + 1]

    # Determine delimiter from Heading section when present (default to tabs)
    delim = "\t"
    for line in lines[:col_idx]:
        if line.upper().startswith("FIELD_DELIM"):
            parts = line.split("\t")
            if len(parts) >= 2:
                val = parts[1].strip().upper()
                if val == "TABS":
                    delim = "\t"
                elif val == "COMMA":
                    delim = ","
                elif val == "SEMICOLON":
                    delim = ";"
            break

    # Parse headers + data rows using csv so we respect empty fields
    reader = csv.reader([header_line], delimiter=delim)
    headers = next(reader, [])
    headers = [_normalize_header(h) for h in headers]

    data_lines = lines[data_idx + 1 :]
    data_rows: list[list[str]] = []
    if data_lines:
        r = csv.reader(data_lines, delimiter=delim)
        for row in r:
            if not row or all(str(x).strip() == "" for x in row):
                continue
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[: len(headers)]
            data_rows.append([str(x) for x in row])

    df = pd.DataFrame(data_rows, columns=headers)

    # Alias map to align ALE variants to common TSV-style headers.
    alias_to_target = {
        _norm_key("Shoot Day"): "Shoot day",
        _norm_key("ShootDate"): "Shoot Date",
        _norm_key("Shoot Date"): "Shoot Date",
        _norm_key("Color Space Notes"): "Color space note",
        _norm_key("Color Space Note"): "Color space note",
        _norm_key("ColorSpaceNotes"): "Color space note",
        _norm_key("Input Color Space"): "Input color spac",
        _norm_key("colorspaceinput"): "Input color spac",
        _norm_key("ColorSpaceInput"): "Input color spac",
        _norm_key("Output Color Space"): "colorspaceoutput",
        _norm_key("colorspaceoutput"): "colorspaceoutput",
        _norm_key("ColorSpaceOutput"): "colorspaceoutput",
        _norm_key("Camera #"): "Camera",
        _norm_key("Camera#"): "Camera",
        _norm_key("Camera"): "Camera",
        _norm_key("Camera Model"): "Camera model",
        _norm_key("CameraModel"): "Camera model",
        _norm_key("Grade Info"): "Grade info",
        _norm_key("GradeInfo"): "Grade info",
        _norm_key("Detected Secondaries"): "Detected seconda",
        _norm_key("Detected Seconda"): "Detected seconda",
        _norm_key("Detected Dynamics"): "Detected dynamic",
        _norm_key("Detected Dynamic"): "Detected dynamic",
        _norm_key("ASC_SOP"): "ASC_SOP",
        _norm_key("ASC SOP"): "ASC_SOP",
        _norm_key("FPS"): "FPS",
        _norm_key("Camera FPS"): "FPS",
        _norm_key("VideoFramerate"): "FPS",
        _norm_key("Framerate"): "FPS",
        _norm_key("Resolution"): "Resolution",
        _norm_key("Image Size"): "Image Size",
        _norm_key("Name"): "Name",
        _norm_key("Tape"): "Tape",
        _norm_key("Scene"): "Scene",
        _norm_key("Take"): "Take",
        _norm_key("Start"): "Start",
        _norm_key("End"): "End",
        _norm_key("Duration"): "Duration",
        _norm_key("Frames"): "Frames",
        _norm_key("Camroll"): "Camroll",
    }

    preferred_source_key_by_target = {
        "FPS": _norm_key("Framerate"),
        "Camera": _norm_key("Camera"),
        "Input color spac": _norm_key("Input Color Space"),
        "Camera model": _norm_key("Camera Model"),
    }

    target_order: list[str] = []
    target_to_sources: dict[str, list[str]] = {}

    for c in df.columns:
        key = _norm_key(c)
        target = alias_to_target.get(key, c)
        if target not in target_to_sources:
            target_to_sources[target] = []
            target_order.append(target)
        target_to_sources[target].append(c)

    out = pd.DataFrame(index=df.index)
    for target in target_order:
        sources = target_to_sources.get(target, [])
        if len(sources) <= 1:
            out[target] = df[sources[0]] if sources else ""
            continue

        pref_key = preferred_source_key_by_target.get(target)
        if pref_key:
            preferred = [s for s in sources if _norm_key(s) == pref_key]
            others = [s for s in sources if _norm_key(s) != pref_key]
            sources_ordered = preferred + others
        else:
            sources_ordered = sources

        out[target] = _coalesce_first_nonempty(df, sources_ordered)

    df = out
    df.columns = [_normalize_header(c) for c in df.columns]

    if opts.add_source_column:
        source_name = _normalize_header(opts.source_column_name)
        if source_name in df.columns:
            df[source_name] = path.name
        else:
            df.insert(0, source_name, path.name)

    return df


def _looks_like_ale(path: Path) -> bool:
    try:
        head = path.read_bytes()[:256]
    except Exception:
        return False
    return head.lstrip().startswith(b"Heading")


def _read_table(path: Path, opts: CombineOptions) -> pd.DataFrame:
    """Read either a TSV/tab-delimited file or an ALE."""
    if _looks_like_ale(path) or path.suffix.lower() == ".ale":
        return _read_ale(path, opts)
    return _read_tsv(path, opts)


def combine_tab_files(
    file_paths: Sequence[os.PathLike | str],
    opts: Optional[CombineOptions] = None,
) -> pd.DataFrame:
    opts = opts or CombineOptions()

    paths = [Path(p).expanduser().resolve() for p in file_paths]
    if not paths:
        raise TabCombineError("No input files provided")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise TabCombineError(f"Input file(s) not found: {', '.join(missing)}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        if p.stat().st_size == 0:
            continue
        frames.append(_read_table(p, opts))

    if not frames:
        raise TabCombineError("All input files were empty (0 bytes) or unreadable")

    combined = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    combined.columns = [_normalize_header(c) for c in combined.columns]

    # Drop columns that are completely blank (all values empty string)
    blank_cols = [
        col
        for col in combined.columns
        if combined[col].astype(str).str.strip().eq("").all()
    ]
    if blank_cols:
        combined = combined.drop(columns=blank_cols)

    return combined


def _build_output_from_sources(source_df: pd.DataFrame, output_columns: list[str]) -> pd.DataFrame:
    """
    Reduce/transform the combined source dataframe to the ShotGrid schema.

    Rules:
      - Source Clip Name == Tape (unique identifier)
      - Slate output comes from Scene
      - Take Number output comes from Take
      - Take Name output comes from Name
      - Detected fields copied verbatim
      - Comments combined/deduped if multiple exist
      - Missing values left blank
      - Rows deduped by Tape / Source Clip Name
    """
    df = source_df.copy()
    df.columns = [_normalize_header(c) for c in df.columns]

    key_to_cols: dict[str, list[str]] = {}
    for c in df.columns:
        key_to_cols.setdefault(_norm_key(c), []).append(c)

    def _pick(*aliases: str) -> list[str]:
        cols: list[str] = []
        for a in aliases:
            cols.extend(key_to_cols.get(_norm_key(a), []))
        return cols

    out = pd.DataFrame(index=df.index)

    # Derived per user requirements
    tape_cols = _pick("Tape")
    out[_normalize_header("Tape")] = _coalesce_first_nonempty(df, tape_cols)
    out[_normalize_header("Source Clip Name")] = out[_normalize_header("Tape")]

    scene_cols = _pick("Scene")
    take_cols = _pick("Take")
    name_cols = _pick("Name")

    out[_normalize_header("Slate")] = _coalesce_first_nonempty(df, scene_cols)
    out[_normalize_header("Take Number")] = _coalesce_first_nonempty(df, take_cols)
    out[_normalize_header("Take Name")] = _coalesce_first_nonempty(df, name_cols)

    copy_map: dict[str, tuple[str, ...]] = {
        "ASC_SAT": ("ASC_SAT", "ASC SAT"),
        "ASC_SOP": ("ASC_SOP", "ASC SOP"),
        "Camera Format": ("Camera Format", "Format", "CameraFormat"),
        "Camera Letter": ("Camera Letter", "Camera"),
        "Camera Roll Angle": ("Camera Roll Angle", "CameraRollAngle"),
        "Shutter Angle": ("Shutter Angle", "ShutterAngle"),
        "Camera Tilt Angle": ("Camera Tilt Angle", "CameraTiltAngle"),
        "Camera Model": ("Camera Model", "Camera model", "CameraModel", "Uniquecameramode"),
        "ColorTemp": ("ColorTemp", "Color Temp", "Color Temperature"),
        "ISO": ("ISO", "ISO.1"),
        "Resolution": ("Resolution",),
        "FPS": ("FPS", "Framerate", "Frame Rate"),
        "Shoot Date": ("Shoot Date", "ShootDate"),
        "Detected": ("Detected",),
        "Detected Secondaries": ("Detected Secondaries", "Detected seconda", "Detected Seconda"),
        "Detected Dynamics": ("Detected Dynamics", "Detected dynamic", "Detected Dynamic"),
        "Grade Info": ("Grade Info", "Grade info", "GradeInfo"),
    }

    # Comments: combine + dedupe if multiple comment-like columns exist
    comment_cols = _pick("Comments", "Comment", "Notes", "Note")
    if comment_cols:
        out[_normalize_header("Comments")] = df[comment_cols].astype(str).apply(
            lambda r: _dedupe_join(r.tolist()), axis=1
        )
    else:
        out[_normalize_header("Comments")] = ""

    for target, aliases in copy_map.items():
        t = _normalize_header(target)
        if t in out.columns:
            continue
        cols = _pick(*aliases)
        out[t] = _coalesce_first_nonempty(df, cols)

    # Ensure all requested output columns exist
    for c in output_columns:
        if c not in out.columns:
            out[c] = ""

    out = out.loc[:, output_columns]

    # Deduplicate by unique identifier (Source Clip Name == Tape)
    uid = _normalize_header("Source Clip Name")
    if uid in out.columns:
        def _first_nonempty(series: pd.Series) -> str:
            for v in series.astype(str).tolist():
                if str(v).strip() != "":
                    return str(v)
            return ""

        agg: dict[str, callable] = {}
        for c in out.columns:
            if c == _normalize_header("Comments"):
                agg[c] = lambda s: _dedupe_join(s.astype(str).tolist())
            else:
                agg[c] = _first_nonempty

        out = out.groupby(uid, as_index=False, dropna=False).agg(agg)

    return out


def run_app() -> None:
    import tempfile
    import streamlit as st

    st.set_page_config(page_title="Dailies Importer", layout="wide")

    st.title("Dailies Importer")
    st.caption(
        "Drop one or more dailies bins as ALEs or tab-delimeted files and export one combined CSV."
    )

    uploaded = st.file_uploader(
        "Upload tab-delimited files",
        type=["txt", "tab", "tsv", "ale"],
        accept_multiple_files=True,
    )

    template = st.file_uploader(
        "Upload ShotGrid template CSV (optional)",
        type=["csv"],
        accept_multiple_files=False,
        help="If provided, the output CSV will match this header order (with known template mistakes auto-corrected).",
    )

    # Cache management: if the upload set changes, clear cached combine results
    upload_key = (
        tuple(sorted((f.name, f.size) for f in uploaded)) if uploaded else tuple(),
        (template.name, template.size) if template else (None, None),
    )
    if st.session_state.get("_upload_key") != upload_key:
        st.session_state["_upload_key"] = upload_key
        st.session_state.pop("combined_df_full", None)

    with st.sidebar:
        st.header("Options")

        with st.expander("Show options", expanded=False):
            add_source = st.checkbox("Add SourceFile column", value=True)
            source_col = st.text_input(
                "Source column name",
                value="SourceFile",
                disabled=not add_source,
            )

            encoding = st.selectbox(
                "Encoding",
                options=["utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman"],
                index=0,
            )

            errors = st.selectbox(
                "Encoding errors",
                ["strict", "replace", "ignore"],
                index=0,
            )

        st.session_state.setdefault("preview_rows", 50)
        st.session_state.setdefault("output_name", "combined.csv")

        st.subheader("Output")
        st.caption("These settings apply only after you click Apply.")

        with st.form("output_settings_form", clear_on_submit=False):
            preview_rows_ui = st.slider(
                "Preview rows",
                10,
                500,
                int(st.session_state.get("preview_rows", 50)),
                10,
            )

            output_name_ui = st.text_input(
                "Output filename",
                value=str(st.session_state.get("output_name", "combined.csv")),
            )

            applied = st.form_submit_button("Apply output settings")

        if applied:
            st.session_state["preview_rows"] = int(preview_rows_ui)
            st.session_state["output_name"] = str(output_name_ui).strip() or "combined.csv"

        preview_rows = int(st.session_state.get("preview_rows", 50))
        output_name = str(st.session_state.get("output_name", "combined.csv"))

        st.divider()
        st.write("**Notes**")
        st.write(
            "- Files are read as **tab-delimited** (TSV) or **ALE**.\n"
            "- Values are treated as **text** (safe for IDs/timecode).\n"
            "- Output is reduced to the **ShotGrid schema** (template CSV if provided).\n"
            "- Rows are **deduplicated by Tape** (Source Clip Name == Tape)."
        )

    def _combine_uploaded_files() -> pd.DataFrame:
        if not uploaded:
            raise TabCombineError("No files uploaded")

        opts = CombineOptions(
            encoding=encoding.strip() or "utf-8",
            errors=errors,
            add_source_column=add_source,
            source_column_name=source_col.strip() or "SourceFile",
        )

        with tempfile.TemporaryDirectory(prefix="tabcombiner_") as tmpdir:
            tmp_paths: list[str] = []
            for f in uploaded:
                p = Path(tmpdir) / f.name
                p.write_bytes(f.getbuffer())
                tmp_paths.append(str(p))

            combined_sources = combine_tab_files(tmp_paths, opts=opts)

            # Output schema from template if provided
            if template is not None:
                tpl_path = Path(tmpdir) / template.name
                tpl_path.write_bytes(template.getbuffer())
                output_columns = _load_template_columns(tpl_path, encoding=encoding.strip() or "utf-8")
            else:
                output_columns = [
                    "Source Clip Name",
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
                    "Resolution",
                    "FPS",
                    "Comments",
                    "Slate",
                    "Take Name",
                    "Take Number",
                    "Shoot Date",
                    "Detected",
                    "Detected Secondaries",
                    "Detected Dynamics",
                    "Grade Info",
                ]
                output_columns = [_normalize_header(c) for c in output_columns]

            output_df = _build_output_from_sources(combined_sources, output_columns)
            return output_df

    if not uploaded:
        st.info("Upload one or more .txt/.tab/.tsv files (and optionally a template CSV) to get started.")
        return

    col_a, col_b, col_c = st.columns([1, 1, 2], vertical_alignment="center")
    with col_a:
        do_combine = st.button("Combine", type="primary")
    with col_b:
        st.write(f"**Files:** {len(uploaded)}")
    with col_c:
        names = ", ".join([f.name for f in uploaded][:6])
        if len(uploaded) > 6:
            names += f" … (+{len(uploaded) - 6} more)"
        st.write(names)

    if do_combine or (st.session_state.get("combined_df_full") is None):
        try:
            with st.spinner("Combining…"):
                combined_df_full = _combine_uploaded_files()
                st.session_state["combined_df_full"] = combined_df_full
            st.success(f"Combined {len(uploaded)} file(s)")
        except TabCombineError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return

    combined_df_full_cached = st.session_state.get("combined_df_full")
    if combined_df_full_cached is None:
        return

    download_df = combined_df_full_cached

    csv_bytes = download_df.to_csv(sep=",", index=False).encode(encoding.strip() or "utf-8")
    st.download_button(
        label="Download combined CSV",
        data=csv_bytes,
        file_name=output_name.strip() or "combined.csv",
        mime="text/csv",
    )

    st.subheader("Preview")
    st.dataframe(download_df.head(preview_rows), width="stretch")

    st.subheader("Stats")
    st.write(f"Rows: **{len(download_df):,}** | Columns: **{len(download_df.columns):,}**")


if __name__ == "__main__":
    run_app()