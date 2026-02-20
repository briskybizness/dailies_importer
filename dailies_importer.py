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


# Aggressive key normalization for header alias matching
def _norm_key(name: str) -> str:
    """Aggressive key normalization for header alias matching (case/space/punct-insensitive)."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    # Keep only alphanumerics so 'colorSpaceOutput', 'Color Space Output', etc. match.
    return "".join(ch for ch in s if ch.isalnum())


def _coalesce_first_nonempty(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return a Series with the first non-empty value across cols (left-to-right)."""
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    block = df[cols].astype(str)
    block = block.replace(r"^\s*$", pd.NA, regex=True)
    return block.bfill(axis=1).iloc[:, 0].fillna("")


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


# -------- ALE/Avid reader and table dispatcher --------

def _read_ale(path: Path, opts: CombineOptions) -> pd.DataFrame:
    """Read an Avid ALE file into a DataFrame.

    ALE structure is typically:
      Heading\n...\nColumn\n<tab-separated header line>\nData\n<tab-separated rows>

    This reader:
      - Parses the Column/Data sections
      - Applies a small alias map so ALE header variants align with our TSV headers
      - Optionally inserts SourceFile column
    """
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
            # Skip completely empty lines
            if not row or all(str(x).strip() == "" for x in row):
                continue
            # Pad/truncate to header count
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[: len(headers)]
            data_rows.append([str(x) for x in row])

    df = pd.DataFrame(data_rows, columns=headers)

    # Align common ALE column-name variants to the TSV headers we already use.
    # Keys are aggressively-normalized so different spacing/case/camelCase still match.
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

    # Build target groups in original column order so we can coalesce duplicates deterministically.
    # IMPORTANT: For some fields we prefer a specific ALE source header when multiple map to the same target.
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

    # Create a new dataframe with unique column names.
    out = pd.DataFrame(index=df.index)
    for target in target_order:
        sources = target_to_sources.get(target, [])
        if len(sources) <= 1:
            out[target] = df[sources[0]] if sources else ""
            continue

        # Reorder sources so preferred header wins when present.
        pref_key = preferred_source_key_by_target.get(target)
        if pref_key:
            preferred = [s for s in sources if _norm_key(s) == pref_key]
            others = [s for s in sources if _norm_key(s) != pref_key]
            sources_ordered = preferred + others
        else:
            sources_ordered = sources

        out[target] = _coalesce_first_nonempty(df, sources_ordered)

    df = out

    # Normalize columns after coalescing
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
    # ALE files typically begin with 'Heading' followed by CRLF.
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


def select_columns(df: pd.DataFrame, ordered_columns: List[str], strict: bool = False) -> pd.DataFrame:
    if not ordered_columns:
        return df

    df = df.copy()
    df.columns = [_normalize_header(c) for c in df.columns]

    norm_to_actual = {_normalize_header(c): c for c in df.columns}
    wanted_norm = [_normalize_header(c) for c in ordered_columns]

    missing = [ordered_columns[i] for i, n in enumerate(wanted_norm) if n not in norm_to_actual]
    if missing and strict:
        raise TabCombineError(f"Missing expected column(s): {', '.join(missing)}")

    keep_actual = [norm_to_actual[n] for n in wanted_norm if n in norm_to_actual]
    return df.loc[:, keep_actual]


BASE_DESIRED_COLUMNS = [
    "Tape",
    "Name",
    "Scene",
    "Take",
    "Camera",
    "Start",
    "End",
    "Duration",
    "Frames",
    "Camroll",
    "FPS",
    "Shoot Date",
    "Shoot day",
    "Image Size",
    "Resolution",
    "ASC_SOP",
    "Color space note",
    "Input color spac",
    "colorspaceoutput",
    "Camera model",
    "Grade info",
    "Detected seconda",
    "Detected dynamic",
]

BASE_DEFAULT_OUTPUT_HEADERS = [
    "Dailies Name",
    "Clip Name",
    "Scene",
    "Take",
    "Camera",
    "Start Timecode",
    "End Timecode",
    "Duration",
    "Frames",
    "Camroll",
    "FPS",
    "Shoot Date",
    "Shoot Day",
    "Dailies Resolution",
    "Camera Resolution",
    "ASC_SOP",
    "Camera Color Space",
    "Input Color Space",
    "Output Color Space",
    "Camera Model",
    "Grade Info",
    "Secondary Color",
    "Dynamic Color",
]


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

    # Cache management: if the upload set changes, clear cached combine results
    upload_key = tuple(sorted((f.name, f.size) for f in uploaded)) if uploaded else tuple()
    if st.session_state.get("_upload_key") != upload_key:
        st.session_state["_upload_key"] = upload_key
        st.session_state.pop("combined_df_full", None)
        # removed: st.session_state.pop("available_extra_columns", None)

    with st.sidebar:
        st.header("Options")

        with st.expander("Show options", expanded=False):
            add_source = st.checkbox("Add SourceFile column", value=True)
            source_col = st.text_input(
                "Source column name",
                value="SourceFile",
                disabled=not add_source,
            )

            # Common encodings for dailies/editorial exports
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

        # Defaults stored in session state so changes only take effect when applied
        st.session_state.setdefault("selected_extra_columns", [])
        st.session_state.setdefault("preview_rows", 50)
        st.session_state.setdefault("output_name", "combined.csv")

        st.subheader("Output")
        st.caption("These settings apply only after you click Apply.")

        # If we already combined, offer additional columns from the uploaded files
        combined_df_full_cached = st.session_state.get("combined_df_full")
        if combined_df_full_cached is not None:
            source_col_norm_for_options = _normalize_header(
                (st.session_state.get("_source_col_last") or "SourceFile")
            )
            base_set = set(BASE_DESIRED_COLUMNS)
            extras_all = [
                c
                for c in combined_df_full_cached.columns
                if c not in base_set and c != source_col_norm_for_options
            ]
            extras_all = sorted(extras_all)
        else:
            extras_all = []

        with st.form("output_settings_form", clear_on_submit=False):
            selected_extras_ui = st.multiselect(
                "Additional columns",
                options=extras_all,
                default=st.session_state.get("selected_extra_columns", []),
                help="Optional: append extra columns from your tab files to the output.",
                disabled=len(extras_all) == 0,
            )

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
            st.session_state["selected_extra_columns"] = selected_extras_ui
            st.session_state["preview_rows"] = int(preview_rows_ui)
            st.session_state["output_name"] = str(output_name_ui).strip() or "combined.csv"

        # Use the currently applied values below
        preview_rows = int(st.session_state.get("preview_rows", 50))
        output_name = str(st.session_state.get("output_name", "combined.csv"))

        st.divider()
        st.write("**Notes**")
        st.write(
            "- Files are read as **tab-delimited** (TSV) or **ALE**.\n"
            "- Values are treated as **text** (safe for IDs/timecode).\n"
            "- Blank columns are dropped.\n"
            "- Output includes only the core columns listed in the code."
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
        # Persist last-used source column name for sidebar option building
        st.session_state["_source_col_last"] = source_col.strip() or "SourceFile"

        with tempfile.TemporaryDirectory(prefix="tabcombiner_") as tmpdir:
            tmp_paths: list[str] = []
            for f in uploaded:
                p = Path(tmpdir) / f.name
                p.write_bytes(f.getbuffer())
                tmp_paths.append(str(p))

            return combine_tab_files(tmp_paths, opts=opts)

    if not uploaded:
        st.info("Upload two or more .txt/.tab/.tsv files to get started.")
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

    # Auto-combine once when uploads exist so extras populate
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

    source_col_norm = _normalize_header(source_col.strip() or "SourceFile")
    desired_columns = BASE_DESIRED_COLUMNS.copy()

    # Append any extra columns the user selected (if present in the combined data)
    selected_extras = st.session_state.get("selected_extra_columns", [])
    for c in selected_extras:
        if c and c not in desired_columns:
            desired_columns.append(c)

    if add_source:
        desired_columns = [source_col_norm] + desired_columns

    combined_df = select_columns(combined_df_full_cached, desired_columns, strict=False)

    st.subheader("Download headers")
    st.caption("Edit the output header names below. This only affects the downloaded CSV — the preview keeps original headers.")

    # Build default header map for the *base* columns
    base_header_map = {BASE_DESIRED_COLUMNS[i]: BASE_DEFAULT_OUTPUT_HEADERS[i] for i in range(len(BASE_DESIRED_COLUMNS))}

    # Initialize a persistent header map in session state (per original column name)
    if "output_header_map" not in st.session_state:
        st.session_state["output_header_map"] = {}

    # Ensure SourceFile has a default output header
    if add_source:
        st.session_state["output_header_map"].setdefault(source_col_norm, "Source File")

    # Seed defaults for base columns
    for k, v in base_header_map.items():
        st.session_state["output_header_map"].setdefault(k, v)

    # Seed defaults for any extra columns (default to original column name)
    for c in desired_columns:
        st.session_state["output_header_map"].setdefault(c, c)

    # Single-row editor: original headers are the (non-editable) column names
    editor_df = pd.DataFrame(
        [[st.session_state["output_header_map"].get(c, c) for c in desired_columns]],
        index=["Output header"],
        columns=desired_columns,
    )

    edited = st.data_editor(
        editor_df,
        width="stretch",
    )

    # Persist output header edits into the map
    for col in desired_columns:
        st.session_state["output_header_map"][col] = str(edited.loc["Output header", col])

    # Build rename map for download
    rename_map = {
        col: _normalize_header(st.session_state["output_header_map"].get(col, col))
        for col in desired_columns
    }

    download_df = combined_df.rename(columns=rename_map)
    csv_bytes = download_df.to_csv(sep=",", index=False).encode(encoding.strip() or "utf-8")
    st.download_button(
        label="Download combined CSV",
        data=csv_bytes,
        file_name=output_name.strip() or "combined.csv",
        mime="text/csv",
    )

    st.subheader("Preview")
    st.dataframe(combined_df.head(preview_rows), width="stretch")

    st.subheader("Stats")
    st.write(f"Rows: **{len(combined_df):,}** | Columns: **{len(combined_df.columns):,}**")


if __name__ == "__main__":
    run_app()