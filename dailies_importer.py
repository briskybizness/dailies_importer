"""Dailies Importer

Streamlit app to combine Avid TAB-delimited text exports and ALE files into a single CSV.

Key behaviors (as requested):
- Always stamps each record with the originating uploaded filename in a first column: SourceFile.
- SourceFile is ALWAYS the first column in the output CSV (even when using a template).
- Camera Format is sourced ONLY from Input Color Space (no fallback to Format).
- Shoot Day is normalized to PREFIX_<2-digit day> by default (configurable width).
- Shoot Date is normalized to mm/dd/yyyy (handles yyyymmdd like 20251003).
- Adds a Take column derived from Clip Name using the same logic as Conformer.
- Take naming UI matches Conformer app screenshot: dropdowns + preview + test clip name.
- Adds a UI field to strip unwanted substrings from the extracted Slate/Camera tokens
  before generating the VFX Take (e.g. remove PU, SER, etc.).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

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
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    block = df[cols].astype(str)
    block = block.replace(r"^\s*$", pd.NA, regex=True)
    return block.bfill(axis=1).iloc[:, 0].fillna("")


# -----------------------------
# Take parsing (borrowed from Conformer.py) + Conformer-UI-style builder
# -----------------------------

# Keep a sensible default parser pattern (named groups) for internal use
DEFAULT_TAKE_PARSER_PATTERN = r"(?P<slate>V?[0-9A-Z]+)-(?P<take>\d+)(?P<cam>[A-Z]+)?(?P<star>\*)?"


def _compile_take_parser_regex(pattern: str) -> re.Pattern:
    """Compile the user-provided take PARSER regex."""
    pat = (pattern or "").strip() or DEFAULT_TAKE_PARSER_PATTERN
    return re.compile(pat, re.IGNORECASE)


# Default parser regex
EDITORIAL_TAKE_RE = _compile_take_parser_regex(DEFAULT_TAKE_PARSER_PATTERN)


# -----------------------------
# Conformer-app-style take naming UI helpers
# -----------------------------

TAKE_DIGITS_OPTIONS: dict[str, str] = {
    "1+ digits": r"\d+",
    "1 digit": r"\d{1}",
    "2 digits": r"\d{2}",
    "3 digits": r"\d{3}",
}

DELIMITER_OPTIONS: dict[str, str] = {
    "-": "-",
    "_": "_",
    "None": "",
}

CAMERA_TOKEN_OPTIONS: dict[str, str] = {
    "None": "none",
    "Optional A-Z": "optional",
    "Required A-Z": "required",
}

CAMERA_POSITION_OPTIONS = ["After take", "Before take"]


def _build_take_regex_from_ui(
    *,
    take_digits_label: str,
    slate_take_delim_label: str,
    camera_token_label: str,
    camera_position_label: str,
    camera_take_delim_label: str,
) -> tuple[re.Pattern, str]:
    """Build a Conformer-style take regex and an example token for preview.

    The generated regex ALWAYS includes named groups (?P<slate>) and (?P<take>),
    and includes (?P<cam>) when camera is enabled.

    Returns: (compiled_regex, example_take_token)
    """
    take_pat = TAKE_DIGITS_OPTIONS.get(take_digits_label, r"\d+")
    st_delim_raw = DELIMITER_OPTIONS.get(slate_take_delim_label, "-")
    st_delim = re.escape(st_delim_raw)

    cam_mode = CAMERA_TOKEN_OPTIONS.get(camera_token_label, "optional")
    cam_delim_raw = DELIMITER_OPTIONS.get(camera_take_delim_label, "")
    cam_delim = re.escape(cam_delim_raw)

    slate_pat = r"(?P<slate>V?[0-9A-Z]+)"
    take_group = rf"(?P<take>{take_pat})"

    cam_group = ""
    cam_example = ""

    # IMPORTANT: the overall regex is compiled with IGNORECASE so that slates match
    # regardless of case, but the camera token must be STRICTLY uppercase to avoid
    # picking up pickup suffixes like 'PUa' where 'a' is not a camera letter.
    # `(?-i:...)` disables IGNORECASE for the camera subpattern.
    cam_letter_pat = r"(?-i:[A-Z])"

    if cam_mode == "required":
        cam_group = rf"(?P<cam>{cam_letter_pat})"
        cam_example = "B"
    elif cam_mode == "optional":
        cam_group = rf"(?P<cam>{cam_letter_pat})?"
        cam_example = "B"

    # Optional star marker (ignored by conversion)
    star_group = r"(?P<star>\*)?"

    if cam_mode == "none":
        # {slate}{delim}{take}
        core = rf"{slate_pat}{st_delim}{take_group}{star_group}"
        example = "99E" + st_delim_raw + "12"
        return re.compile(core, re.IGNORECASE), example

    if camera_position_label == "Before take":
        # {slate}{delim}{cam}{cam_delim}{take}
        core = rf"{slate_pat}{st_delim}{cam_group}{cam_delim}{take_group}{star_group}"
        example = "99E" + st_delim_raw + cam_example + cam_delim_raw + "12"
        return re.compile(core, re.IGNORECASE), example

    # After take: {slate}{delim}{take}{cam_delim}{cam}
    core = rf"{slate_pat}{st_delim}{take_group}{cam_delim}{cam_group}{star_group}"
    example = "99E" + st_delim_raw + "12" + cam_delim_raw + cam_example
    return re.compile(core, re.IGNORECASE), example


# -----------------------------
# Strip-token helpers (UI-driven)
# -----------------------------

def _parse_strip_tokens(raw: str) -> list[str]:
    """Parse a user-entered token list (comma/space/line separated)."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    # Allow commas, semicolons, whitespace, and newlines as separators.
    parts = re.split(r"[;,\n\r\t ]+", s)
    tokens = [p.strip() for p in parts if p and p.strip()]

    # Dedupe case-insensitively while preserving original casing (first occurrence).
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _strip_tokens(text: str, tokens: list[str], *, preserve_leading_v: bool = True) -> str:
    """Remove literal token substrings from `text` (case-insensitive)."""
    if not text:
        return ""
    if not tokens:
        return str(text)

    s = str(text)

    prefix = ""
    rest = s
    if preserve_leading_v and rest[:1].upper() == "V":
        prefix = rest[:1]
        rest = rest[1:]

    for tok in tokens:
        t = (tok or "").strip()
        if not t:
            continue
        rest = re.sub(re.escape(t), "", rest, flags=re.IGNORECASE)

    # Clean up any doubled delimiters/underscores created by stripping.
    rest = re.sub(r"__+", "_", rest)
    rest = re.sub(r"--+", "-", rest)
    rest = rest.strip(" _-")

    return prefix + rest


def editorial_take_to_vfx_take(
    editorial_clip_name: str,
    source_file: str = "",
    match_re: Optional[re.Pattern] = None,
    parser_re: Optional[re.Pattern] = None,
    strip_tokens: Optional[list[str]] = None,
) -> Optional[str]:
    """Convert editorial take naming to VFX take naming.

    Editorial: {slate}-{take}{cam}
    VFX:       {slate}_{cam}-{take:02d}

    Returns None if no match.
    """
    s = (editorial_clip_name or "").strip()
    tokens = strip_tokens or []

    # Apply token stripping to the full clip-name string BEFORE parsing.
    # This prevents patterns like 'PU' from being interpreted as the camera token.
    parse_s = s
    for tok in tokens:
        t = (tok or "").strip()
        if not t:
            continue
        parse_s = re.sub(re.escape(t), "", parse_s, flags=re.IGNORECASE)

    mrx = match_re or re.compile(r"V?[0-9A-Z]+-\d+[A-Z]*\*?", re.IGNORECASE)
    prx = parser_re or EDITORIAL_TAKE_RE

    if not mrx.search(parse_s):
        return None

    m = prx.search(parse_s)
    if not m:
        return None

    gd = m.groupdict()
    slate = (gd.get("slate") or "").strip().replace("*", "")
    take_raw = (gd.get("take") or "").strip().replace("*", "")
    cam = (gd.get("cam") or "").strip().replace("*", "")

    # Normalize take to digits only
    take_raw = re.sub(r"\D", "", take_raw)

    # Apply user stripping first (so PUa -> a if user wants PU removed)
    slate = _strip_tokens(slate, tokens, preserve_leading_v=True)
    cam = _strip_tokens(cam, tokens, preserve_leading_v=False)

    # Legacy PU cleanup (kept for backward compatibility)
    up_slate = slate.upper()
    if up_slate.startswith("VPU") and len(slate) > 3:
        slate = "V" + slate[3:]
    else:
        if up_slate.startswith("PU") and len(slate) > 2:
            slate = slate[2:]
        if slate.upper().endswith("PU") and len(slate) > 2:
            slate = slate[:-2]

    if cam.upper().startswith("PU"):
        cam = cam[2:].strip()

    if not cam:
        sf = (source_file or "").strip()
        if sf:
            first = sf[0].upper()
            if "A" <= first <= "Z":
                cam = first

    try:
        take_num = int(take_raw)
    except ValueError:
        return None

    take_padded = f"{take_num:02d}"

    if cam:
        return f"{slate}_{cam}-{take_padded}"
    return f"{slate}_-{take_padded}"


def _fmt_shoot_day(value: object, *, sep: str = "_", width: int = 2) -> str:
    """Normalize shoot day strings."""
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
    """Force Shoot Date to mm/dd/yyyy."""
    s = str(value).strip()
    if not s:
        return ""

    if re.fullmatch(r"\d{8}", s):
        try:
            dt = datetime.strptime(s, "%Y%m%d")
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return s

    if re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", s):
        try:
            dt = datetime.strptime(s.replace("/", "-"), "%Y-%m-%d")
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return s

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

    src = _normalize_header(opts.source_column_name or "SourceFile")
    if src in df.columns:
        df[src] = uploaded_file.name
    else:
        df.insert(0, src, uploaded_file.name)

    return df


def _read_ale(uploaded_file, *, opts: ReadOptions) -> pd.DataFrame:
    """Very small ALE reader."""
    text = _read_text_bytes(uploaded_file, opts=opts)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

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
    for ln in lines[data_idx + 1:]:
        if not ln.strip():
            continue
        rows.append(ln.split("\t"))

    df = pd.DataFrame(rows, columns=headers).fillna("").astype(str)
    df = df.replace({"nan": "", "NaN": "", "NAN": ""})

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
    return _read_tsv(uploaded_file, opts=opts)


# -----------------------------
# Output building
# -----------------------------

COPY_MAP: dict[str, tuple[str, ...]] = {
    "SourceFile": ("SourceFile",),
    "Tape": ("Tape", "Reel", "Reel Name", "Tape Name", "tape", "reel"),
    "ASC_SAT": ("ASC_SAT", "ASC SAT"),
    "ASC_SOP": ("ASC_SOP", "ASC SOP"),
    # Camera Format ONLY from Input Color Space (no fallbacks)
    "Camera Format": ("Input Color Space", "Input color spac", "colorspaceinput", "ColorSpaceInput"),
    "Camera Letter": ("Camera Letter", "Camera", "CameraLetter", "Cam", "Cam Letter", "CamLetter"),
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
    "ColorTemp": ("ColorTemp", "Color Temp", "Color Temperature", "Kelvin", "White Balance", "WhiteBalance", "WB"),
    "ISO": ("ISO", "ISO.1", "EI", "Exposure Index", "ExposureIndex", "ASA", "ISOSpeed", "ISO Speed"),
    "Camera Resolution": ("Resolution", "Camera Resolution", "CameraResolution"),
    "FPS": ("FPS", "Framerate", "Frame Rate"),
    "Comments": ("Comments", "Comment"),
    "Slate": ("Slate", "Scene"),
    "Clip Name": ("Take Name", "TakeName", "Name"),
    "Take Number": ("Take Number", "TakeNumber", "Take"),
    "Shoot Date": ("Shoot Date", "ShootDate"),
    "Shoot Day": ("Shoot Day", "Shoot day", "ShootDay"),
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
    "Take",
    "Take Number",
    "Shoot Date",
    "Shoot Day",
    "Detected Secondaries",
    "Detected Dynamics",
    "Grade Info",
]


def _build_output(
    combined_sources: pd.DataFrame,
    *,
    output_columns: list[str],
    shoot_day_separator: str = "_",
    shoot_day_number_width: int = 2,
    take_match_regex: Optional[re.Pattern] = None,
    take_parser_regex: Optional[re.Pattern] = None,
    strip_tokens: Optional[list[str]] = None,
) -> pd.DataFrame:
    src_df = combined_sources.copy()
    src_df.columns = [_normalize_header(c) for c in src_df.columns]
    src_df = src_df.fillna("")
    src_df = src_df.replace({"nan": "", "NaN": "", "NAN": ""})

    key_to_cols: dict[str, list[str]] = {}
    for c in src_df.columns:
        key_to_cols.setdefault(_norm_key(c), []).append(c)

    def pick_cols(*aliases: str) -> list[str]:
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

    out_cols = [_normalize_header(c) for c in output_columns]
    src_col = _normalize_header("SourceFile")
    out_cols = [c for c in out_cols if c != src_col]
    out_cols.insert(0, src_col)

    comments_key = _normalize_header("Comments")
    if comments_key in out_cols:
        comment_cols = pick_cols("Comments", "Comment", "Notes", "Note")
        if comment_cols:
            out[comments_key] = src_df[comment_cols].astype(str).apply(lambda r: _dedupe_join(r.tolist()), axis=1)
            out[comments_key] = out[comments_key].replace({"nan": "", "NaN": "", "NAN": ""})
        else:
            out[comments_key] = ""

    for target in out_cols:
        if target == comments_key:
            continue
        aliases = COPY_MAP.get(target, (target,))
        cols = pick_cols(*aliases)
        if not cols:
            out[target] = ""
        else:
            out[target] = _coalesce_first_nonempty(src_df, cols).replace({"nan": "", "NaN": "", "NAN": ""})

    slate_col = _normalize_header("Slate")
    clip_name_col = _normalize_header("Clip Name")
    take_num_col = _normalize_header("Take Number")

    def _blank(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().eq("")

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

    take_col = _normalize_header("Take")
    if take_col in out.columns:
        cam_letter_col = _normalize_header("Camera Letter")

        clip_series = out.get(clip_name_col, pd.Series([""] * len(out), index=out.index)).astype(str)
        if cam_letter_col in out.columns:
            cam_series = out[cam_letter_col].astype(str)
        else:
            cam_series = pd.Series([""] * len(out), index=out.index)

        toks = strip_tokens or []
        out[take_col] = [
            editorial_take_to_vfx_take(
                cn,
                (cl or ""),
                match_re=take_match_regex,
                parser_re=take_parser_regex,
                strip_tokens=toks,
            ) or ""
            for cn, cl in zip(clip_series.tolist(), cam_series.tolist())
        ]

    out = out.replace({"nan": "", "NaN": "", "NAN": ""})

    sd_col = _normalize_header("Shoot Day")
    if sd_col in out.columns:
        out[sd_col] = out[sd_col].map(lambda v: _fmt_shoot_day(v, sep=shoot_day_separator, width=shoot_day_number_width))

    sdate_col = _normalize_header("Shoot Date")
    if sdate_col in out.columns:
        out[sdate_col] = out[sdate_col].map(_fmt_shoot_date)

    out = out.loc[:, out_cols]
    return out


def _load_template_columns(uploaded_template) -> list[str]:
    raw = uploaded_template.getvalue()
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
        with st.expander("File decoding", expanded=False):
            encoding = st.selectbox(
                "Encoding",
                options=["utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman"],
                index=0,
            )
            errors = st.selectbox("Encoding errors", options=["strict", "replace", "ignore"], index=0)

        # --- Take naming (match Conformer app UI screenshot)
        st.subheader("Take naming")

        st.caption("Optional: remove unwanted substrings from extracted Slate/Camera tokens (comma/space separated).")
        strip_tokens_raw = st.text_input(
            "Remove from slate/camera",
            value="",
            placeholder="e.g. PU, SER",
        )
        strip_tokens = _parse_strip_tokens(strip_tokens_raw)

        take_digits_label = st.selectbox(
            "Take digits",
            options=list(TAKE_DIGITS_OPTIONS.keys()),
            index=0,  # 1+ digits
        )

        slate_take_delim_label = st.selectbox(
            "Take/slate delimiter",
            options=list(DELIMITER_OPTIONS.keys()),
            index=0,  # '-'
        )

        camera_token_label = st.selectbox(
            "Camera token",
            options=list(CAMERA_TOKEN_OPTIONS.keys()),
            index=1,  # Optional A-Z
        )

        camera_position_label = st.selectbox(
            "Camera position",
            options=CAMERA_POSITION_OPTIONS,
            index=0,  # After take
        )

        camera_take_delim_label = st.selectbox(
            "Camera/take delimiter",
            options=list(DELIMITER_OPTIONS.keys()),
            index=2,  # None
        )

        try:
            take_regex, example_token = _build_take_regex_from_ui(
                take_digits_label=take_digits_label,
                slate_take_delim_label=slate_take_delim_label,
                camera_token_label=camera_token_label,
                camera_position_label=camera_position_label,
                camera_take_delim_label=camera_take_delim_label,
            )
        except re.error as e:
            st.error(f"Invalid take naming config: {e}. Falling back to defaults.")
            take_regex, example_token = _build_take_regex_from_ui(
                take_digits_label="1+ digits",
                slate_take_delim_label="-",
                camera_token_label="Optional A-Z",
                camera_position_label="After take",
                camera_take_delim_label="None",
            )

        st.markdown("### Preview")
        st.caption("Example take token (format preview — optional '*' is ignored)")
        st.code(example_token)

        st.caption("Test a clip name")
        test_clip_name = st.text_input("", value="", placeholder="Paste Clip Name")
        if test_clip_name.strip():
            parsed = editorial_take_to_vfx_take(
                test_clip_name,
                source_file="",
                match_re=take_regex,
                parser_re=take_regex,
                strip_tokens=strip_tokens,
            )
            if parsed:
                st.success(parsed)
            else:
                st.warning("No take found in clip name.")

        st.session_state["_take_match_regex"] = take_regex
        st.session_state["_take_parser_regex"] = take_regex
        st.session_state["_strip_tokens"] = strip_tokens

        # --- Shoot day UI
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

    # Quick preview of take parsing behavior (first 25 rows)
    clip_col_guess = None
    for c in combined.columns:
        if _norm_key(c) in (_norm_key("Clip Name"), _norm_key("Take Name"), _norm_key("Name")):
            clip_col_guess = c
            break

    take_match_regex = st.session_state.get("_take_match_regex")
    take_parser_regex = st.session_state.get("_take_parser_regex")
    strip_tokens = st.session_state.get("_strip_tokens") or []

    if clip_col_guess and take_match_regex and take_parser_regex:
        with st.expander("Preview: Take parsing on first 25 rows"):
            sample = combined[[clip_col_guess]].head(25).copy()
            sample["Matches?"] = sample[clip_col_guess].astype(str).apply(lambda s: bool(take_match_regex.search((s or "").strip())))
            sample["Parsed Take"] = sample[clip_col_guess].astype(str).apply(
                lambda s: editorial_take_to_vfx_take(
                    (s or ""),
                    "",
                    match_re=take_match_regex,
                    parser_re=take_parser_regex,
                    strip_tokens=strip_tokens,
                ) or ""
            )
            sample.rename(columns={clip_col_guess: "Clip Name"}, inplace=True)
            st.dataframe(sample, use_container_width=True)

    if template is not None:
        try:
            output_columns = _load_template_columns(template)
        except Exception as e:
            st.error(f"Template could not be read: {e}")
            output_columns = DEFAULT_OUTPUT_COLUMNS.copy()
    else:
        output_columns = DEFAULT_OUTPUT_COLUMNS.copy()

    must_have = ["SourceFile", "Take"]
    norm_cols = [_normalize_header(c) for c in output_columns]

    for c in must_have:
        cc = _normalize_header(c)
        if cc in norm_cols:
            continue

        if cc == _normalize_header("Take"):
            clip_key = _normalize_header("Clip Name")
            if clip_key in norm_cols:
                idx = norm_cols.index(clip_key) + 1
                norm_cols.insert(idx, cc)
            else:
                norm_cols.append(cc)
        else:
            norm_cols.insert(0, cc)

    output_columns = norm_cols

    output_df = _build_output(
        combined,
        output_columns=output_columns,
        shoot_day_separator=shoot_day_separator,
        shoot_day_number_width=int(shoot_day_number_width),
        take_match_regex=take_match_regex,
        take_parser_regex=take_parser_regex,
        strip_tokens=strip_tokens,
    )

    tape_col = _normalize_header("Tape")
    if tape_col in output_df.columns:
        tape_series = output_df[tape_col].astype(str).str.strip()
        nonempty = tape_series.ne("")
        dup_mask = nonempty & tape_series.duplicated(keep=False)
        dup_rows = int(dup_mask.sum())

        if dup_rows > 0:
            dup_groups = int(tape_series[dup_mask].drop_duplicates().shape[0])
            st.warning(
                f"⚠️ Duplicate Tape names detected: **{dup_rows}** rows across **{dup_groups}** duplicate Tape values. "
                "No rows were removed."
            )

            with st.expander("Show duplicate Tape examples"):
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
                        _normalize_header("Take"),
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

    csv_bytes = output_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download combined CSV",
        data=csv_bytes,
        file_name="dailies_combined.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    run_app()