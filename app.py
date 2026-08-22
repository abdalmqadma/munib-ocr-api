from pathlib import Path
import tempfile
import re

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from docling.document_converter import DocumentConverter


app = FastAPI(title="Munib Imsakia API")
converter = DocumentConverter()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "munib-imsakia"
    }


# ============================================================
# Helpers
# ============================================================

def clean_cell(value):
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .replace("|", "")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "")
    )


def get_biggest_table(document_dict):
    tables = document_dict.get("tables", [])

    if not tables:
        return None

    return max(
        tables,
        key=lambda table: (
            table.get("data", {}).get("num_rows", 0)
            *
            table.get("data", {}).get("num_cols", 0)
        )
    )


# ============================================================
# Time parsing
# ============================================================

def normalize_time(
    value,
    previous_hour,
    min_hour,
    max_hour
):
    value = clean_cell(value)

    if not value:
        return None, previous_hour

    full = re.search(
        r"(\d{1,2}):(\d{2})",
        value
    )

    if full:
        hour = int(full.group(1))
        minute = int(full.group(2))

        if 0 <= minute <= 59:

            if min_hour <= hour <= max_hour:
                return (
                    f"{hour:02d}:{minute:02d}",
                    hour
                )

            if previous_hour is not None:
                return (
                    f"{previous_hour:02d}:{minute:02d}",
                    previous_hour
                )

    short = re.search(
        r":(\d{2})",
        value
    )

    if short and previous_hour is not None:
        minute = int(short.group(1))

        if 0 <= minute <= 59:
            return (
                f"{previous_hour:02d}:{minute:02d}",
                previous_hour
            )

    return None, previous_hour


def convert_to_24h(value, prayer):
    if value is None:
        return None

    hour, minute = map(
        int,
        value.split(":")
    )

    if prayer == "asr" and 1 <= hour <= 6:
        hour += 12

    elif prayer == "maghrib" and 5 <= hour <= 8:
        hour += 12

    elif prayer == "isha" and 7 <= hour <= 11:
        hour += 12

    return f"{hour:02d}:{minute:02d}"


# ============================================================
# Extract actual Docling prayer rows
# ============================================================

def extract_docling_rows(document_dict):
    table = get_biggest_table(
        document_dict
    )

    if table is None:
        return []

    grid = (
        table
        .get("data", {})
        .get("grid", [])
    )

    rows = []

    for grid_index, row in enumerate(grid):

        if len(row) != 7:
            continue

        cells = [
            clean_cell(
                cell.get("text", "")
            )
            for cell in row
        ]

        time_count = sum(
            1
            for value in cells
            if re.search(
                r"\d{1,2}:\d{2}|:\d{2}",
                value
            )
        )

        # تجاهل الهيدر
        if time_count < 3:
            continue

        rows.append({
            "grid_index": grid_index,
            "cells": cells
        })

    rows.sort(
        key=lambda item: item["grid_index"]
    )

    return rows


# ============================================================
# Normalize Docling rows
# ============================================================

def normalize_rows(rows):

    hours = {
        "isha": None,
        "maghrib": None,
        "asr": None,
        "dhuhr": None,
        "sunrise": None,
        "fajr": None,
        "fajr_first": None,
    }

    result = []

    for item in rows:

        row = item["cells"]

        isha, hours["isha"] = normalize_time(
            row[0],
            hours["isha"],
            7,
            11
        )

        maghrib, hours["maghrib"] = normalize_time(
            row[1],
            hours["maghrib"],
            5,
            8
        )

        asr, hours["asr"] = normalize_time(
            row[2],
            hours["asr"],
            3,
            6
        )

        dhuhr, hours["dhuhr"] = normalize_time(
            row[3],
            hours["dhuhr"],
            11,
            13
        )

        sunrise, hours["sunrise"] = normalize_time(
            row[4],
            hours["sunrise"],
            5,
            7
        )

        fajr, hours["fajr"] = normalize_time(
            row[5],
            hours["fajr"],
            3,
            6
        )

        fajr_first, hours["fajr_first"] = normalize_time(
            row[6],
            hours["fajr_first"],
            3,
            6
        )

        result.append({
            "grid_index":
                item["grid_index"],

            "fajr":
                fajr,

            "sunrise":
                sunrise,

            "dhuhr":
                dhuhr,

            "asr":
                convert_to_24h(
                    asr,
                    "asr"
                ),

            "maghrib":
                convert_to_24h(
                    maghrib,
                    "maghrib"
                ),

            "isha":
                convert_to_24h(
                    isha,
                    "isha"
                ),

            "fajr_first":
                fajr_first,

            "reconstructed":
                False,

            "review_required":
                False,

            "reconstruction_source":
                None
        })

    return result


# ============================================================
# Time utilities
# ============================================================

def time_to_minutes(value):
    if value is None:
        return None

    h, m = map(
        int,
        value.split(":")
    )

    return h * 60 + m


def minutes_to_time(value):
    value = int(round(value))

    return (
        f"{value // 60:02d}:"
        f"{value % 60:02d}"
    )


# ============================================================
# Repair missing CELLS first
# ============================================================

def repair_missing_cells(days):

    fields = [
        "fajr",
        "sunrise",
        "dhuhr",
        "asr",
        "maghrib",
        "isha",
        "fajr_first"
    ]

    for field in fields:

        for i in range(len(days)):

            if days[i].get(field) is not None:
                continue

            previous_index = None
            next_index = None

            for p in range(
                i - 1,
                -1,
                -1
            ):
                if days[p].get(field) is not None:
                    previous_index = p
                    break

            for n in range(
                i + 1,
                len(days)
            ):
                if days[n].get(field) is not None:
                    next_index = n
                    break

            if (
                previous_index is None
                or next_index is None
            ):
                continue

            previous = time_to_minutes(
                days[previous_index][field]
            )

            following = time_to_minutes(
                days[next_index][field]
            )

            gap = (
                next_index
                -
                previous_index
            )

            if gap <= 0:
                continue

            if abs(
                following - previous
            ) > 20:
                continue

            step = (
                following - previous
            ) / gap

            predicted = (
                previous
                +
                step
                *
                (i - previous_index)
            )

            days[i][field] = (
                minutes_to_time(
                    predicted
                )
            )

    return days


# ============================================================
# OpenCV horizontal-line detection
# ============================================================

def cluster_positions(values, tolerance=4):

    if not values:
        return []

    values = sorted(values)

    clusters = [
        [values[0]]
    ]

    for value in values[1:]:

        current_average = (
            sum(clusters[-1])
            /
            len(clusters[-1])
        )

        if abs(
            value - current_average
        ) <= tolerance:

            clusters[-1].append(
                value
            )

        else:
            clusters.append(
                [value]
            )

    return [
        round(
            sum(cluster)
            /
            len(cluster)
        )
        for cluster in clusters
    ]


def get_table_bounds(
    document_dict,
    image_height
):
    table = get_biggest_table(
        document_dict
    )

    if table is None:
        return None

    grid = (
        table
        .get("data", {})
        .get("grid", [])
    )

    xs = []
    ys = []

    for row in grid:

        for cell in row:

            bbox = cell.get(
                "bbox"
            )

            if not bbox:
                continue

            l = bbox.get("l")
            r = bbox.get("r")
            t = bbox.get("t")
            b = bbox.get("b")

            if None in (
                l, r, t, b
            ):
                continue

            xs.extend([
                float(l),
                float(r)
            ])

            ys.extend([
                float(t),
                float(b)
            ])

    if not xs or not ys:
        return None

    return {
        "left":
            max(
                0,
                int(min(xs)) - 10
            ),

        "right":
            int(max(xs)) + 10,

        "top":
            max(
                0,
                int(min(ys)) - 10
            ),

        "bottom":
            min(
                image_height - 1,
                int(max(ys)) + 10
            )
    }


def detect_horizontal_lines(
    image_path,
    bounds
):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return []

    crop = image[
        bounds["top"]:
        bounds["bottom"],

        bounds["left"]:
        bounds["right"]
    ]

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    width = binary.shape[1]

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    30,
                    int(width * 0.45)
                ),
                1
            )
        )
    )

    horizontal = (
        cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )
    )

    projection = np.sum(
        horizontal > 0,
        axis=1
    )

    threshold = (
        width * 0.30
    )

    positions = [
        y
        for y, value
        in enumerate(projection)
        if value >= threshold
    ]

    positions = cluster_positions(
        positions
    )

    return [
        y + bounds["top"]
        for y in positions
    ]


def build_data_bands(lines):

    lines = sorted(lines)

    bands = []

    for i in range(
        len(lines) - 1
    ):

        top = lines[i]
        bottom = lines[i + 1]

        height = (
            bottom - top
        )

        # Actual timetable rows are ~30-40 px.
        if 25 <= height <= 45:

            bands.append({
                "top": top,
                "bottom": bottom,
                "height": height
            })

    return bands


# ============================================================
# Find best missing-row position
# ============================================================

def transition_score(
    before,
    after
):
    """
    Higher = stronger indication that one day
    is missing between these two rows.
    """

    fields = [
        "fajr",
        "sunrise",
        "dhuhr",
        "asr",
        "maghrib",
        "isha",
        "fajr_first"
    ]

    score = 0
    details = {}

    for field in fields:

        a = time_to_minutes(
            before.get(field)
        )

        b = time_to_minutes(
            after.get(field)
        )

        if a is None or b is None:
            continue

        diff = abs(
            b - a
        )

        details[field] = diff

        # 0/1 minute = normal daily variation.
        # 2 minutes = evidence of one skipped day.
        if diff == 2:
            score += 2

        elif diff == 3:
            score += 1

        elif diff >= 4:
            score -= 2

    return score, details


def find_missing_row_index(days):

    best_index = None
    best_score = -999
    best_details = {}

    for i in range(
        len(days) - 1
    ):

        score, details = (
            transition_score(
                days[i],
                days[i + 1]
            )
        )

        if score > best_score:

            best_score = score
            best_index = i + 1
            best_details = details

    return (
        best_index,
        best_score,
        best_details
    )


# ============================================================
# Reconstruct one missing row
# ============================================================

def interpolate_time(a, b):

    a_minutes = time_to_minutes(a)
    b_minutes = time_to_minutes(b)

    if (
        a_minutes is None
        and b_minutes is None
    ):
        return None

    if a_minutes is None:
        return b

    if b_minutes is None:
        return a

    if abs(
        a_minutes - b_minutes
    ) > 20:
        return a

    return minutes_to_time(
        (
            a_minutes
            +
            b_minutes
        ) / 2
    )


def reconstruct_row(
    before,
    after
):

    fields = [
        "fajr",
        "sunrise",
        "dhuhr",
        "asr",
        "maghrib",
        "isha",
        "fajr_first"
    ]

    result = {
        "grid_index": None,
        "reconstructed": True,
        "review_required": True,
        "reconstruction_source": "interpolation"
    }

    for field in fields:

        result[field] = (
            interpolate_time(
                before.get(field),
                after.get(field)
            )
        )

    return result


# ============================================================
# Validation
# ============================================================

def validate_day(day):

    required = [
        "fajr",
        "sunrise",
        "dhuhr",
        "asr",
        "maghrib",
        "isha"
    ]

    missing = [
        field
        for field in required
        if day.get(field) is None
    ]

    return {
        "valid":
            len(missing) == 0,

        "missing":
            missing
    }


# ============================================================
# API
# ============================================================

@app.post("/extract")
async def extract_imsakia(
    file: UploadFile = File(...)
):

    temp_path = None

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Empty file"
            )

        suffix = (
            Path(
                file.filename
                or "imsakia.jpg"
            ).suffix
            or ".jpg"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(contents)

            temp_path = Path(
                tmp.name
            )

        image = cv2.imread(
            str(temp_path)
        )

        if image is None:

            raise HTTPException(
                status_code=400,
                detail="Could not read image"
            )

        # ------------------------------------------------
        # Docling
        # ------------------------------------------------

        result = converter.convert(
            str(temp_path)
        )

        document_dict = (
            result
            .document
            .export_to_dict()
        )

        table = get_biggest_table(
            document_dict
        )

        if table is None:

            raise HTTPException(
                status_code=422,
                detail="No table detected"
            )

        table_data = (
            table.get(
                "data",
                {}
            )
        )

        raw_rows = (
            extract_docling_rows(
                document_dict
            )
        )

        days = normalize_rows(
            raw_rows
        )

        # IMPORTANT:
        # restore our stable repair step
        days = repair_missing_cells(
            days
        )

        rows_from_docling = (
            len(days)
        )

        # ------------------------------------------------
        # OpenCV count of actual physical rows
        # ------------------------------------------------

        bounds = get_table_bounds(
            document_dict,
            image.shape[0]
        )

        horizontal_lines = []

        physical_bands = []

        if bounds:

            horizontal_lines = (
                detect_horizontal_lines(
                    temp_path,
                    bounds
                )
            )

            physical_bands = (
                build_data_bands(
                    horizontal_lines
                )
            )

        physical_row_count = (
            len(physical_bands)
        )

        reconstructed_index = None
        gap_score = None
        gap_details = None

        # ------------------------------------------------
        # Only reconstruct if IMAGE proves
        # there is exactly one extra physical row
        # ------------------------------------------------

        if (
            physical_row_count
            ==
            rows_from_docling + 1
        ):

            (
                reconstructed_index,
                gap_score,
                gap_details
            ) = find_missing_row_index(
                days
            )

            if (
                reconstructed_index
                is not None
                and
                reconstructed_index > 0
                and
                reconstructed_index < len(days)
            ):

                new_row = reconstruct_row(
                    days[
                        reconstructed_index - 1
                    ],
                    days[
                        reconstructed_index
                    ]
                )

                days.insert(
                    reconstructed_index,
                    new_row
                )

        # ------------------------------------------------
        # Final validation
        # ------------------------------------------------

        valid_rows = 0
        invalid_rows = 0
        reconstructed_rows = 0
        review_rows = []

        for index, day in enumerate(days):

            day["row"] = (
                index + 1
            )

            validation = (
                validate_day(day)
            )

            day["valid"] = (
                validation["valid"]
            )

            day["missing"] = (
                validation["missing"]
            )

            if day["valid"]:
                valid_rows += 1
            else:
                invalid_rows += 1

            if day.get(
                "reconstructed"
            ):
                reconstructed_rows += 1

                review_rows.append({
                    "row": day["row"],
                    "reason": "missing_row_reconstructed",
                    "source": day.get(
                        "reconstruction_source",
                        "interpolation"
                    ),
                    "suggested_values": {
                        "fajr": day.get("fajr"),
                        "sunrise": day.get("sunrise"),
                        "dhuhr": day.get("dhuhr"),
                        "asr": day.get("asr"),
                        "maghrib": day.get("maghrib"),
                        "isha": day.get("isha"),
                        "fajr_first": day.get("fajr_first"),
                    }
                })

        requires_user_review = (
            len(review_rows) > 0
        )

        return {
            "success": True,

            "docling_grid_rows":
                table_data.get(
                    "num_rows",
                    0
                ),

            "docling_grid_cols":
                table_data.get(
                    "num_cols",
                    0
                ),

            "docling_prayer_rows":
                rows_from_docling,

            "horizontal_lines_detected":
                len(
                    horizontal_lines
                ),

            "physical_data_rows":
                physical_row_count,

            "rows_final":
                len(days),

            "reconstructed_rows":
                reconstructed_rows,

            "reconstructed_after_row":
                (
                    reconstructed_index
                    if reconstructed_index
                    is not None
                    else None
                ),

            "gap_score":
                gap_score,

            "gap_details":
                gap_details,

            "valid_rows":
                valid_rows,

            "invalid_rows":
                invalid_rows,

            "requires_user_review":
                requires_user_review,

            "review_message":
                (
                    "One or more timetable rows were missing from OCR and were estimated. "
                    "Please review the highlighted row(s), edit them if needed, or accept the suggested values."
                    if requires_user_review
                    else None
                ),

            "review_rows":
                review_rows,

            "days":
                days
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if (
            temp_path is not None
            and temp_path.exists()
        ):

            try:
                temp_path.unlink()

            except Exception:
                pass