import re
import time

import cv2
import numpy as np
import pytesseract
from fastapi import HTTPException, UploadFile

COLS = ["isha", "maghrib", "asr", "dhuhr", "sunrise", "fajr", "fajr_first"]
HOUR_RANGES = [(18, 23), (17, 20), (14, 18), (11, 13), (4, 8), (3, 6), (3, 6)]


def cluster_positions(values, tolerance=4):
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for value in values[1:]:
        avg = sum(clusters[-1]) / len(clusters[-1])
        if abs(value - avg) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [round(sum(c) / len(c)) for c in clusters]


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.35)), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    hp = np.sum(horizontal > 0, axis=1)
    hlines = cluster_positions([y for y, v in enumerate(hp) if v >= w * 0.25])

    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * 0.25))))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    vp = np.sum(vertical > 0, axis=0)
    vlines = cluster_positions([x for x, v in enumerate(vp) if v >= h * 0.20])
    return hlines, vlines


def best_8_vertical(vlines, width):
    if len(vlines) < 8:
        return None
    best, best_score = None, -999
    for i in range(len(vlines) - 7):
        cand = vlines[i : i + 8]
        span = cand[-1] - cand[0]
        if span < width * 0.40:
            continue
        gaps = np.diff(cand)
        mean = float(np.mean(gaps))
        if mean <= 0:
            continue
        score = span / width - float(np.std(gaps) / mean) * 0.35
        if score > best_score:
            best_score, best = score, cand
    return best


def data_horizontal_run(lines):
    if not lines:
        return []
    runs, current = [], [lines[0]]
    for y in lines[1:]:
        gap = y - current[-1]
        if 18 <= gap <= 45:
            current.append(y)
        else:
            runs.append(current)
            current = [y]
    runs.append(current)
    best = max(runs, key=len)
    if len(best) >= 20:
        return best

    best, current = [], [lines[0]]
    for y in lines[1:]:
        gap = y - current[-1]
        if 18 <= gap <= 70:
            current.append(y)
        elif gap >= 18:
            if len(current) > len(best):
                best = current
            current = [y]
    if len(current) > len(best):
        best = current
    return best


def preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    hk = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(25, int(crop.shape[1] * 0.10)), 1)
    )
    vk = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, int(crop.shape[0] * 0.03)))
    )
    lines = cv2.bitwise_or(
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk),
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk),
    )
    return 255 - cv2.subtract(inv, lines)


def boxes_to_cells(image, hgrid, vgrid, y0, x0, psm):
    boxes = pytesseract.image_to_boxes(
        image,
        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789:",
    )
    height = image.shape[0]
    vr = [x - x0 for x in vgrid]
    hr = [y - y0 for y in hgrid]
    cellchars = [[[] for _ in range(7)] for _ in range(len(hr) - 1)]

    for line in boxes.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        ch = parts[0]
        try:
            left, bottom, right, top = map(int, parts[1:5])
        except ValueError:
            continue
        cx = (left + right) / 2
        cy = height - (bottom + top) / 2
        ri = next((i for i in range(len(hr) - 1) if hr[i] <= cy <= hr[i + 1]), None)
        ci = next((i for i in range(7) if vr[i] <= cx <= vr[i + 1]), None)
        if ri is not None and ci is not None:
            cellchars[ri][ci].append((cx, ch))

    return [
        ["".join(ch for _, ch in sorted(cellchars[r][c])) for c in range(7)]
        for r in range(len(hr) - 1)
    ]


def ocr_one_row(row_image, vgrid, x0):
    boxes = pytesseract.image_to_boxes(
        row_image,
        config="--psm 7 -c tessedit_char_whitelist=0123456789:",
    )
    vr = [x - x0 for x in vgrid]
    cells = [[] for _ in range(7)]
    for line in boxes.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        ch = parts[0]
        try:
            left, _, right, _ = map(int, parts[1:5])
        except ValueError:
            continue
        cx = (left + right) / 2
        ci = next((i for i in range(7) if vr[i] <= cx <= vr[i + 1]), None)
        if ci is not None:
            cells[ci].append((cx, ch))
    return ["".join(ch for _, ch in sorted(cell)) for cell in cells]


def candidate_scores(raw, ci):
    raw = raw or ""
    digits = "".join(re.findall(r"\d", raw))
    out = {}
    low_hour, high_hour = HOUR_RANGES[ci]

    def add(value, penalty):
        if low_hour * 60 <= value <= high_hour * 60 + 59:
            out[value] = min(out.get(value, 99), penalty)

    if not digits:
        return out

    full = re.search(r"(\d{1,2}):(\d{2})", raw)
    if full:
        hour, minute = int(full.group(1)), int(full.group(2))
        if minute < 60:
            if ci <= 2 and hour < 12:
                hour += 12
            add(hour * 60 + minute, 0)

    if len(digits) >= 3:
        hour, minute = int(digits[-3]), int(digits[-2:])
        if minute < 60:
            if ci <= 2 and hour < 12:
                hour += 12
            add(hour * 60 + minute, 2)

    if len(digits) >= 4:
        hour, minute = int(digits[-4:-2]), int(digits[-2:])
        if minute < 60:
            add(hour * 60 + minute, 2)

    if len(digits) >= 2:
        minute = int(digits[-2:])
        if minute < 60:
            penalty = 1 if len(digits) == 2 else 4
            for hour in range(low_hour, high_hour + 1):
                add(hour * 60 + minute, penalty)

    if len(digits) == 1:
        last_digit = int(digits)
        for tens in range(6):
            for hour in range(low_hour, high_hour + 1):
                add(hour * 60 + tens * 10 + last_digit, 7)

    return out


def decode_column(raw_pairs, ci):
    low_hour, high_hour = HOUR_RANGES[ci]
    states = np.arange(low_hour * 60, high_hour * 60 + 60, dtype=np.int32)
    count = len(states)
    transition = np.maximum(0, np.abs(states[:, None] - states[None, :]) - 2) * 3.0
    costs, backs = [], []
    previous = None

    for index, (a, b) in enumerate(raw_pairs):
        observation = np.full(count, 10.0, dtype=np.float32)
        for raw in (a, b):
            for value, penalty in candidate_scores(raw, ci).items():
                state_index = value - low_hour * 60
                if 0 <= state_index < count and penalty < observation[state_index]:
                    observation[state_index] = penalty

        if index == 0:
            current = observation
            back = np.full(count, -1, dtype=np.int32)
        else:
            total = transition + previous[None, :]
            back = np.argmin(total, axis=1).astype(np.int32)
            current = observation + total[np.arange(count), back]

        costs.append(current)
        backs.append(back)
        previous = current

    state_index = int(np.argmin(costs[-1]))
    sequence = [int(states[state_index])]
    for index in range(len(raw_pairs) - 1, 0, -1):
        state_index = int(backs[index][state_index])
        sequence.append(int(states[state_index]))
    return list(reversed(sequence))


def extract_image(image):
    timings = {}
    started = time.perf_counter()

    def mark(name, step_started):
        timings[name] = round(time.perf_counter() - step_started, 3)

    step = time.perf_counter()
    hlines, vlines = detect_lines(image)
    mark("detect_grid_lines", step)

    step = time.perf_counter()
    vgrid = best_8_vertical(vlines, image.shape[1])
    hgrid = data_horizontal_run(hlines)
    mark("select_grid", step)
    if vgrid is None or len(hgrid) < 20:
        raise ValueError("Could not detect timetable grid")

    x0, x1 = vgrid[0], vgrid[-1]
    y0, y1 = hgrid[0], hgrid[-1]
    crop = image[y0:y1, x0:x1]

    step = time.perf_counter()
    clean = preprocess(crop)
    mark("preprocess", step)

    step = time.perf_counter()
    rows_psm6 = boxes_to_cells(clean, hgrid, vgrid, y0, x0, 6)
    mark("tesseract_psm6", step)

    step = time.perf_counter()
    rows_psm11 = boxes_to_cells(clean, hgrid, vgrid, y0, x0, 11)
    mark("tesseract_psm11", step)

    step = time.perf_counter()
    fallback_rows = 0
    for ri in range(len(rows_psm6)):
        useful = sum(
            bool(re.search(r"\d{2}", (rows_psm6[ri][ci] or "") + (rows_psm11[ri][ci] or "")))
            for ci in range(7)
        )
        if useful < 4 or ri == len(rows_psm6) - 1:
            row_image = image[hgrid[ri] : hgrid[ri + 1], x0:x1]
            fallback = ocr_one_row(row_image, vgrid, x0)
            fallback_rows += 1
            for ci in range(7):
                if fallback[ci]:
                    if not rows_psm6[ri][ci]:
                        rows_psm6[ri][ci] = fallback[ci]
                    if not rows_psm11[ri][ci]:
                        rows_psm11[ri][ci] = fallback[ci]
    mark("fallback_rows", step)

    step = time.perf_counter()
    sequences = [
        decode_column(
            [(rows_psm6[i][ci], rows_psm11[i][ci]) for i in range(len(rows_psm6))],
            ci,
        )
        for ci in range(7)
    ]
    mark("sequence_decode", step)

    days = []
    for row_index in range(len(rows_psm6)):
        values = [
            f"{sequences[ci][row_index] // 60:02d}:{sequences[ci][row_index] % 60:02d}"
            for ci in range(7)
        ]
        days.append(
            {
                "row": row_index + 1,
                "fajr": values[5],
                "sunrise": values[4],
                "dhuhr": values[3],
                "asr": values[2],
                "maghrib": values[1],
                "isha": values[0],
                "fajr_first": values[6],
                "reconstructed": False,
                "review_required": False,
                "reconstruction_source": None,
                "valid": True,
                "missing": [],
            }
        )

    timings["total"] = round(time.perf_counter() - started, 3)
    return days, timings, fallback_rows


async def extract_imsakia(file: UploadFile):
    request_started = time.perf_counter()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    if image.shape[1] > 2200:
        scale = 2200 / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    try:
        days, timings, fallback_rows = extract_image(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    total_elapsed = time.perf_counter() - request_started
    timings["request_total"] = round(total_elapsed, 3)
    return {
        "success": True,
        "rows_final": len(days),
        "valid_rows": len(days),
        "invalid_rows": 0,
        "reconstructed_rows": 0,
        "requires_user_review": False,
        "review_message": None,
        "review_rows": [],
        "ocr_backend": "tesseract-grid-viterbi",
        "processing_ms": int(total_elapsed * 1000),
        "timings_seconds": timings,
        "fallback_rows": fallback_rows,
        "days": days,
    }
