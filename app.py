from pathlib import Path
import tempfile
import re
import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from rapidocr import RapidOCR

app = FastAPI(title="Munib Imsakia API")
ocr_engine = RapidOCR()

@app.get("/")
def root():
    return {"status": "ok", "service": "munib-imsakia", "ocr_backend": "rapidocr-onnxruntime", "ocr_mode": "grid-batched-recognition-only"}

def clean_cell(value):
    if value is None:
        return ""
    return (str(value).strip().replace("|","").replace("[","").replace("]","")
            .replace("(","").replace(")","").replace(" ",""))

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
    return [round(sum(c)/len(c)) for c in clusters]

def normalize_time(value, previous_hour, min_hour, max_hour):
    value = clean_cell(value).replace("٫",":").replace("،",":").replace(";",":").replace("：",":")
    if not value:
        return None, previous_hour
    full = re.search(r"(\d{1,2}):(\d{2})", value)
    if full:
        hour, minute = int(full.group(1)), int(full.group(2))
        if 0 <= minute <= 59:
            if min_hour <= hour <= max_hour:
                return f"{hour:02d}:{minute:02d}", hour
            if previous_hour is not None:
                return f"{previous_hour:02d}:{minute:02d}", previous_hour
    short = re.search(r":(\d{2})", value)
    if short and previous_hour is not None:
        minute = int(short.group(1))
        if 0 <= minute <= 59:
            return f"{previous_hour:02d}:{minute:02d}", previous_hour
    return None, previous_hour

def convert_to_24h(value, prayer):
    if value is None:
        return None
    hour, minute = map(int, value.split(":"))
    if prayer == "asr" and 1 <= hour <= 6:
        hour += 12
    elif prayer == "maghrib" and 5 <= hour <= 8:
        hour += 12
    elif prayer == "isha" and 7 <= hour <= 11:
        hour += 12
    return f"{hour:02d}:{minute:02d}"

def _box_center(box):
    pts = np.asarray(box, dtype=float).reshape(-1, 2)
    return float(np.mean(pts[:,0])), float(np.mean(pts[:,1]))

def run_ocr(image):
    result = ocr_engine(image)
    items = []
    txts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    if txts is not None and boxes is not None:
        for text, box in zip(txts, boxes):
            if text and box is not None:
                cx, cy = _box_center(box)
                items.append({"text": str(text), "cx": cx, "cy": cy})
        return items
    candidate = result[0] if isinstance(result, tuple) and result else result
    if isinstance(candidate, (list, tuple)):
        for row in candidate:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                box, text = row[0], row[1]
                try:
                    cx, cy = _box_center(box)
                except Exception:
                    continue
                items.append({"text": str(text), "cx": cx, "cy": cy})
    return items

def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,15)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT,(max(30,int(w*0.35)),1))
    horizontal = cv2.morphologyEx(binary,cv2.MORPH_OPEN,hk)
    hp = np.sum(horizontal > 0, axis=1)
    hlines = cluster_positions([y for y,v in enumerate(hp) if v >= w*0.25])
    vk = cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(30,int(h*0.25))))
    vertical = cv2.morphologyEx(binary,cv2.MORPH_OPEN,vk)
    vp = np.sum(vertical > 0, axis=0)
    vlines = cluster_positions([x for x,v in enumerate(vp) if v >= h*0.20])
    return hlines, vlines

def best_8_vertical(vlines, width):
    if len(vlines) < 8:
        return None
    best, best_score = None, -999
    for i in range(len(vlines)-7):
        cand = vlines[i:i+8]
        span = cand[-1]-cand[0]
        if span < width*0.40:
            continue
        gaps = np.diff(cand)
        mean = float(np.mean(gaps))
        if mean <= 0:
            continue
        score = span/width - float(np.std(gaps)/mean)*0.35
        if score > best_score:
            best_score, best = score, cand
    return best

def densest_horizontal_run(hlines):
    if not hlines:
        return []
    best, cur = [], [hlines[0]]
    for y in hlines[1:]:
        gap = y-cur[-1]
        if 18 <= gap <= 70:
            cur.append(y)
        elif gap >= 18:
            if len(cur) > len(best):
                best = cur
            cur = [y]
    if len(cur) > len(best):
        best = cur
    return best if len(best) >= 20 else hlines

def build_data_bands(lines):
    lines = sorted(lines)
    return [{"top":lines[i],"bottom":lines[i+1],"height":lines[i+1]-lines[i]}
            for i in range(len(lines)-1) if 18 <= lines[i+1]-lines[i] <= 70]

def prepare_cell_for_recognition(cell):
    """
    Prepare one already-segmented timetable cell for recognition only.

    The grid gives us the text location already, so running RapidOCR's detector
    again is wasted CPU. We crop a few pixels away from the borders and let only
    the recognition model read the time.
    """
    if cell is None or cell.size == 0:
        return None

    h, w = cell.shape[:2]
    inset_x = max(2, int(w * 0.03))
    inset_y = max(2, int(h * 0.08))

    if w > inset_x * 2 + 4 and h > inset_y * 2 + 4:
        cell = cell[inset_y:h-inset_y, inset_x:w-inset_x]

    # A modest, fixed text height is enough for the recognizer and avoids
    # feeding unnecessarily large images to ONNX Runtime.
    target_h = 40
    ch, cw = cell.shape[:2]
    if ch > 0 and ch != target_h:
        scale = target_h / ch
        new_w = max(24, min(220, int(round(cw * scale))))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        cell = cv2.resize(cell, (new_w, target_h), interpolation=interpolation)

    # White padding keeps digits away from the edge and improves recognition
    # of leading/trailing digits such as 03:48.
    cell = cv2.copyMakeBorder(
        cell, 4, 4, 6, 6,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return cell


def recognize_cells_batch(cell_images, batch_size=64):
    """
    Recognition-only OCR.

    RapidOCR's normal call performs DET -> CLS -> REC. Because our OpenCV grid
    already segmented every cell, we bypass DET and CLS completely and send
    the cropped cells directly to the recognition model in batches.
    """
    texts = []

    for start in range(0, len(cell_images), batch_size):
        batch = cell_images[start:start + batch_size]
        if not batch:
            continue

        rec_res = ocr_engine.recognize_txt(batch)
        batch_texts = getattr(rec_res, "txts", None)

        if batch_texts is None:
            texts.extend([""] * len(batch))
            continue

        batch_texts = list(batch_texts)
        if len(batch_texts) < len(batch):
            batch_texts.extend([""] * (len(batch) - len(batch_texts)))

        texts.extend(clean_cell(t) for t in batch_texts[:len(batch)])

    return texts


def extract_grid_rows(image, hlines, vlines):
    """
    Extract the 7 prayer-time columns using the already detected grid.

    Old version:
      one huge table crop -> full RapidOCR detector/classifier/recognizer
      (~286 seconds on Render Free in the measured request)

    New version:
      grid cells -> recognition model only, batched
    """
    candidate_rows = []
    cell_images = []
    cell_slots = []

    for ri in range(len(hlines) - 1):
        y1, y2 = hlines[ri], hlines[ri + 1]
        if not (18 <= y2 - y1 <= 70):
            continue

        candidate_rows.append(ri)

        for ci in range(7):
            x1, x2 = vlines[ci], vlines[ci + 1]

            # Crop inside the detected borders so the recognizer never sees
            # the table lines themselves.
            pad_x = max(1, int((x2 - x1) * 0.02))
            pad_y = max(1, int((y2 - y1) * 0.05))

            xa = max(0, x1 + pad_x)
            xb = min(image.shape[1], x2 - pad_x)
            ya = max(0, y1 + pad_y)
            yb = min(image.shape[0], y2 - pad_y)

            crop = image[ya:yb, xa:xb]
            prepared = prepare_cell_for_recognition(crop)

            if prepared is None:
                # Keep positional alignment with the 7-column grid.
                prepared = np.full((48, 48, 3), 255, dtype=np.uint8)

            cell_slots.append((ri, ci))
            cell_images.append(prepared)

    if not cell_images:
        return []

    recognized = recognize_cells_batch(cell_images)

    row_map = {
        ri: [""] * 7
        for ri in candidate_rows
    }

    for (ri, ci), text in zip(cell_slots, recognized):
        row_map[ri][ci] = clean_cell(text)

    rows = []
    for ri in candidate_rows:
        cells = row_map[ri]
        time_count = sum(
            bool(re.search(r"\d{1,2}:\d{2}|:\d{2}", value))
            for value in cells
        )

        # Same conservative filter as before: only rows containing at least
        # three time-looking cells are treated as prayer timetable rows.
        if time_count >= 3:
            rows.append({
                "grid_index": ri,
                "cells": cells,
            })

    return rows

def normalize_rows(rows):
    hours = {k:None for k in ["isha","maghrib","asr","dhuhr","sunrise","fajr","fajr_first"]}
    result = []
    for item in rows:
        row = item["cells"]
        if len(row) != 7: continue
        isha,hours["isha"] = normalize_time(row[0],hours["isha"],7,11)
        maghrib,hours["maghrib"] = normalize_time(row[1],hours["maghrib"],5,8)
        asr,hours["asr"] = normalize_time(row[2],hours["asr"],3,6)
        dhuhr,hours["dhuhr"] = normalize_time(row[3],hours["dhuhr"],11,13)
        sunrise,hours["sunrise"] = normalize_time(row[4],hours["sunrise"],5,7)
        fajr,hours["fajr"] = normalize_time(row[5],hours["fajr"],3,6)
        fajr_first,hours["fajr_first"] = normalize_time(row[6],hours["fajr_first"],3,6)
        result.append({
            "grid_index":item["grid_index"],"fajr":fajr,"sunrise":sunrise,"dhuhr":dhuhr,
            "asr":convert_to_24h(asr,"asr"),"maghrib":convert_to_24h(maghrib,"maghrib"),
            "isha":convert_to_24h(isha,"isha"),"fajr_first":fajr_first,
            "reconstructed":False,"review_required":False,"reconstruction_source":None
        })
    return result

def time_to_minutes(value):
    if value is None: return None
    h,m = map(int,value.split(":"))
    return h*60+m

def minutes_to_time(value):
    value = int(round(value))
    return f"{value//60:02d}:{value%60:02d}"

def repair_missing_cells(days):
    fields=["fajr","sunrise","dhuhr","asr","maghrib","isha","fajr_first"]
    for field in fields:
        for i in range(len(days)):
            if days[i].get(field) is not None: continue
            p=n=None
            for j in range(i-1,-1,-1):
                if days[j].get(field) is not None: p=j; break
            for j in range(i+1,len(days)):
                if days[j].get(field) is not None: n=j; break
            if p is None or n is None: continue
            a,b=time_to_minutes(days[p][field]),time_to_minutes(days[n][field])
            gap=n-p
            if gap <= 0 or abs(b-a)>20: continue
            days[i][field]=minutes_to_time(a+(b-a)/gap*(i-p))
    return days

def transition_score(before, after):
    fields=["fajr","sunrise","dhuhr","asr","maghrib","isha","fajr_first"]
    score,details=0,{}
    for field in fields:
        a,b=time_to_minutes(before.get(field)),time_to_minutes(after.get(field))
        if a is None or b is None: continue
        diff=abs(b-a); details[field]=diff
        if diff==2: score+=2
        elif diff==3: score+=1
        elif diff>=4: score-=2
    return score,details

def find_missing_row_index(days):
    bi,bs,bd=None,-999,{}
    for i in range(len(days)-1):
        s,d=transition_score(days[i],days[i+1])
        if s>bs: bi,bs,bd=i+1,s,d
    return bi,bs,bd

def interpolate_time(a,b):
    am,bm=time_to_minutes(a),time_to_minutes(b)
    if am is None and bm is None:return None
    if am is None:return b
    if bm is None:return a
    if abs(am-bm)>20:return a
    return minutes_to_time((am+bm)/2)

def reconstruct_row(before, after):
    fields=["fajr","sunrise","dhuhr","asr","maghrib","isha","fajr_first"]
    r={"grid_index":None,"reconstructed":True,"review_required":True,"reconstruction_source":"interpolation"}
    for field in fields:r[field]=interpolate_time(before.get(field),after.get(field))
    return r

def validate_day(day):
    required=["fajr","sunrise","dhuhr","asr","maghrib","isha"]
    missing=[f for f in required if day.get(f) is None]
    return {"valid":not missing,"missing":missing}

@app.post("/extract")
async def extract_imsakia(file: UploadFile = File(...)):
    temp_path=None
    request_started = time.perf_counter()
    timings = {}
    def mark(name, started):
        elapsed = time.perf_counter() - started
        timings[name] = round(elapsed, 3)
        print(f"[PERF] {name}: {elapsed:.3f}s", flush=True)
        return elapsed
    try:
        step_started = time.perf_counter()
        contents=await file.read()
        mark("read_upload", step_started)
        if not contents:
            raise HTTPException(status_code=400,detail="Empty file")
        suffix=Path(file.filename or "imsakia.jpg").suffix or ".jpg"
        step_started = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
            tmp.write(contents); temp_path=Path(tmp.name)
        mark("write_temp", step_started)

        step_started = time.perf_counter()
        image=cv2.imread(str(temp_path))
        mark("decode_image", step_started)
        if image is None:
            raise HTTPException(status_code=400,detail="Could not read image")

        # Cap image size to reduce peak RAM.
        step_started = time.perf_counter()
        if image.shape[1] > 2200:
            scale=2200/image.shape[1]
            image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        mark("resize_image", step_started)

        step_started = time.perf_counter()
        hlines,vlines=detect_lines(image)
        mark("detect_grid_lines", step_started)
        step_started = time.perf_counter()
        vgrid=best_8_vertical(vlines,image.shape[1])
        hgrid=densest_horizontal_run(hlines)
        mark("select_grid", step_started)
        if vgrid is None or len(hgrid)<20:
            raise HTTPException(status_code=422,detail="Could not detect timetable grid")

        step_started = time.perf_counter()
        raw_rows=extract_grid_rows(image,hgrid,vgrid)
        mark("rapidocr_recognition_only", step_started)

        step_started = time.perf_counter()
        days=repair_missing_cells(normalize_rows(raw_rows))
        mark("normalize_and_repair", step_started)
        rows_from_ocr=len(days)
        physical_row_count=len(build_data_bands(hgrid))

        reconstructed_index=gap_score=gap_details=None
        if physical_row_count == rows_from_ocr + 1:
            reconstructed_index,gap_score,gap_details=find_missing_row_index(days)
            if reconstructed_index is not None and 0 < reconstructed_index < len(days):
                days.insert(reconstructed_index,reconstruct_row(days[reconstructed_index-1],days[reconstructed_index]))

        valid_rows=invalid_rows=reconstructed_rows=0
        review_rows=[]
        for index,day in enumerate(days):
            day["row"]=index+1
            validation=validate_day(day)
            day["valid"]=validation["valid"]; day["missing"]=validation["missing"]
            if day["valid"]: valid_rows+=1
            else: invalid_rows+=1
            if day.get("reconstructed"):
                reconstructed_rows+=1
                review_rows.append({
                    "row":day["row"],"reason":"missing_row_reconstructed",
                    "source":day.get("reconstruction_source","interpolation"),
                    "suggested_values":{k:day.get(k) for k in ["fajr","sunrise","dhuhr","asr","maghrib","isha","fajr_first"]}
                })

        requires_user_review=bool(review_rows)
        total_elapsed = time.perf_counter() - request_started
        timings["total"] = round(total_elapsed, 3)
        print(f"[PERF] TOTAL: {total_elapsed:.3f}s | {timings}", flush=True)
        return {
            "success":True,
            "docling_grid_rows":len(hgrid),
            "docling_grid_cols":7,
            "docling_prayer_rows":rows_from_ocr,
            "horizontal_lines_detected":len(hgrid),
            "physical_data_rows":physical_row_count,
            "rows_final":len(days),
            "reconstructed_rows":reconstructed_rows,
            "reconstructed_after_row":reconstructed_index,
            "gap_score":gap_score,
            "gap_details":gap_details,
            "valid_rows":valid_rows,
            "invalid_rows":invalid_rows,
            "requires_user_review":requires_user_review,
            "review_message":(
                "One or more timetable rows were missing from OCR and were estimated. "
                "Please review the highlighted row(s), edit them if needed, or accept the suggested values."
                if requires_user_review else None
            ),
            "review_rows":review_rows,
            "ocr_backend":"rapidocr-onnxruntime",
            "processing_ms": int(total_elapsed * 1000),
            "timings_seconds": timings,
            "days":days,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    finally:
        if temp_path is not None and temp_path.exists():
            try: temp_path.unlink()
            except Exception: pass
