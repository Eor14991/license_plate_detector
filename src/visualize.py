import ast
import cv2
import numpy as np
import pandas as pd
import os


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10,
                line_length_x=200, line_length_y=200):
    """Draws fancy L-shaped borders around a bounding box."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Bottom-left
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Top-right
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-right
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def safe_parse_bbox(bbox_str):
    """Parses bbox string safely into integers [x1, y1, x2, y2]."""
    try:
        return list(map(int, ast.literal_eval(
            bbox_str.replace('[ ', '[')
                    .replace('   ', ' ')
                    .replace('  ', ' ')
                    .replace(' ', ',')
        )))
    except Exception as e:
        print(f"[WARN] Failed to parse bbox: {bbox_str} | {e}")
        return [0, 0, 0, 0]


# =============================
# Load data & setup video paths
# =============================

results = pd.read_csv('./test.csv')

video_path = './temp.mp4'   # input video
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# ====================================
# Extract best license plate per car_id
# ====================================

license_plate = {}
for car_id in np.unique(results['car_id']):
    car_rows = results[results['car_id'] == car_id]
    if car_rows.empty:
        continue

    # pick best scored license
    max_score = np.amax(car_rows['license_number_score'])
    best_row = car_rows[car_rows['license_number_score'] == max_score].iloc[0]

    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': best_row['license_number']
    }

    frame_idx = int(best_row['frame_nmr'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"[WARN] Could not read frame {frame_idx} for car {car_id}")
        continue

    # parse license plate bbox
    x1, y1, x2, y2 = safe_parse_bbox(best_row['license_plate_bbox'])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x2 > x1 and y2 > y1:
        license_crop = frame[y1:y2, x1:x2, :]
        if license_crop.size > 0:
            license_crop = cv2.resize(
                license_crop,
                (int((x2 - x1) * 400 / (y2 - y1)), 400)
            )
            license_plate[car_id]['license_crop'] = license_crop


# ======================
# Draw results on frames
# ======================


frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_nmr += 1

    df_ = results[results['frame_nmr'] == frame_nmr]
    for _, row in df_.iterrows():
        car_x1, car_y1, car_x2, car_y2 = safe_parse_bbox(row['car_bbox'])
        draw_border(frame, (car_x1, car_y1), (car_x2, car_y2),
                    (0, 255, 0), 25, 200, 200)

        # license plate box
        x1, y1, x2, y2 = safe_parse_bbox(row['license_plate_bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

        # overlay license crop
        car_id = row['car_id']
        license_crop = license_plate.get(car_id, {}).get('license_crop')
        if license_crop is None:
            continue

        H, W, _ = license_crop.shape
        try:
            # clamp overlay position
            x_start = max(0, int((car_x2 + car_x1 - W) / 2))
            x_end = min(width, x_start + W)
            y_start = max(0, car_y1 - H - 100)
            y_end = min(height, y_start + H)

            if y_end > y_start and x_end > x_start:
                frame[y_start:y_end, x_start:x_end, :] = license_crop[:y_end-y_start, :x_end-x_start]

            # white box for text
            text_box_y1 = max(0, car_y1 - H - 400)
            text_box_y2 = max(0, car_y1 - H - 100)
            frame[text_box_y1:text_box_y2, x_start:x_end, :] = (255, 255, 255)

            # license number text
            license_num = str(license_plate[car_id]['license_plate_number'])
            (text_width, text_height), _ = cv2.getTextSize(
                license_num, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17
            )
            text_x = int((car_x2 + car_x1 - text_width) / 2)
            text_y = int(car_y1 - H - 250 + (text_height / 2))
            text_x = max(0, min(width - text_width, text_x))
            text_y = max(text_height, min(height - 10, text_y))

            cv2.putText(frame, license_num, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
        except Exception as e:
            print(f"[WARN] Overlay failed for car {car_id} on frame {frame_nmr}: {e}")

    out.write(frame)

out.release()
cap.release()
print("Video saved to ./out.mp4")
