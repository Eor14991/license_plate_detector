import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w') as f:
        f.write(
            'frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id] and 'license_plate' in results[frame_nmr][car_id] and \
                        'text' in results[frame_nmr][car_id]['license_plate']:
                    car = results[frame_nmr][car_id]['car']
                    lp = results[frame_nmr][car_id]['license_plate']

                    car_bbox_str = '[{} {} {} {}]'.format(car['bbox'][0], car['bbox'][1], car['bbox'][2],
                                                          car['bbox'][3])
                    lp_bbox_str = '[{} {} {} {}]'.format(lp['bbox'][0], lp['bbox'][1], lp['bbox'][2], lp['bbox'][3])

                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        car_bbox_str,
                        lp_bbox_str,
                        lp['bbox_score'],
                        lp['text'],
                        lp['text_score']
                    ))


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char) and \
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char) and \
            (text[2] in '0123456789' or text[2] in dict_char_to_int) and \
            (text[3] in '0123456789' or text[3] in dict_char_to_int) and \
            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char) and \
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char) and \
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char,
               5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(len(text)):
        if j in mapping and text[j] in mapping[j]:
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    This function now iterates through all detections and returns the first one
    that complies with the license plate format.
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    # Return None if no compliant license plate is found
    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    This function now uses Intersection over Union (IoU) for robust matching.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    max_iou = 0
    best_car_id = -1
    best_car_bbox = (-1, -1, -1, -1)

    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        # Calculate intersection area
        inter_x1 = max(x1, xcar1)
        inter_y1 = max(y1, ycar1)
        inter_x2 = min(x2, xcar2)
        inter_y2 = min(y2, ycar2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union area
        plate_area = (x2 - x1) * (y2 - y1)
        car_area = (xcar2 - xcar1) * (ycar2 - ycar1)
        union_area = plate_area + car_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0

        if iou > max_iou:
            max_iou = iou
            best_car_id = car_id
            best_car_bbox = (xcar1, ycar1, xcar2, ycar2)

    # Return the car with the highest IoU, assuming it's a reasonable match (e.g., IoU > 0)
    if max_iou > 0:
        return best_car_bbox[0], best_car_bbox[1], best_car_bbox[2], best_car_bbox[3], best_car_id

    return -1, -1, -1, -1, -1