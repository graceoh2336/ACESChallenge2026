import cv2
import numpy as np
import math
from collections import deque


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROJECT NOTES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This script detects blue emergency-light candidates from video.
#
# Visual states:
# - Blue object = blue object, not trusted yet
# - Possible flash = amber candidate, worth tracking
# - Emergency light = confirmed visual detection
#
# Audio boost idea:
# - Visual detection always runs.
# - If siren audio is detected, the detector becomes more sensitive.
# - This helps detect weaker/farther emergency lights quicker.
#
# Future fusion:
# - Audio detector will provide audio_detected True/False.
# - Fusion script will decide final driver notification.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN SETTINGS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#VIDEO_SOURCE = "emergency_light.mp4"
#VIDEO_SOURCE = "emergency_light_2.mp4"
#VIDEO_SOURCE = "night_emergency_lights.mp4"
#VIDEO_SOURCE = "nolights.mp4"
#VIDEO_SOURCE = "city_lights.mp4"
#VIDEO_SOURCE = "lights.mp4"
#VIDEO_SOURCE = "sky.mp4"
#VIDEO_SOURCE = "more_night.mp4"
#VIDEO_SOURCE = "dash1.mp4"
VIDEO_SOURCE = "oncoming.mp4"
#VIDEO_SOURCE = "pass by.mp4"
#VIDEO_SOURCE = "Garda Skoda Enyaq Siren Demo - Limerick Emergency Videos (1080p).mp4"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

LONG_HISTORY = 20

MAX_MATCH_DISTANCE = 40
MAX_MISSING_FRAMES = 18

TRACK_CONFIDENCE_MAX = 10

NORMAL_VISUAL_POSSIBLE_THRESHOLD = 0.40
NORMAL_VISUAL_DETECTED_THRESHOLD = 0.70

NIGHT_VISUAL_POSSIBLE_THRESHOLD = 0.50
NIGHT_VISUAL_DETECTED_THRESHOLD = 0.82

NORMAL_CONFIRMED_FLASHES_FOR_ALERT = 3
AUDIO_BOOST_CONFIRMED_FLASHES_FOR_ALERT = 2

PRINT_OUTPUT = True
PRINT_EVERY_N_FRAMES = 1

# Simulated audio input for now.
# False = normal visual mode
# True  = siren heard, visual becomes more sensitive
AUDIO_SIREN_DETECTED = True


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BACKGROUND BLUE SETTINGS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BLUE_BACKGROUND_RATIO_THRESHOLD = 0.18
BACKGROUND_CONTEXT_PAD = 18

GREEN_SURROUND_RATIO_MEDIUM = 0.22
GREEN_SURROUND_RATIO_STRONG = 0.35

SMALL_BACKGROUND_BLOB_AREA = 140
UPPER_BACKGROUND_REGION_LIMIT = 0.60
BACKGROUND_CLUSTER_MIN_BLOBS = 6


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADAPTIVE SETTINGS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_adaptive_detection_settings(hsv_frame, audio_siren_detected=False):
    """
    Choose detection settings depending on scene brightness.

    If audio_siren_detected is True, the visual detector becomes more sensitive.
    """

    average_frame_brightness = float(np.mean(hsv_frame[:, :, 2]))

    # --------------------------------------------------------
    # Night mode
    # --------------------------------------------------------
    if average_frame_brightness < 70:
        scene_mode = "night"

        lower_blue = np.array([90, 80, 145])
        upper_blue = np.array([145, 255, 255])

        flash_min_brightness = 160
        flash_change_threshold = 0.32

        required_flash_frames = 6
        required_flash_transitions = 5
        required_brightness_transitions = 4

        min_blob_area = 4
        max_blob_area = 6000

        possible_threshold = NIGHT_VISUAL_POSSIBLE_THRESHOLD
        detected_threshold = NIGHT_VISUAL_DETECTED_THRESHOLD

        min_light_quality_for_alert = 0.45

    # --------------------------------------------------------
    # Bright day mode
    # --------------------------------------------------------
    elif average_frame_brightness > 150:
        scene_mode = "bright_day"

        lower_blue = np.array([95, 140, 100])
        upper_blue = np.array([140, 255, 255])

        flash_min_brightness = 135
        flash_change_threshold = 0.25

        required_flash_frames = 4
        required_flash_transitions = 3
        required_brightness_transitions = 3

        min_blob_area = 6
        max_blob_area = 9000

        possible_threshold = NORMAL_VISUAL_POSSIBLE_THRESHOLD
        detected_threshold = NORMAL_VISUAL_DETECTED_THRESHOLD

        min_light_quality_for_alert = 0.35

    # --------------------------------------------------------
    # Normal mode
    # --------------------------------------------------------
    else:
        scene_mode = "normal"

        lower_blue = np.array([90, 120, 120])
        upper_blue = np.array([142, 255, 255])

        flash_min_brightness = 140
        flash_change_threshold = 0.25

        required_flash_frames = 4
        required_flash_transitions = 3
        required_brightness_transitions = 3

        min_blob_area = 6
        max_blob_area = 9000

        possible_threshold = NORMAL_VISUAL_POSSIBLE_THRESHOLD
        detected_threshold = NORMAL_VISUAL_DETECTED_THRESHOLD

        min_light_quality_for_alert = 0.35

    # --------------------------------------------------------
    # Audio boost mode
    # --------------------------------------------------------
    # If a siren is heard, the visual detector becomes more willing
    # to accept weak/far/small flashing lights.
    if audio_siren_detected:
        min_blob_area = max(3, min_blob_area - 2)

        flash_change_threshold = max(0.22, flash_change_threshold - 0.05)

        possible_threshold = max(0.30, possible_threshold - 0.15)
        detected_threshold = max(0.60, detected_threshold - 0.20)

        min_light_quality_for_alert = max(0.25, min_light_quality_for_alert - 0.10)

        required_flash_frames = max(3, required_flash_frames - 1)
        required_flash_transitions = max(2, required_flash_transitions - 1)
        required_brightness_transitions = max(2, required_brightness_transitions - 1)

        confirmed_flashes_for_alert = AUDIO_BOOST_CONFIRMED_FLASHES_FOR_ALERT
    else:
        confirmed_flashes_for_alert = NORMAL_CONFIRMED_FLASHES_FOR_ALERT

    return {
        "scene_mode": scene_mode,
        "frame_brightness": average_frame_brightness,

        "lower_blue": lower_blue,
        "upper_blue": upper_blue,

        "flash_min_v": flash_min_brightness,
        "flash_change_threshold": flash_change_threshold,

        "required_flash_frames": required_flash_frames,
        "required_flash_transitions": required_flash_transitions,
        "required_brightness_transitions": required_brightness_transitions,

        "min_area": min_blob_area,
        "max_area": max_blob_area,

        "possible_threshold": possible_threshold,
        "detected_threshold": detected_threshold,

        "min_source_quality_for_alert": min_light_quality_for_alert,
        "confirmed_flashes_for_alert": confirmed_flashes_for_alert
    }


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BACKGROUND BLUE ANALYSIS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def analyse_background_blue_context(hsv_frame, blue_mask, x, y, w, h, blob_area):
    """
    Checks whether a blue blob looks like background scenery.

    This helps reduce false positives from:
    - sky through trees
    - sea
    - blue glass
    - blue scenery
    """

    frame_height, frame_width = blue_mask.shape[:2]
    padding = BACKGROUND_CONTEXT_PAD

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame_width, x + w + padding)
    y2 = min(frame_height, y + h + padding)

    surrounding_hsv = hsv_frame[y1:y2, x1:x2]

    if surrounding_hsv.size == 0:
        return {
            "background_score": 0.0,
            "green_ratio": 0.0,
            "upper_background_region": False,
            "small_background_blob": False
        }

    # Green around blue often means sky through trees.
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(surrounding_hsv, lower_green, upper_green)

    green_ratio = cv2.countNonZero(green_mask) / float(green_mask.size + 1e-6)

    centre_y = y + h / 2
    vertical_position = centre_y / float(frame_height)

    upper_background_region = vertical_position < UPPER_BACKGROUND_REGION_LIMIT
    small_background_blob = blob_area <= SMALL_BACKGROUND_BLOB_AREA

    blob_hsv = hsv_frame[y:y+h, x:x+w]
    blob_mask = blue_mask[y:y+h, x:x+w]
    blue_pixels_inside_blob = blob_mask > 0

    if np.count_nonzero(blue_pixels_inside_blob) > 0:
        hue_values = blob_hsv[:, :, 0][blue_pixels_inside_blob]
        saturation_values = blob_hsv[:, :, 1][blue_pixels_inside_blob]
        brightness_values = blob_hsv[:, :, 2][blue_pixels_inside_blob]

        mean_hue = float(np.mean(hue_values))
        mean_saturation = float(np.mean(saturation_values))
        mean_brightness = float(np.mean(brightness_values))
        brightness_variation = float(np.std(brightness_values))
    else:
        mean_hue = 0.0
        mean_saturation = 0.0
        mean_brightness = 0.0
        brightness_variation = 0.0

    sky_like_colour = (
        95 <= mean_hue <= 125 and
        mean_saturation <= 220 and
        mean_brightness >= 130
    )

    smooth_blue = brightness_variation < 35

    background_score = 0.0

    if green_ratio >= GREEN_SURROUND_RATIO_STRONG:
        background_score += 0.45
    elif green_ratio >= GREEN_SURROUND_RATIO_MEDIUM:
        background_score += 0.28

    if sky_like_colour:
        background_score += 0.20

    if smooth_blue:
        background_score += 0.15

    if upper_background_region:
        background_score += 0.12

    if small_background_blob:
        background_score += 0.12

    background_score = max(0.0, min(1.0, background_score))

    return {
        "background_score": background_score,
        "green_ratio": green_ratio,
        "upper_background_region": upper_background_region,
        "small_background_blob": small_background_blob
    }


def estimate_background_cluster_pressure(contours, hsv_frame, blue_mask, adaptive):
    """
    Checks whether there are many small blue blobs in upper/green areas.
    This helps detect sky gaps between tree leaves.
    """

    small_background_like_blobs = 0

    for contour in contours:
        blob_area = cv2.contourArea(contour)

        if blob_area < adaptive["min_area"] or blob_area > SMALL_BACKGROUND_BLOB_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w <= 0 or h <= 0:
            continue

        background_context = analyse_background_blue_context(
            hsv_frame=hsv_frame,
            blue_mask=blue_mask,
            x=x,
            y=y,
            w=w,
            h=h,
            blob_area=blob_area
        )

        if (
            background_context["upper_background_region"] and
            background_context["green_ratio"] >= GREEN_SURROUND_RATIO_MEDIUM
        ):
            small_background_like_blobs += 1

    cluster_pressure = small_background_like_blobs >= BACKGROUND_CLUSTER_MIN_BLOBS

    return {
        "small_background_like_blobs": small_background_like_blobs,
        "cluster_pressure": cluster_pressure
    }


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LIGHT SOURCE QUALITY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def calculate_light_source_quality(
    x, y, w, h,
    blob_area,
    mean_brightness,
    max_brightness,
    blue_ratio,
    frame_width,
    frame_height,
    scene_mode
):
    """
    Scores whether a blue blob looks like a compact light source.
    """

    aspect_ratio = w / float(h + 1e-6)

    width_ratio = w / float(frame_width)
    height_ratio = h / float(frame_height)
    vertical_position = (y + h / 2) / float(frame_height)

    light_source_quality = 0.0

    large_object_penalty = False
    reflection_penalty = False
    weak_source_penalty = False

    # Brightness evidence
    if max_brightness >= 230:
        light_source_quality += 0.28
    elif max_brightness >= 200:
        light_source_quality += 0.22
    elif max_brightness >= 180:
        light_source_quality += 0.14

    if mean_brightness >= 180:
        light_source_quality += 0.20
    elif mean_brightness >= 150:
        light_source_quality += 0.14
    elif mean_brightness >= 125:
        light_source_quality += 0.08

    # Size evidence
    if 3 <= blob_area <= 900:
        light_source_quality += 0.25
    elif 900 < blob_area <= 1800:
        light_source_quality += 0.10
    else:
        light_source_quality -= 0.15

    # Shape evidence
    if 0.35 <= aspect_ratio <= 3.5:
        light_source_quality += 0.15
    else:
        light_source_quality -= 0.10

    # Blue concentration evidence
    if blue_ratio >= 0.35:
        light_source_quality += 0.15
    elif blue_ratio >= 0.18:
        light_source_quality += 0.08
    else:
        light_source_quality -= 0.08

    # Large blue object checks
    if width_ratio > 0.30 and height_ratio > 0.07:
        large_object_penalty = True

    if width_ratio > 0.42:
        large_object_penalty = True

    if blob_area > 2500:
        large_object_penalty = True

    # Night reflection checks
    if scene_mode == "night":
        if vertical_position > 0.72:
            reflection_penalty = True

        if vertical_position > 0.60 and width_ratio > 0.12:
            reflection_penalty = True

        if h > w * 1.5 and vertical_position > 0.55:
            reflection_penalty = True

        if blob_area > 1000:
            reflection_penalty = True

        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            reflection_penalty = True

        if light_source_quality < 0.45:
            weak_source_penalty = True

    light_source_quality = max(0.0, min(1.0, light_source_quality))

    return {
        "source_quality": light_source_quality,
        "large_object_penalty": large_object_penalty,
        "reflection_penalty": reflection_penalty,
        "weak_source_penalty": weak_source_penalty,
        "aspect_ratio": aspect_ratio,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "vertical_position": vertical_position
    }


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DETECTOR CLASS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VisualEmergencyDetector:
    def __init__(self):
        self.tracks = {}
        self.next_track_id = 0
        self.frame_index = 0
        self.previous_blue_mask = None

    def create_track(self, centre_x, centre_y):
        return {
            "position": (centre_x, centre_y),

            "brightness_history": deque(maxlen=LONG_HISTORY),
            "change_history": deque(maxlen=LONG_HISTORY),
            "binary_flash_history": deque(maxlen=LONG_HISTORY),
            "source_quality_history": deque(maxlen=LONG_HISTORY),
            "background_score_history": deque(maxlen=LONG_HISTORY),

            "seen_frames": 0,
            "missing_frames": 0,

            "confidence": 0,
            "confirmed_flash_count": 0
        }

    def clamp_confidence(self, value):
        return max(0, min(TRACK_CONFIDENCE_MAX, value))

    def get_position_label(self, centre_x, frame_width):
        if centre_x < frame_width / 3:
            return "left"
        elif centre_x < 2 * frame_width / 3:
            return "centre"
        else:
            return "right"

    def process_frame(self, frame):
        self.frame_index += 1

        # --------------------------------------------------------
        # Pre-processing
        # --------------------------------------------------------
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        display = frame.copy()

        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        adaptive = get_adaptive_detection_settings(
            hsv_frame,
            audio_siren_detected=AUDIO_SIREN_DETECTED
        )

        # --------------------------------------------------------
        # Blue mask
        # --------------------------------------------------------
        blue_mask = cv2.inRange(
            hsv_frame,
            adaptive["lower_blue"],
            adaptive["upper_blue"]
        )

        # Extra night mask for blue-white glare
        if adaptive["scene_mode"] == "night":
            blue_white_lower = np.array([80, 20, 190])
            blue_white_upper = np.array([150, 150, 255])
            blue_white_mask = cv2.inRange(hsv_frame, blue_white_lower, blue_white_upper)

            blue_mask = cv2.bitwise_or(blue_mask, blue_white_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --------------------------------------------------------
        # Background-blue pressure
        # --------------------------------------------------------
        blue_background_ratio = cv2.countNonZero(blue_mask) / float(blue_mask.size + 1e-6)
        blue_background_pressure = blue_background_ratio > BLUE_BACKGROUND_RATIO_THRESHOLD

        # --------------------------------------------------------
        # Blue change mask
        # --------------------------------------------------------
        if self.previous_blue_mask is None:
            blue_change_mask = np.zeros_like(blue_mask)
        else:
            blue_change_mask = cv2.absdiff(blue_mask, self.previous_blue_mask)
            _, blue_change_mask = cv2.threshold(
                blue_change_mask,
                25,
                255,
                cv2.THRESH_BINARY
            )
            blue_change_mask = cv2.morphologyEx(
                blue_change_mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1
            )

        self.previous_blue_mask = blue_mask.copy()

        # --------------------------------------------------------
        # Find blue blobs
        # --------------------------------------------------------
        contours, _ = cv2.findContours(
            blue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        background_cluster_info = estimate_background_cluster_pressure(
            contours=contours,
            hsv_frame=hsv_frame,
            blue_mask=blue_mask,
            adaptive=adaptive
        )

        background_cluster_pressure = background_cluster_info["cluster_pressure"]

        # Age tracks
        for track_id in self.tracks:
            self.tracks[track_id]["missing_frames"] += 1

        best_track_data = None
        best_track_score = -1

        # ========================================================
        # PROCESS EACH BLUE BLOB
        # ========================================================

        for contour in contours:
            blob_area = cv2.contourArea(contour)

            if blob_area < adaptive["min_area"] or blob_area > adaptive["max_area"]:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h

            if box_area <= 0:
                continue

            aspect_ratio = w / float(h + 1e-6)

            if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                continue

            fill_ratio = blob_area / float(box_area + 1e-6)

            if fill_ratio < 0.08:
                continue

            centre_x = x + w // 2
            centre_y = y + h // 2

            blob_blue_mask = blue_mask[y:y+h, x:x+w]
            blob_change_mask = blue_change_mask[y:y+h, x:x+w]
            blob_hsv = hsv_frame[y:y+h, x:x+w]

            blue_pixel_count = cv2.countNonZero(blob_blue_mask)
            changed_blue_pixel_count = cv2.countNonZero(blob_change_mask)

            if blue_pixel_count == 0:
                continue

            blue_pixels_inside_blob = blob_blue_mask > 0
            brightness_channel = blob_hsv[:, :, 2]

            mean_brightness = float(np.mean(brightness_channel[blue_pixels_inside_blob]))
            max_brightness = float(np.max(brightness_channel[blue_pixels_inside_blob]))

            blue_ratio = blue_pixel_count / float(box_area + 1e-6)
            change_ratio = changed_blue_pixel_count / float(blue_pixel_count + 1e-6)

            brightness_signal = mean_brightness * blue_ratio

            # ----------------------------------------------------
            # Light source quality
            # ----------------------------------------------------
            quality_info = calculate_light_source_quality(
                x=x,
                y=y,
                w=w,
                h=h,
                blob_area=blob_area,
                mean_brightness=mean_brightness,
                max_brightness=max_brightness,
                blue_ratio=blue_ratio,
                frame_width=FRAME_WIDTH,
                frame_height=FRAME_HEIGHT,
                scene_mode=adaptive["scene_mode"]
            )

            light_source_quality = quality_info["source_quality"]

            # ----------------------------------------------------
            # Background context
            # ----------------------------------------------------
            background_context = analyse_background_blue_context(
                hsv_frame=hsv_frame,
                blue_mask=blue_mask,
                x=x,
                y=y,
                w=w,
                h=h,
                blob_area=blob_area
            )

            background_score = background_context["background_score"]
            green_ratio = background_context["green_ratio"]

            # Skip obvious large blue objects during day
            if adaptive["scene_mode"] != "night":
                if quality_info["large_object_penalty"] and light_source_quality < 0.45:
                    continue

            # ----------------------------------------------------
            # Track matching
            # ----------------------------------------------------
            matched_track_id = None
            best_distance = MAX_MATCH_DISTANCE

            for track_id, track in self.tracks.items():
                old_x, old_y = track["position"]
                distance = math.hypot(centre_x - old_x, centre_y - old_y)

                if distance < best_distance:
                    best_distance = distance
                    matched_track_id = track_id

            if matched_track_id is None:
                matched_track_id = self.next_track_id
                self.tracks[matched_track_id] = self.create_track(centre_x, centre_y)
                self.next_track_id += 1

            track = self.tracks[matched_track_id]

            track["position"] = (centre_x, centre_y)
            track["brightness_history"].append(brightness_signal)
            track["change_history"].append(change_ratio)
            track["source_quality_history"].append(light_source_quality)
            track["background_score_history"].append(background_score)

            flash_like_now = 1 if (
                mean_brightness >= adaptive["flash_min_v"] and
                change_ratio >= adaptive["flash_change_threshold"] and
                light_source_quality >= 0.25
            ) else 0

            track["binary_flash_history"].append(flash_like_now)

            track["seen_frames"] += 1
            track["missing_frames"] = 0

            # ====================================================
            # FLASH DETECTION
            # ====================================================

            is_short_flash = False
            is_long_flash = False

            static_penalty = False
            moving_blue_object_penalty = False
            reflection_penalty = False
            weak_source_penalty = False
            large_object_penalty = False

            background_blue_penalty = False
            repeated_background_penalty = False
            cluster_background_penalty = False
            general_background_penalty = False

            # Short flash check
            if len(track["change_history"]) >= 3:
                recent_change = list(track["change_history"])[-3:]
                recent_brightness = list(track["brightness_history"])[-3:]
                recent_quality = list(track["source_quality_history"])[-3:]

                peak_change = max(recent_change)
                brightness_jump = max(recent_brightness) - min(recent_brightness)
                quality_peak = max(recent_quality)

                if (
                    peak_change > 0.45 and
                    brightness_jump > 25 and
                    quality_peak >= 0.30
                ):
                    is_short_flash = True

            # Long flash pattern check
            if len(track["brightness_history"]) >= 10:
                signal_history = np.array(track["brightness_history"], dtype=np.float32)
                change_history = np.array(track["change_history"], dtype=np.float32)
                flash_history = np.array(track["binary_flash_history"], dtype=np.int32)
                quality_history = np.array(track["source_quality_history"], dtype=np.float32)
                background_history = np.array(track["background_score_history"], dtype=np.float32)

                mean_signal = float(np.mean(signal_history))
                signal_std = float(np.std(signal_history))
                relative_signal_std = signal_std / (mean_signal + 1e-6)

                strong_changes = int(np.sum(
                    change_history > adaptive["flash_change_threshold"]
                ))

                flash_frames = int(np.sum(flash_history))
                flash_transitions = int(np.sum(np.abs(np.diff(flash_history))))

                brightness_range = float(np.max(signal_history) - np.min(signal_history))
                brightness_diff = np.diff(signal_history)

                brightness_transitions = int(np.sum(
                    np.abs(brightness_diff) > 12
                ))

                average_source_quality = float(np.mean(quality_history))
                max_source_quality = float(np.max(quality_history))
                average_background_score = float(np.mean(background_history))
                max_background_score = float(np.max(background_history))

                mostly_static_blue = (
                    signal_std < 8 and
                    flash_transitions < 2 and
                    strong_changes < 3
                )

                moving_blue_object = (
                    strong_changes >= 4 and
                    flash_transitions < 2 and
                    brightness_range < 22
                )

                good_flash_pattern = (
                    flash_frames >= adaptive["required_flash_frames"] and
                    flash_transitions >= adaptive["required_flash_transitions"] and
                    brightness_transitions >= adaptive["required_brightness_transitions"] and
                    brightness_range > 25 and
                    max_source_quality >= adaptive["min_source_quality_for_alert"]
                )

                if average_background_score >= 0.45:
                    good_flash_pattern = (
                        good_flash_pattern and
                        brightness_range > 32 and
                        max_source_quality >= adaptive["min_source_quality_for_alert"] + 0.05
                    )

                if (
                    good_flash_pattern and
                    strong_changes >= 3 and
                    relative_signal_std > 0.15
                ):
                    is_long_flash = True

                if mostly_static_blue:
                    static_penalty = True

                if moving_blue_object:
                    moving_blue_object_penalty = True

                if quality_info["large_object_penalty"]:
                    large_object_penalty = True

                if quality_info["reflection_penalty"]:
                    reflection_penalty = True

                if quality_info["weak_source_penalty"]:
                    weak_source_penalty = True

                if max_background_score >= 0.65:
                    background_blue_penalty = True

                if average_background_score >= 0.45:
                    repeated_background_penalty = True

                if background_cluster_pressure and average_background_score >= 0.35:
                    cluster_background_penalty = True

                if blue_background_pressure and average_source_quality < 0.55:
                    general_background_penalty = True

                if adaptive["scene_mode"] == "night":
                    if not is_long_flash and average_source_quality < 0.50:
                        reflection_penalty = True

            else:
                if background_score >= 0.55:
                    background_blue_penalty = True

                if background_cluster_pressure and background_score >= 0.35:
                    cluster_background_penalty = True

                if blue_background_pressure and light_source_quality < 0.55:
                    general_background_penalty = True

            # ====================================================
            # CONFIRMED FLASH COUNT
            # ====================================================

            if is_long_flash:
                track["confirmed_flash_count"] += 1
            else:
                track["confirmed_flash_count"] = max(
                    0,
                    track["confirmed_flash_count"] - 1
                )

            track["confirmed_flash_count"] = min(
                track["confirmed_flash_count"],
                5
            )

            # ====================================================
            # CONFIDENCE UPDATE
            # ====================================================

            if is_short_flash and is_long_flash:
                track["confidence"] += 2
            elif is_long_flash:
                track["confidence"] += 1
            elif is_short_flash:
                track["confidence"] += 1
            else:
                track["confidence"] -= 2

            if static_penalty:
                track["confidence"] -= 2

            if moving_blue_object_penalty:
                track["confidence"] -= 2

            if large_object_penalty:
                track["confidence"] -= 1

            if reflection_penalty:
                track["confidence"] -= 2

            if weak_source_penalty:
                track["confidence"] -= 1

            if background_blue_penalty:
                track["confidence"] -= 2

            if repeated_background_penalty:
                track["confidence"] -= 1

            if cluster_background_penalty:
                track["confidence"] -= 1

            if general_background_penalty:
                track["confidence"] -= 2

            if background_score >= 0.45 and not is_long_flash:
                track["confidence"] -= 1

            track["confidence"] = self.clamp_confidence(track["confidence"])
            track_confidence_normalised = track["confidence"] / float(TRACK_CONFIDENCE_MAX)

            # ====================================================
            # CANDIDATE SCORING
            # ====================================================

            confirmed_flash_bonus = track["confirmed_flash_count"] * 0.08
            light_quality_bonus = light_source_quality * 0.12

            reflection_score_penalty = 0.20 if reflection_penalty else 0.0
            large_object_score_penalty = 0.10 if large_object_penalty else 0.0
            background_score_penalty = background_score * 0.18

            candidate_score = (
                track_confidence_normalised
                + confirmed_flash_bonus
                + light_quality_bonus
                - reflection_score_penalty
                - large_object_score_penalty
                - background_score_penalty
            )

            # ====================================================
            # VISUAL STATE FOR THIS BLOB
            # ====================================================

            confirmed_enough_for_alert = (
                track["confirmed_flash_count"] >= adaptive["confirmed_flashes_for_alert"]
            )

            source_good_enough = (
                light_source_quality >= adaptive["min_source_quality_for_alert"]
            )

            is_reflection_like = reflection_penalty

            # These stop sky, dashboard reflections, and edge noise
            # from turning into POSSIBLE FLASH or EMERGENCY LIGHT.
            vertical_position = (y + h / 2) / float(FRAME_HEIGHT)

            # Useful road area. Still allows most road/lane/vehicle area,
            # but reduces top sky and bottom dashboard false positives.
            in_alert_zone = (
                vertical_position > 0.22 and
                vertical_position < 0.88
            )

            # Sky/background looking blobs high in the frame should not become alerts.
            sky_false_positive = (
                vertical_position < 0.45 and
                background_score >= 0.45
            )

            # Very weak light sources should not become alerts.
            weak_visual_candidate = (
                light_source_quality < adaptive["min_source_quality_for_alert"]
            )

            is_alert_candidate = (
                track_confidence_normalised >= adaptive["detected_threshold"] and
                confirmed_enough_for_alert and
                source_good_enough and
                not is_reflection_like and
                in_alert_zone and
                not sky_false_positive and
                not weak_visual_candidate and
                background_score < 0.55
            )

            # Amber appears earlier if the object starts behaving flash-like.
            is_possible_candidate = (
                (
                    track_confidence_normalised >= adaptive["possible_threshold"]
                    or is_short_flash
                    or track["confirmed_flash_count"] >= 1
                )
                and source_good_enough
                and not is_reflection_like
                and in_alert_zone
                and not sky_false_positive
                and background_score < 0.65
            )

            if candidate_score > best_track_score:
                best_track_score = candidate_score

                best_track_data = {
                    "track_id": matched_track_id,
                    "bbox": (x, y, w, h),
                    "centre_x": centre_x,
                    "centre_y": centre_y,
                    "position": self.get_position_label(centre_x, FRAME_WIDTH),
                    "confidence": round(track_confidence_normalised, 3),
                    "signal": round(float(brightness_signal), 2),
                    "change_ratio": round(float(change_ratio), 3),
                    "mean_brightness": round(float(mean_brightness), 2),
                    "max_brightness": round(float(max_brightness), 2),
                    "blue_ratio": round(float(blue_ratio), 3),
                    "source_quality": round(float(light_source_quality), 3),
                    "confirmed_flash_count": track["confirmed_flash_count"],
                    "large_object_penalty": large_object_penalty,
                    "reflection_penalty": reflection_penalty,
                    "weak_source_penalty": weak_source_penalty,
                    "background_score": round(float(background_score), 3),
                    "green_ratio": round(float(green_ratio), 3),
                    "is_alert_candidate": is_alert_candidate,
                    "is_possible_candidate": is_possible_candidate
                }

            # ====================================================
            # DRAW BOX
            # ====================================================

            if is_alert_candidate:
                colour = (0, 0, 255)
                label = "EMERGENCY LIGHT"
                thickness = 2

            elif is_possible_candidate:
                colour = (0, 165, 255)
                label = "POSSIBLE FLASH"
                thickness = 2

            else:
                colour = (255, 0, 0)
                label = "BLUE OBJECT"
                thickness = 1

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                colour,
                thickness
            )

            cv2.putText(
                display,
                f"{label} T{matched_track_id}",
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1
            )

            cv2.putText(
                display,
                f"Conf:{track_confidence_normalised:.2f} "
                f"F:{track['confirmed_flash_count']} "
                f"Q:{light_source_quality:.2f} "
                f"Bg:{background_score:.2f}",
                (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1
            )

        # ========================================================
        # REMOVE OLD TRACKS
        # ========================================================

        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if track["missing_frames"] <= MAX_MISSING_FRAMES
        }

        # ========================================================
        # FINAL SYSTEM RESULT
        # ========================================================

        if best_track_data is None:
            visual_confidence = 0.0
            visual_state = "clear"
            best_track_id = None
            best_bbox = None
            best_position = None
            best_source_quality = 0.0
            best_confirmed_flash_count = 0
            best_is_reflection = False
            best_background_score = 0.0

        else:
            visual_confidence = best_track_data["confidence"]
            best_source_quality = best_track_data["source_quality"]
            best_confirmed_flash_count = best_track_data["confirmed_flash_count"]
            best_is_reflection = best_track_data.get("reflection_penalty", False)
            best_background_score = best_track_data.get("background_score", 0.0)

            if best_track_data.get("is_alert_candidate", False):
                visual_state = "detected"

            elif best_track_data.get("is_possible_candidate", False):
                visual_state = "possible"

            else:
                visual_state = "clear"

            best_track_id = best_track_data["track_id"]
            best_bbox = best_track_data["bbox"]
            best_position = best_track_data["position"]

        # Driver alert only happens on confirmed visual detection.
        # Later, fusion can also allow audio + possible to trigger warning.
        driver_alert = visual_state == "detected"

        output_data = {
            "frame_index": self.frame_index,
            "driver_alert": driver_alert,
            "visual_detected": visual_state == "detected",
            "visual_confidence": round(float(visual_confidence), 3),
            "visual_state": visual_state,
            "best_track_id": best_track_id,
            "best_bbox": best_bbox,
            "best_position": best_position,
            "best_source_quality": round(float(best_source_quality), 3),
            "confirmed_flash_count": best_confirmed_flash_count,
            "best_is_reflection": best_is_reflection,
            "best_background_score": round(float(best_background_score), 3),
            "num_active_tracks": len(self.tracks),
            "scene_mode": adaptive["scene_mode"],
            "frame_brightness": round(float(adaptive["frame_brightness"]), 2),
            "blue_background_ratio": round(float(blue_background_ratio), 3),
            "blue_background_pressure": blue_background_pressure,
            "background_cluster_pressure": background_cluster_pressure,
            "small_background_like_blobs": background_cluster_info["small_background_like_blobs"],
            "audio_boost": AUDIO_SIREN_DETECTED,
            "confirmed_flashes_required": adaptive["confirmed_flashes_for_alert"]
        }

        # ========================================================
        # DISPLAY STATUS
        # ========================================================

        if visual_state == "detected":
            status_text = f"SYSTEM STATUS: EMERGENCY DETECTED ({visual_confidence:.2f})"
            status_colour = (0, 0, 255)

        elif visual_state == "possible":
            status_text = f"SYSTEM STATUS: POSSIBLE FLASH CANDIDATE ({visual_confidence:.2f})"
            status_colour = (0, 165, 255)

        else:
            status_text = f"SYSTEM STATUS: CLEAR ({visual_confidence:.2f})"
            status_colour = (0, 255, 0)

        cv2.putText(
            display,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_colour,
            2
        )

        if best_track_data is not None:
            cv2.putText(
                display,
                f"Best T:{best_track_id} Pos:{best_position} "
                f"F:{best_confirmed_flash_count} "
                f"Q:{best_source_quality:.2f} "
                f"Bg:{best_background_score:.2f} "
                f"Ref:{best_is_reflection}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

        debug_text = (
            f"Scene:{adaptive['scene_mode']} "
            f"Bright:{adaptive['frame_brightness']:.1f} "
            f"BlueBg:{blue_background_ratio:.2f} "
            f"BgBlobs:{background_cluster_info['small_background_like_blobs']} "
            f"H:{adaptive['lower_blue'][0]}-{adaptive['upper_blue'][0]} "
            f"S>={adaptive['lower_blue'][1]} "
            f"V>={adaptive['lower_blue'][2]}"
        )

        cv2.putText(
            display,
            debug_text,
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        if AUDIO_SIREN_DETECTED:
            cv2.putText(
                display,
                "AUDIO BOOST: SIREN DETECTED",
                (20, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1
            )

        return output_data, display, blue_mask, blue_change_mask


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN LOOP
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Error: could not open video source: {VIDEO_SOURCE}")
        return

    detector = VisualEmergencyDetector()

    print("Visual emergency detector running...")
    print("Video:", VIDEO_SOURCE)
    print("Audio boost:", AUDIO_SIREN_DETECTED)
    print("Press q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        output_data, display, blue_mask, blue_change_mask = detector.process_frame(frame)

        if PRINT_OUTPUT and output_data["frame_index"] % PRINT_EVERY_N_FRAMES == 0:
            print(output_data)

        cv2.imshow("Detection View", display)
        cv2.imshow("Blue Mask", blue_mask)
        cv2.imshow("Blue Change Mask", blue_change_mask)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()