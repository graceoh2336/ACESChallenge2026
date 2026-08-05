import cv2
import time
from picamera2 import Picamera2

import bestattempt


# ============================================================
# LIVE CAMERA SETTINGS
# ============================================================

# False = normal visual mode
# True  = simulate siren audio, visual becomes more sensitive
bestattempt.AUDIO_SIREN_DETECTED = False

# Reduce terminal spam
bestattempt.PRINT_EVERY_N_FRAMES = 10

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SHOW_DEBUG_WINDOWS = False

# ------------------------------------------------------------
# COLOUR MODE
# ------------------------------------------------------------

COLOUR_MODE = "raw_no_convert" #had issue with a blue filter being on the whole video feed. seems to have fixed it


def main():
    print("Starting live Raspberry Pi camera detector...")
    print("Press q in the video window to quit.")
    print("Press Ctrl+C in the terminal to stop.")
    print(f"Colour mode: {COLOUR_MODE}")

    picam2 = Picamera2()

    # --------------------------------------------------------
    # Camera colour format setup
    # --------------------------------------------------------
    if COLOUR_MODE == "bgr_no_convert":
        camera_format = "BGR888"
    else:
        camera_format = "RGB888"

    camera_config = picam2.create_video_configuration(
        main={
            "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
            "format": camera_format
        }
    )

    picam2.configure(camera_config)

    # Use auto white balance. Do not use manual colour gains for now.
    try:
        picam2.set_controls({
            "AwbEnable": True
        })
    except Exception as e:
        print("Could not set auto white balance:", e)

    picam2.start()

    # Give camera time to settle exposure and white balance
    time.sleep(3)

    detector = bestattempt.VisualEmergencyDetector()

    try:
        while True:
            # --------------------------------------------------------
            # Capture camera frame
            # --------------------------------------------------------
            camera_frame = picam2.capture_array()

            # --------------------------------------------------------
            # Convert frame if needed
            # --------------------------------------------------------
            if COLOUR_MODE == "rgb_to_bgr":
                # Camera gives RGB, OpenCV expects BGR
                frame_bgr = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)

            elif COLOUR_MODE == "raw_no_convert":
                # Use whatever the camera gives directly
                frame_bgr = camera_frame

            elif COLOUR_MODE == "bgr_no_convert":
                # Camera was asked to give BGR directly
                frame_bgr = camera_frame

            else:
                raise ValueError(f"Unknown COLOUR_MODE: {COLOUR_MODE}")

            # --------------------------------------------------------
            # Run your existing detector
            # --------------------------------------------------------
            output_data, display, blue_mask, blue_change_mask = detector.process_frame(frame_bgr)

            if output_data["frame_index"] % bestattempt.PRINT_EVERY_N_FRAMES == 0:
                print(output_data)

            cv2.imshow("Live Detection View", display)

            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Blue Mask", blue_mask)
                cv2.imshow("Blue Change Mask", blue_change_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping live camera detector...")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stopped cleanly.")


if __name__ == "__main__":
    main()
