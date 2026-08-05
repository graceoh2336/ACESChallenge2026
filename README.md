\# ACES OpenCV Emergency Light detection



\## Overview



This folder contains OpenCV-based visual detection scripts for detecting blue emergency-light candidates from video footage and Raspberry Pi camera input.



The visual detector is intended to act as one evidence source in a wider emergency vehicle detection system. It should not be treated as the final decision maker by itself. The final system should combine visual detection with siren audio detection through a fusion layer before sending a driver notification.





\#SCRIPTS

\### `lights.py`



`lights.py` is the more forgiving version of the detector.



It is designed to pick up more potential emergency light cases, including weaker, smaller, or less stable blue flashing lights. This makes it useful for testing difficult footage where emergency lights may be distant, compressed, partly washed out, or only visable for short periods.



The trade-off is that `lights.py` can produce more false positives, especially in scenes with:



\- blue sky

\- reflections

\- dashcam compression video files

\- blue-tinted road or vehicle highlights



\### `bestattempt.py`



`bestattempt.py` is the stricter version of the detector.



Is is more cautious and generally better at avoiding false positives from sky, background blue, and reflections. However, because it is stricter, it may miss emergency lights in harder conditions, such as:



\- fast-moving dashcam footage

\- close or overexposed lights

\- low-quality compressed video

\- blue lights that change shape or size quickly

\- lights that only appear briefly



\### `live\_camera.py`

`live\_camera.py` connects the Raspberry Pi Camera Module to the visual detector.



It captures live frames from the Pi camera and passes them into `lights.py`.\*

For the current Raspberry Pi camera setup, the best colour setting has been:



```python

COLOUR\_MODE = "raw\_no\_convert"

