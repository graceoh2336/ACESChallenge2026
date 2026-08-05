# Visual Emergency Light Detection: Background, Approach, and Limitations

The visual detection part of this project focuses on identifying possible emergency vehicle lights from video footage or a live Raspberry Pi camera feed. The main target is blue flashing emergency lights, as blue is the clearest visual signal used by emergency vehicles in Europe. The purpose of the visual detector is not to make the final driver alert decision by itself. Instead, it is designed to act as an evidence source that can later be combined with siren audio detection in a fusion layer.

The basic idea is that a camera can provide useful visual evidence, while an audio detector can provide siren evidence. By combining both, the system should be able to make a more reliable decision than either system could make alone. For example, a visual detector may see a blue object that looks like a light, but it may not know whether that object is a sign, sky, a reflection, or a real emergency vehicle. Similarly, an audio detector may hear a siren-like sound, but it may not know where the emergency vehicle is. The fusion layer should combine both sources of evidence before sending any driver notification.


## How the Visual Detector Works
The detector works by processing each video frame one at a time. First, each frame is resized to a consistent size so that the system behaves the same across different videos and camera feeds. The frame is then slightly blurred to reduce small pixel-level noise. After this, the frame is converted from normal colour format into HSV colour space.

HSV separates colour, colour strength, and brightness. This makes it easier to isolate blue areas than using normal RGB or BGR colour values. A blue HSV upper and lower bounds are defined. Once the frame is in HSV range, the script creates a blue mask. In this mask, blue or blue-white pixels are kept, while everything else is removed. The white pixels in the mask represent areas that might be blue emergency lights, while black areas are ignored.
After creating the blue mask, the script cleans it using image processing operations. This helps remove tiny random pixels and fill small gaps. The script then finds connected groups of blue pixels, called blobs. These blobs are not automatically treated as emergency lights. They are only treated as possible blue objects at first.
Each blue blob is measured using several features, including size, shape, brightness, how much of the bounding box is actually blue, and whether it looks like a compact light source. The script also tracks blobs across frames. This is important because emergency lights are not only blue, they should also flash or change over time. By tracking a blob, the script can build a history and check whether the object behaves like a flashing light rather than a static blue object.
The detector checks for short flashes and longer flashing patterns. It looks at brightness changes, changes in the blue mask, repeated flash-like frames, and whether there are enough transitions between on/off states. Based on this, the script updates a confidence score and assigns a visual state.

## Visual Output States
The visual detector has three main output states.

### Blue Object
A blue object means the script has found something blue, but it does not yet trust the object as an emergency light. This could be a blue sign, sky patch, reflection, blue vehicle bodywork, or some other blue object in the scene. This state should not trigger a driver alert.

### Possible flash
A possible flash means the object has started to behave more like a flashing emergency light candidate. It may have enough blue change, brightness change, or repeated activity to be worth tracking. This is an amber-level state. It is useful evidence, but it should not trigger a final driver warning by itself.

### Emergency Light
An emergency light means the script has stronger visual evidence. The object has built up enough confidence and flash-like behaviour for the script to mark it as a detected emergency-light candidate. Even then, this should still be treated as visual evidence rather than a final decision. The final alert should come from audio-visual fusion.

## lights.py

lights.py is the more forgiving version of the detector. It is more willing to pick up weaker or more difficult emergency-light candidates. This makes it useful for footage where the lights are small, distant, compressed, partly washed out, or only visible briefly.

- The benefit of lights.py is that it is more likely to notice potential emergency lights in difficult footage. 
- The weakness is that it can produce more false positives. It is more likely to be confused by blue sky, reflections, road glare, signs, or video compression artefacts.

This version is useful as the active visual evidence script because it gives the fusion layer more information to work with. However, its output should be treated carefully and should not be used alone as the final driver notification. I wanted to have a script that could be able to handle edge cases as opposed to needing to define each scenario as I came across it.

## bestattempt.py

bestattempt.py is the stricter version of the detector. It is more cautious and generally better at reducing false positives from background blue objects. It handles scenes like sky, signs, and reflections more carefully.

- The benefit of bestattempt.py is that it is less likely to falsely alert on random blue areas. 
- The weakness is that it may miss real emergency lights in harder situations, especially when the lights are small, fast-moving, overexposed, compressed, or only visible for a short time.

This script is useful as a comparison version. It shows the other side of the trade-off: better false-positive control, but more missed detections. I included this as a stable working version.

## Main Challenges
The main challenge with this visual system is that emergency lights and false positives can look very similar to a camera.

A blue emergency light is not the only thing that appears blue in video footage. The detector can also see blue from:

- daytime sky
- sky through trees
- road reflections
- glass reflections
- blue signs
- blue vehicle paint
- dashcam compression artefacts
- blue-tinted highlights

The goal was to avoid writing a separate hard-coded rule for every possible edge case. For example, it would not be ideal to keep adding a “sky rule”, “sea rule”, “sign rule”, “reflection rule”, and so on forever. Instead, the script tries to judge whether a blue object behaves like a real flashing light source. This is a more flexible approach, but it still has limits.

## Known Weaknesses
The visual detector works as a prototype and evidence source, but it is not perfect. The main weak points are:

### Blue Sky
Blue sky can fall into the same colour range as the blue emergency lights. This can cause false positives, especially in daytime dashcam videos where the sky is large and bright. The detector includes background-blue checks, but sky can still be difficult when it appears as small fragmented blobs through trees or along the top of the frame. Ideally this additional filter would need to be added but it was a major and common issue.

### Reflections and Glare
Reflections from roads, glass, signs, and vehicle surfaces can look like blue flashing lights. This is especially difficult in night scenes or wet road conditions. Reflections can sometimes change between frames, which makes them look more like flashing.

### Compressed Dashcam Footage
Some dashcam or YouTube footage used in the testing phase makes emergency lights harder to detect. Compression can make the lights look blurry, blocky, smeared, or almost white instead of clearly blue. This can cause the detector to miss real emergency lights or only mark them as possible candidates.

### Close or Overexposed Lights
When a light is very close to the camera, it can become too large, too bright, blurry, or washed out. The detector is strongest when the light appears as a compact flashing source. If the light fills too much of the frame, the script may treat it like a large blue object rather than a clear emergency light. This size limit was to restrict the detector detecting big blue objects that could be in the frame, like a blue car right beside the camera. This is a trade off as in the tests the light being right infront of the camera running the detector didnt detect the flashing as it was concidered too big. The hope would be that the whole system would have sent a notification to the driver before the object becomes too big to be dismissed.
As well as that the detector has a weak spot on very sunny days when the siren light is no longer bright and vivid and misses the siren light.

### Visual Flickering
The visual state does not always stay stable. It may flicker between clear, possible, and detected as the light changes shape, brightness, or visibility across the scene. The possible solution is the fusion layer would account for this. The fusion script could include certain limits and criteria to be met by both the audio and visual detectors in order to send a notfication to the driver. 

## Live Camera
live_camera.py is a wrapper script that connects the Raspberry Pi Camera Module to the visual detector. The main detection logic still lives inside lights.py. The live camera script simply captures frames from the Raspberry Pi camera and passes each frame into the detector for processing.

The live camera script allows the project to move beyond saved video files and test the detector in a more realistic real-time setup. This is important because the final system is intended to work from vehicle-mounted cameras rather than only pre-recorded videos.

The live camera setup currently:

- starts the Raspberry Pi camera feed
- captures frames continuously
- passes each frame into lights.py
- displays the detector output live
- keeps running until the user presses `q` or stops the terminal
- allows simulated audio boost through the `AUDIO_SIREN_DETECTED` flag

During testing, the camera colour format needed to be adjusted because the live feed initially had a strong blue tint. The current working setting is:
COLOUR_MODE = "raw_no_convert"

The live camera feed has been tested and is working with the Raspberry Pi camera. However, full real-world validation using the Pi camera has not yet been completed. Current testing has focused on proving that the live camera pipeline works and that frames can be processed by the visual detector in real time.
Further testing is still needed to decide the best demo method and to represent real-world conditions as accurately as possible.

## Summary
The visual side is working as a prototype. It can:

- read saved video files
- run on a Raspberry Pi live camera feed
- isolate blue regions using HSV
- track blue objects over time
- check for flashing behaviour
- classify visual evidence as blue object, possible flash, or emergency detected
- simulate audio boost using a manual flag

However, some edge cases remain. The detector can still produce false positives in scenes with sky, reflections, or blue artefacts. It can also miss emergency lights in low-quality or difficult footage. These weaknesses are important to address.