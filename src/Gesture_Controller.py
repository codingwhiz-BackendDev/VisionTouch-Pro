# Imports

import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ──────────────────────────────────────────────
#  Gesture Encodings
# ──────────────────────────────────────────────
# New Gesture → Action map:
#   PALM              → Neutral   (open palm, do nothing)
#   INDEX             → Move Cursor
#   PINCH_MAJOR       → Left Click  /  Volume  (thumb + index)
#   PINCH_MINOR       → Right Click /  Brightness (thumb + middle)
#   TWO_FINGER_CLOSED → Double Click  (index + middle together)
#   V_GEST            → Scroll        (index + middle spread)
#   FIST              → Drag & Drop   (all fingers closed)
#   THREE_FINGERS     → Multi-select  (index + middle + ring)
# ──────────────────────────────────────────────

class Gest(IntEnum):
    """Enum for all hand gestures (4-bit finger pattern + extras)."""

    # 4-bit encoded: bit3=index | bit2=mid | bit1=ring | bit0=pinky
    FIST              = 0
    PINKY             = 1
    RING              = 2
    MID               = 4
    LAST3             = 7
    INDEX             = 8
    FIRST2            = 12
    THREE_FINGERS_RAW = 14   # index + mid + ring (raw bits)
    LAST4             = 15
    THUMB             = 16
    PALM              = 31   # all fingers open (neutral)

    # Special / composite gestures
    V_GEST            = 33   # index + middle spread  → Scroll
    TWO_FINGER_CLOSED = 34   # index + middle close   → Double Click
    PINCH_MAJOR       = 35   # thumb + index pinch    → Left Click / Volume
    PINCH_MINOR       = 36   # thumb + middle pinch   → Right Click / Brightness
    THREE_FINGERS     = 37   # index + mid + ring     → Multi-select


# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1


# ──────────────────────────────────────────────
#  Hand Recognition
# ──────────────────────────────────────────────
class HandRecog:
    """Convert Mediapipe Landmarks to recognizable Gestures."""

    def __init__(self, hand_label):
        self.finger      = 0        # 4-bit finger state
        self.thumb_open  = False    # True when thumb is extended
        self.ori_gesture = Gest.PALM
        self.prev_gesture= Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label  = hand_label

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        """Signed Euclidean distance; positive when point[0] is above point[1]."""
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist) * sign

    def get_dist(self, point):
        """Euclidean distance between two landmarks."""
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist)

    def get_dz(self, point):
        """Absolute Z-axis difference between two landmarks."""
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)

    def set_finger_state(self):
        """
        Compute self.finger (4-bit) and self.thumb_open.
        Finger bit = 1 if the finger tip is raised above its base knuckle.
        Thumb open = tip.x < mcp.x in the mirrored image.
        """
        if self.hand_result is None:
            return

        # [tip, middle_knuckle, base_knuckle] for index→pinky
        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0

        for point in points:
            dist  = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            try:
                ratio = round(dist / dist2, 1)
            except ZeroDivisionError:
                ratio = round(dist / 0.01, 1)
            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1

        # Thumb: tip(4) left of mcp(2) → open (works for mirrored/right hand)
        self.thumb_open = (
            self.hand_result.landmark[4].x < self.hand_result.landmark[2].x
        )

    def get_gesture(self):
        """
        Return smoothed Gest enum value for this hand.
        A gesture must persist for >4 consecutive frames to be confirmed.
        """
        if self.hand_result is None:
            return Gest.PALM

        current_gesture = Gest.PALM

        # ── Priority 1: PINCH_MAJOR  (thumb + index pinch) ──────────────────
        if self.thumb_open and (self.finger & 8):          # index bit set
            if self.get_dist([4, 8]) < 0.05:
                current_gesture = Gest.PINCH_MAJOR

        # ── Priority 2: PINCH_MINOR  (thumb + middle pinch) ─────────────────
        elif self.thumb_open and (self.finger & 4):        # mid bit set
            if self.get_dist([4, 12]) < 0.05:
                current_gesture = Gest.PINCH_MINOR

        # ── Priority 3: Three fingers (index + middle + ring) ────────────────
        elif (self.finger & 0x0E) == 0x0E:                # bits 1110
            current_gesture = Gest.THREE_FINGERS

        # ── Priority 4: FIRST2  (index + middle both up) ─────────────────────
        elif (self.finger & 0x0C) == 0x0C:                # bits 1100
            dist_tips = self.get_dist([8, 12])
            dist_base = self.get_dist([5, 9])
            try:
                ratio = dist_tips / dist_base
            except ZeroDivisionError:
                ratio = 0

            if ratio > 1.7:
                current_gesture = Gest.V_GEST            # spread  → Scroll
            else:
                if self.get_dz([8, 12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED  # close → Double Click
                else:
                    current_gesture = Gest.MID

        # ── Priority 5: Raw bit patterns ──────────────────────────────────────
        else:
            if self.finger == 15:                         # all 4 fingers up
                current_gesture = Gest.PALM              # neutral
            else:
                current_gesture = self.finger            # includes INDEX(8), FIST(0) …

        # ── Noise smoothing ───────────────────────────────────────────────────
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4:
            self.ori_gesture = current_gesture

        return self.ori_gesture


# ──────────────────────────────────────────────
#  Controller  –  executes system actions
# ──────────────────────────────────────────────
class Controller:
    """
    Translates confirmed gestures into system actions.

    MAJOR hand  →  Move / Left-click / Drag / Scroll / Multi-select / Volume
    MINOR hand  →  Right-click / Brightness
    """

    tx_old            = 0
    ty_old            = 0
    trial             = True
    flag              = False    # set after V_GEST, cleared on click
    grabflag          = False
    pinchmajorflag    = False
    pinchminorflag    = False
    pinchstartxcoord  = None
    pinchstartycoord  = None
    pinchdirectionflag= None
    prevpinchlv       = 0
    pinchlv           = 0
    framecount        = 0
    prev_hand         = None
    pinch_threshold   = 0.3

    # ── Pinch level helpers ───────────────────────────────────────────────────
    def getpinchylv(hand_result):
        """Vertical displacement from pinch start (index tip)."""
        return round((Controller.pinchstartycoord - hand_result.landmark[8].y) * 10, 1)

    def getpinchxlv(hand_result):
        """Horizontal displacement from pinch start (index tip)."""
        return round((hand_result.landmark[8].x - Controller.pinchstartxcoord) * 10, 1)

    # ── System controls ───────────────────────────────────────────────────────
    def changesystembrightness():
        """Adjust display brightness by Controller.pinchlv step."""
        current = sbcontrol.get_brightness(display=0) / 100.0
        current += Controller.pinchlv / 50.0
        current = max(0.0, min(1.0, current))
        sbcontrol.fade_brightness(int(100 * current),
                                  start=sbcontrol.get_brightness(display=0))

    def changesystemvolume():
        """Adjust system volume by Controller.pinchlv step."""
        devices   = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume    = cast(interface, POINTER(IAudioEndpointVolume))
        current   = volume.GetMasterVolumeLevelScalar()
        current  += Controller.pinchlv / 50.0
        current   = max(0.0, min(1.0, current))
        volume.SetMasterVolumeLevelScalar(current, None)

    def scrollVertical():
        """Scroll page up/down."""
        pyautogui.scroll(120 if Controller.pinchlv > 0.0 else -120)

    def scrollHorizontal():
        """Scroll page left/right."""
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if Controller.pinchlv > 0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    # ── Cursor position (with dampening) ─────────────────────────────────────
    def get_position(hand_result):
        """
        Return (x, y) screen position derived from hand landmark 9 (palm centre).
        Small jitter is suppressed; large fast movements pass through.
        """
        point    = 9
        pos      = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy   = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(pos[0] * sx)
        y = int(pos[1] * sy)

        if Controller.prev_hand is None:
            Controller.prev_hand = x, y

        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]
        distsq  = delta_x ** 2 + delta_y ** 2

        Controller.prev_hand = [x, y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** 0.5)
        else:
            ratio = 2.1

        x = x_old + delta_x * ratio
        y = y_old + delta_y * ratio
        return (x, y)

    # ── Pinch gesture initialiser ─────────────────────────────────────────────
    def pinch_control_init(hand_result):
        """Record the starting position of a pinch gesture."""
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv          = 0
        Controller.prevpinchlv      = 0
        Controller.framecount       = 0

    # ── Pinch gesture controller ──────────────────────────────────────────────
    def pinch_control(hand_result, controlHorizontal, controlVertical):
        """
        Determine pinch direction and call the appropriate control callback
        once the hand has held the position for 5 frames.
        """
        if Controller.framecount == 5:
            Controller.framecount = 0
            Controller.pinchlv    = Controller.prevpinchlv

            if Controller.pinchdirectionflag is True:
                controlHorizontal()
            elif Controller.pinchdirectionflag is False:
                controlVertical()

        lvx = Controller.getpinchxlv(hand_result)
        lvy = Controller.getpinchylv(hand_result)

        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount  = 0

        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount  = 0

    # ── Main gesture dispatcher ───────────────────────────────────────────────
    def handle_controls(gesture, hand_result):
        """Map confirmed gesture to the appropriate system action."""
        x, y = None, None

        # Compute cursor position for all non-neutral gestures
        if gesture not in (Gest.PALM,):
            x, y = Controller.get_position(hand_result)

        # ── Release flags when gestures end ──────────────────────────────────
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button='left')

        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        # ── PALM → Neutral (do nothing) ───────────────────────────────────────
        if gesture == Gest.PALM:
            pass

        # ── INDEX → Move Cursor ────────────────────────────────────────────────
        elif gesture == Gest.INDEX:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)

        # ── FIST → Drag & Drop (hold left mouse button + move) ────────────────
        elif gesture == Gest.FIST:
            if not Controller.grabflag:
                Controller.grabflag = True
                pyautogui.mouseDown(button='left')
            pyautogui.moveTo(x, y, duration=0.1)

        # ── V_GEST → Scroll (two fingers spread) ──────────────────────────────
        elif gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)

        # ── TWO_FINGER_CLOSED → Double Click ──────────────────────────────────
        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False

        # ── MID → Left Click (after cursor move with INDEX) ───────────────────
        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False

        # ── THREE_FINGERS → Multi-select (Ctrl + Click) ───────────────────────
        elif gesture == Gest.THREE_FINGERS:
            pyautogui.keyDown('ctrl')
            pyautogui.click(x, y)
            pyautogui.keyUp('ctrl')

        # ── PINCH_MAJOR → Left Click (tap) or Volume Control (sustained) ──────
        #    Thumb + Index pinch on MAJOR hand
        #    • y-axis movement → Volume
        #    • x-axis movement → (reserved / horizontal scroll fallback)
        elif gesture == Gest.PINCH_MAJOR:
            if not Controller.pinchmajorflag:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
                # Immediate left-click on first detection (tap)
                pyautogui.click(button='left')
            else:
                # Sustained pinch: control volume via vertical movement
                Controller.pinch_control(
                    hand_result,
                    Controller.scrollHorizontal,   # horizontal → scroll
                    Controller.changesystemvolume  # vertical   → volume
                )

        # ── PINCH_MINOR → Right Click (tap) or Brightness Control (sustained) ─
        #    Thumb + Middle pinch on MINOR hand
        #    • y-axis movement → Brightness
        elif gesture == Gest.PINCH_MINOR:
            if not Controller.pinchminorflag:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
                # Immediate right-click on first detection (tap)
                pyautogui.click(button='right')
            else:
                # Sustained pinch: control brightness via vertical movement
                Controller.pinch_control(
                    hand_result,
                    Controller.scrollHorizontal,        # horizontal → scroll
                    Controller.changesystembrightness   # vertical   → brightness
                )


# ──────────────────────────────────────────────
#  GestureController  –  main entry point
# ──────────────────────────────────────────────
class GestureController:
    """
    Manages camera capture, obtains landmarks from MediaPipe,
    classifies hands, and routes gestures to Controller.

    Attributes
    ----------
    gc_mode    : 1 = running, 0 = stopped
    cap        : cv2.VideoCapture object
    CAM_HEIGHT : frame height in pixels
    CAM_WIDTH  : frame width in pixels
    hr_major   : HandRecog for dominant hand (default: right)
    hr_minor   : HandRecog for non-dominant hand (default: left)
    dom_hand   : True = right hand is dominant
    """

    gc_mode    = 0
    cap        = None
    CAM_HEIGHT = None
    CAM_WIDTH  = None
    hr_major   = None
    hr_minor   = None
    dom_hand   = True

    def __init__(self):
        GestureController.gc_mode    = 1
        GestureController.cap        = cv2.VideoCapture(0)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH  = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def classify_hands(results):
        """Assign left/right landmarks to hr_major / hr_minor."""
        left, right = None, None
        try:
            d = MessageToDict(results.multi_handedness[0])
            if d['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else:
                left = results.multi_hand_landmarks[0]
        except Exception:
            pass
        try:
            d = MessageToDict(results.multi_handedness[1])
            if d['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else:
                left = results.multi_hand_landmarks[1]
        except Exception:
            pass

        if GestureController.dom_hand:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else:
            GestureController.hr_major = left
            GestureController.hr_minor = right

    def start(self):
        """Main loop: capture → detect → gesture → action → display."""
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, image = GestureController.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Pre-process frame
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    GestureController.classify_hands(results)
                    handmajor.update_hand_result(GestureController.hr_major)
                    handminor.update_hand_result(GestureController.hr_minor)

                    handmajor.set_finger_state()
                    handminor.set_finger_state()

                    # Minor hand PINCH_MINOR takes priority (brightness/right-click)
                    gest_name = handminor.get_gesture()
                    if gest_name == Gest.PINCH_MINOR:
                        Controller.handle_controls(gest_name, handminor.hand_result)
                    else:
                        # Otherwise use major hand for all other controls
                        gest_name = handmajor.get_gesture()
                        Controller.handle_controls(gest_name, handmajor.hand_result)

                    # Draw hand skeleton overlay
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                    # Overlay current gesture name on screen
                    cv2.putText(
                        image,
                        f'Gesture: {gest_name.name if hasattr(gest_name, "name") else gest_name}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    Controller.prev_hand = None

                cv2.imshow('Gesture Controller', image)
                if cv2.waitKey(5) & 0xFF == 13:   # Enter to quit
                    break

        GestureController.cap.release()
        cv2.destroyAllWindows()


# Uncomment to run directly
gc1 = GestureController()
gc1.start()
