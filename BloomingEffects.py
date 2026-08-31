"""
BloomingEffects.py
Real-time hand-gesture-triggered flower blooming & bouquet VFX — Magical Garden Edition.

Wave your hand in front of the webcam to grow flowers: a single bloom in
an open palm, a tight bud in a fist, a trail of blossoms as you point and
move, twin flowers on a peace sign, a sprouting seed on a pinch, and a
full 3D-layered tied bouquet when you bring two open palms together.

Ambient magic runs throughout: drifting petals, floating pollen, sparkling
flower dust, glowing fireflies, bees and butterflies visiting blooms,
morning dew on petals, soft wind sway, gentle sunlight rays, a warm aura
around every active flower, and slow color-shifting petals. Hold any
gesture for a couple of seconds to trigger a rainbow flower burst.

Dependencies: OpenCV, MediaPipe, NumPy
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import colorsys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for hand tracking and all blooming VFX parameters."""

    # Hand tracking
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5

    # Feature toggles — gestures & core
    enable_skeleton: bool = True
    enable_gesture_vfx: bool = True
    enable_falling_petals: bool = True
    enable_butterflies: bool = True

    # Feature toggles — magical ambience
    enable_bees: bool = True
    enable_pollen: bool = True
    enable_sparkle_dust: bool = True
    enable_fireflies: bool = True
    enable_dew: bool = True
    enable_wind: bool = True
    enable_sun_rays: bool = True
    enable_aura: bool = True
    enable_rainbow_burst: bool = True
    enable_color_shift: bool = True
    enable_background_atmosphere: bool = True
    enable_face_vfx: bool = True
    enable_smile_crown: bool = True

    # Animation speeds
    gesture_fade_speed: float = 5.0   # how fast an effect fades in/out with the pose
    bloom_speed: float = 1.6          # how fast petals unfurl once a pose is held
    color_cycle_speed: float = 0.05   # how fast bloom hues drift over time
    wind_strength: float = 1.0        # multiplier on the ambient sway
    rainbow_hold_time: float = 2.2    # seconds a gesture must be held to trigger a burst

    # Smile detection — mouth-width is measured relative to eye-to-eye distance
    # (a stable reference that doesn't change with expression), then compared
    # against a running baseline of *your* resting (non-smiling) ratio, which
    # the detector learns automatically as you use it. 'smile_ratio_low' /
    # 'smile_ratio_high' are how much that ratio must RISE above your own
    # baseline — not absolute values — so the crown works out of the box
    # regardless of face shape or camera distance. If the crown triggers too
    # easily or not easily enough, nudge these two values — the on-screen
    # "Smile" readout (when enable_face_vfx is on) shows your live ratio and
    # learned baseline so you can calibrate by eye.
    smile_ratio_low: float = 0.05
    smile_ratio_high: float = 0.18
    crown_flower_count: int = 9

    # Background atmosphere (dusk-garden tint, vignette, drifting bokeh lights)
    background_vignette_strength: float = 0.35
    background_tint_strength: float = 0.16
    background_bokeh_count: int = 14

    # Soft hand-tracking overlay color (BGR)
    skeleton_color: Tuple[int, int, int] = (200, 255, 210)

    # Shared plant colors (BGR)
    colors: dict = field(default_factory=lambda: {
        'stem_green':    (60, 150, 40),
        'leaf_green':    (50, 160, 60),
        'flower_center': (60, 200, 255),   # warm yellow
        'ribbon_color':  (120, 80, 255),   # pink-red ribbon/bow
    })

    # Random petal color palette (BGR) — a fresh color is picked each time
    # a bloom-type effect activates. Spans the full hue wheel (not just
    # pinks/purples) so consecutive blooms actually look different.
    petal_palette: list = field(default_factory=lambda: [
        (180, 105, 255),   # pink
        (255, 150, 200),   # lavender
        (80, 60, 220),     # crimson red
        (60, 200, 255),    # golden yellow
        (230, 220, 250),   # pale blush white
        (200, 130, 255),   # soft purple
        (140, 200, 255),   # peach
        (60, 130, 255),    # sunset orange
        (90, 220, 90),     # spring green
        (200, 220, 60),    # lime
        (210, 190, 30),    # teal-cyan
        (235, 150, 40),    # sky blue
        (230, 90, 60),     # cobalt blue
        (200, 60, 130),    # deep violet
        (190, 60, 230),    # magenta
        (150, 220, 255),   # apricot
        (100, 60, 200),    # wine red
        (245, 225, 130),   # soft mint
        (255, 200, 90),    # turquoise
        (110, 90, 255),    # coral red
    ])

# ============================================================================
# HAND TRACKING
# ============================================================================

class HandTracker:
    """Handles hand detection using MediaPipe."""

    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
            model_complexity=1
        )

        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                 # Palm base
        ]

    def detect_hands(self, frame: np.ndarray) -> Optional[List[List[Tuple[int, int]]]]:
        """Detect hands and return pixel landmark positions."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        h, w = frame.shape[:2]
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))
            all_landmarks.append(landmarks)

        return all_landmarks


class SkeletonRenderer:
    """Draws a soft, minimal hand-tracking overlay (thin lines, small dots)."""

    def __init__(self, config: Config):
        self.config = config

    def draw(self, frame: np.ndarray, landmarks_list, connections) -> np.ndarray:
        if not landmarks_list or not self.config.enable_skeleton:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        color = self.config.skeleton_color
        for landmarks in landmarks_list:
            for a, b in connections:
                cv2.line(overlay, landmarks[a], landmarks[b], color, 1, cv2.LINE_AA)
            for pt in landmarks:
                cv2.circle(overlay, pt, 2, color, -1, cv2.LINE_AA)
        cv2.addWeighted(frame, 1.0, overlay, 0.5, 0, dst=frame)
        return frame

# ============================================================================
# FACE TRACKING  (for the smile -> flower crown effect)
# ============================================================================

class FaceTracker:
    """Handles face-mesh detection using MediaPipe, mirroring HandTracker."""

    def __init__(self, config: Config):
        self.config = config
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_face(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Detect a single face and return its pixel landmark positions, or None."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        face = results.multi_face_landmarks[0]
        return [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]


class SmileDetector:
    """
    Estimates a smoothed 0-1 'smile score' from face-mesh landmarks.
    Combines mouth-width (normalized by eye-to-eye distance, which stays
    stable across expressions) with how much the mouth corners lift above
    the lip center — the two classic signs of a smile.
    """

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    TOP_LIP = 13
    BOTTOM_LIP = 14

    def __init__(self, config: Config):
        self.config = config
        self.smoothed = 0.0
        self.raw_ratio = 0.0
        # Learned "resting face" width ratio for this user/camera — the
        # absolute mouth/eye ratio varies a lot by face shape and distance
        # from the camera, so we calibrate a baseline instead of comparing
        # against a fixed number.
        self.baseline_ratio: Optional[float] = None

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, dt, face_landmarks: Optional[List[Tuple[int, int]]]) -> float:
        if face_landmarks is None or len(face_landmarks) < 300:
            raw = 0.0
        else:
            eye_dist = self._dist(face_landmarks[self.LEFT_EYE_OUTER],
                                   face_landmarks[self.RIGHT_EYE_OUTER]) + 1e-5
            mouth_w = self._dist(face_landmarks[self.LEFT_MOUTH], face_landmarks[self.RIGHT_MOUTH])
            width_ratio = mouth_w / eye_dist
            self.raw_ratio = width_ratio

            if self.baseline_ratio is None:
                self.baseline_ratio = width_ratio
            # Only drift the baseline toward the current ratio while we're
            # NOT currently smiling, so a held smile can't drag its own
            # baseline up and quietly disable itself.
            if self.smoothed < 0.3:
                self.baseline_ratio += (width_ratio - self.baseline_ratio) * dt * 0.4

            lip_center_y = (face_landmarks[self.TOP_LIP][1] + face_landmarks[self.BOTTOM_LIP][1]) / 2.0
            corner_y = (face_landmarks[self.LEFT_MOUTH][1] + face_landmarks[self.RIGHT_MOUTH][1]) / 2.0
            lift = (lip_center_y - corner_y) / eye_dist  # positive when corners are raised

            ratio_rise = width_ratio - self.baseline_ratio
            width_score = clamp01((ratio_rise - self.config.smile_ratio_low) /
                                   max(1e-4, (self.config.smile_ratio_high - self.config.smile_ratio_low)))
            lift_score = clamp01(lift / 0.12)
            raw = clamp01(0.65 * width_score + 0.35 * lift_score)

        smooth_speed = 6.0
        self.smoothed += (raw - self.smoothed) * dt * smooth_speed
        self.smoothed = clamp01(self.smoothed)
        return self.smoothed


class FlowerCrownEffect:
    """🌼 Flower Crown — smile to grow a ring of small, color-shifting flowers
    arced across the hairline, following your head's tilt as you move."""

    LEFT_TEMPLE = 127
    RIGHT_TEMPLE = 356
    FOREHEAD = 10
    CHIN = 152

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.bloom = 0.0
        self.time = 0.0
        self.geom = None
        self.base_colors = []
        self.was_active = False
        self.hold_time = 0.0
        self.burst_pending = False

    def update(self, dt, face_landmarks, smile_score, wind_angle=0.0):
        self.time += dt
        target = smile_score if face_landmarks is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = clamp01(self.alpha)
        self.bloom += (target - self.bloom) * dt * self.config.bloom_speed
        self.bloom = clamp01(self.bloom)

        active_now = face_landmarks is not None and smile_score > 0.5
        if active_now and not self.was_active:
            n = self.config.crown_flower_count
            self.base_colors = [random.choice(self.config.petal_palette) for _ in range(n)]
        self.was_active = active_now

        self.burst_pending = False
        prev_hold = self.hold_time
        if active_now:
            self.hold_time += dt
            if (self.config.enable_rainbow_burst and
                    int(self.hold_time / self.config.rainbow_hold_time) >
                    int(prev_hold / self.config.rainbow_hold_time)):
                self.burst_pending = True
        else:
            self.hold_time = 0.0

        if face_landmarks is not None and len(face_landmarks) >= 357:
            self._compute_geometry(face_landmarks)
        self._wind = wind_angle

    def _compute_geometry(self, lm):
        forehead = lm[self.FOREHEAD]
        chin = lm[self.CHIN]
        left_t = lm[self.LEFT_TEMPLE]
        right_t = lm[self.RIGHT_TEMPLE]
        face_h = math.hypot(forehead[0] - chin[0], forehead[1] - chin[1]) + 1e-5
        roll = math.atan2(right_t[1] - left_t[1], right_t[0] - left_t[0])
        self.geom = {
            'forehead': forehead, 'left_t': left_t, 'right_t': right_t,
            'face_h': face_h, 'roll': roll,
        }

    @property
    def center(self):
        if self.geom is None:
            return None
        g = self.geom
        cx = (g['left_t'][0] + g['right_t'][0]) / 2
        cy = g['forehead'][1] - g['face_h'] * 0.20
        return (int(cx), int(cy))

    def _color(self, i):
        base = self.base_colors[i % len(self.base_colors)]
        if not self.config.enable_color_shift:
            return base
        return cycle_color(base, self.time + i * 0.5, self.config.color_cycle_speed)

    def draw(self, frame):
        if self.alpha <= 0.01 or self.geom is None or not self.base_colors:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        g = self.geom
        n = len(self.base_colors)
        wind = getattr(self, '_wind', 0.0)

        cx = (g['left_t'][0] + g['right_t'][0]) / 2
        pivot_y = g['forehead'][1] - g['face_h'] * 0.20
        top_y = g['forehead'][1] - g['face_h'] * 0.08
        arc_h = g['face_h'] * 0.10
        flower_size = max(6, g['face_h'] * 0.10)

        if self.config.enable_aura:
            draw_glow(overlay, (int(cx), int(pivot_y)), int(g['face_h'] * 0.85),
                      self._color(0), self.alpha * 0.4, layers=5)

        cos_r, sin_r = math.cos(g['roll']), math.sin(g['roll'])

        def rotate(x, y):
            dx, dy = x - cx, y - pivot_y
            return (int(dx * cos_r - dy * sin_r + cx), int(dx * sin_r + dy * cos_r + pivot_y))

        pts = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            x = g['left_t'][0] + (g['right_t'][0] - g['left_t'][0]) * t
            y = top_y - math.sin(math.pi * t) * arc_h
            pts.append(rotate(x, y))

        for pt in pts:
            leaf_pt = rotate(pt[0], pt[1] + flower_size * 0.5)
            draw_leaf(overlay, leaf_pt, g['roll'] + wind, flower_size * 0.4,
                      self.config.colors['leaf_green'], alpha=self.alpha)

        for i, pt in enumerate(pts):
            color = self._color(i)
            draw_flower(overlay, pt, flower_size, self.bloom, 6, color,
                        self.config.colors['flower_center'], rotation=self.time * 0.4 + i,
                        alpha=self.alpha, dew=self.config.enable_dew)

        cv2.add(frame, overlay, dst=frame)
        return frame


class GestureRecognizer:
    """Classifies simple static hand poses from MediaPipe landmarks."""

    GESTURE_LABELS = {
        'open_palm': '🌸 Full Bloom',
        'fist':      '🌱 Flower Bud',
        'point':     '🌷 Flower Trail',
        'peace':     '🌺 Twin Blossoms',
        'pinch':     '🌿 Seed Sprout',
        'bouquet':   '💐 Flower Bouquet',
    }

    ANNOUNCE_LABELS = {
        'open_palm': 'FULL BLOOM',
        'fist':      'FLOWER BUD',
        'point':     'FLOWER TRAIL',
        'peace':     'TWIN BLOSSOMS',
        'pinch':     'SEED SPROUT',
        'bouquet':   'FLOWER BOUQUET',
    }

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @classmethod
    def classify(cls, landmarks: Optional[List[Tuple[int, int]]]) -> Optional[str]:
        if landmarks is None or len(landmarks) < 21:
            return None

        wrist = landmarks[0]
        dist = cls._dist

        index_ext  = dist(landmarks[8],  wrist) > dist(landmarks[6],  wrist) * 1.05
        middle_ext = dist(landmarks[12], wrist) > dist(landmarks[10], wrist) * 1.05
        ring_ext   = dist(landmarks[16], wrist) > dist(landmarks[14], wrist) * 1.05
        pinky_ext  = dist(landmarks[20], wrist) > dist(landmarks[18], wrist) * 1.05
        thumb_ext  = dist(landmarks[4],  wrist) > dist(landmarks[2],  wrist) * 1.2

        hand_size = max(1.0, dist(landmarks[0], landmarks[9]))
        pinch_dist = dist(landmarks[4], landmarks[8])
        is_pinch = pinch_dist < hand_size * 0.45

        if is_pinch and not middle_ext and not ring_ext:
            return 'pinch'
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
            return 'fist'
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return 'point'
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return 'peace'
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return 'open_palm'

        return None


def _palm_center(landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:
    idxs = [0, 5, 9, 13, 17]
    x = int(sum(landmarks[i][0] for i in idxs) / len(idxs))
    y = int(sum(landmarks[i][1] for i in idxs) / len(idxs))
    return (x, y)


@dataclass
class GestureParticle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: Tuple[int, int, int]
    rot: float = 0.0
    rot_speed: float = 0.0

# ============================================================================
# COLOR HELPERS  (color-changing flowers, rainbow bursts)
# ============================================================================

def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def cycle_color(base_bgr: Tuple[int, int, int], t: float, speed: float = 0.05) -> Tuple[int, int, int]:
    """Slowly drifts a base BGR color's hue over time, keeping its saturation/value."""
    b, g, r = base_bgr
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h = (h + t * speed) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(b2 * 255), int(g2 * 255), int(r2 * 255))


def rainbow_color(t: float) -> Tuple[int, int, int]:
    """A fully saturated rainbow color that cycles with t."""
    h = t % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))

# ============================================================================
# FLOWER DRAWING HELPERS
# ============================================================================

def draw_glow(overlay, center, radius, color, alpha, layers=5):
    """Soft additive radial glow — used for auras, fireflies, and sparkle halos."""
    if alpha <= 0 or radius <= 0:
        return
    cx, cy = int(center[0]), int(center[1])
    for i in range(layers):
        t = i / max(1, layers - 1)
        r = max(1, int(radius * (0.25 + 0.75 * t)))
        layer_alpha = alpha * (1.0 - t) * 0.45
        c = tuple(int(min(255, ch * layer_alpha)) for ch in color)
        cv2.circle(overlay, (cx, cy), r, c, -1, cv2.LINE_AA)


def draw_flower(overlay, center, size, bloom, petal_count, petal_color, center_color,
                 rotation=0.0, alpha=1.0, layered=True, dew=False):
    """A stylized flower with an optional 3D-style back petal ring for depth,
    a bright gradient center, and optional dew-drop highlights.
    'bloom' in [0, 1] controls how far the petals have unfurled from the bud.
    Returns the list of front-petal tip positions (useful for dew placement)."""
    if size <= 0 or alpha <= 0:
        return []
    cx, cy = center
    bloom = clamp01(bloom)
    petal_len = size * bloom
    petal_width = size * 0.55 * bloom

    def petal_ring(ring_rot, scale, brightness, ring_alpha):
        tips = []
        plen = petal_len * scale
        pw = petal_width * scale
        for i in range(petal_count):
            ang = ring_rot + i * (2 * math.pi / petal_count)
            ecx = cx + math.cos(ang) * plen * 0.5
            ecy = cy + math.sin(ang) * plen * 0.5
            axes = (max(1, int(plen * 0.55)), max(1, int(pw * 0.5)))
            angle_deg = math.degrees(ang)
            pc = tuple(int(min(255, ch * brightness * ring_alpha)) for ch in petal_color)
            cv2.ellipse(overlay, (int(ecx), int(ecy)), axes, angle_deg, 0, 360, pc, -1, cv2.LINE_AA)
            outline = tuple(int(ch * brightness * ring_alpha * 0.55) for ch in petal_color)
            cv2.ellipse(overlay, (int(ecx), int(ecy)), axes, angle_deg, 0, 360, outline, 1, cv2.LINE_AA)
            tips.append((int(cx + math.cos(ang) * plen), int(cy + math.sin(ang) * plen)))
        return tips

    # Back ring: offset half a petal-step, larger and darker — gives the flower
    # a layered, 3D-style depth instead of a single flat ring of petals.
    if layered and petal_count >= 5:
        petal_ring(rotation + (math.pi / petal_count), 1.18, 0.60, alpha)

    tips = petal_ring(rotation, 1.0, 1.0, alpha)

    center_r = max(2, int(size * 0.22 * (0.5 + 0.5 * bloom)))
    cc = tuple(int(ch * alpha) for ch in center_color)
    cv2.circle(overlay, (cx, cy), center_r, cc, -1, cv2.LINE_AA)
    highlight = tuple(min(255, int(ch * alpha * 0.9 + 60 * alpha)) for ch in center_color)
    cv2.circle(overlay, (cx, cy), max(1, int(center_r * 0.5)), highlight, -1, cv2.LINE_AA)

    if dew and bloom > 0.6:
        draw_dew_drops(overlay, tips, alpha)

    return tips


def draw_dew_drops(overlay, tips, alpha):
    """Tiny glossy highlight droplets resting on alternating petal tips."""
    for i, pt in enumerate(tips):
        if i % 2 != 0:
            continue
        base = tuple(int(200 * alpha) for _ in range(3))
        cv2.circle(overlay, pt, 3, base, -1, cv2.LINE_AA)
        spark = tuple(int(255 * alpha) for _ in range(3))
        cv2.circle(overlay, (pt[0] - 1, pt[1] - 1), 1, spark, -1, cv2.LINE_AA)


def draw_stem(overlay, base, tip, color, thickness=3, alpha=1.0, curve=15):
    """A gently curved stem from base to tip using a quadratic-curve approximation."""
    steps = 12
    mid = ((base[0] + tip[0]) / 2 + curve, (base[1] + tip[1]) / 2)
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * base[0] + 2 * (1 - t) * t * mid[0] + t ** 2 * tip[0]
        y = (1 - t) ** 2 * base[1] + 2 * (1 - t) * t * mid[1] + t ** 2 * tip[1]
        pts.append((int(x), int(y)))
    # A soft light/dark edge along the stem gives it a rounded, realistic feel.
    dark = tuple(int(ch * alpha * 0.65) for ch in color)
    light = tuple(int(min(255, ch * alpha * 1.15)) for ch in color)
    for i in range(len(pts) - 1):
        cv2.line(overlay, pts[i], pts[i + 1], dark, thickness + 1, cv2.LINE_AA)
    for i in range(len(pts) - 1):
        cv2.line(overlay, pts[i], pts[i + 1], light, max(1, thickness - 1), cv2.LINE_AA)
    return pts


def draw_leaf(overlay, point, angle, size, color, alpha=1.0):
    axes = (max(1, int(size)), max(1, int(size * 0.45)))
    c = tuple(int(ch * alpha) for ch in color)
    cv2.ellipse(overlay, point, axes, math.degrees(angle), 0, 360, c, -1, cv2.LINE_AA)
    vein = tuple(int(ch * alpha * 0.6) for ch in color)
    cv2.ellipse(overlay, point, axes, math.degrees(angle), 0, 360, vein, 1, cv2.LINE_AA)


def draw_ribbon(overlay, center, color, alpha, size=18):
    """An elegant bow: two loops, a center knot, and two flowing tails."""
    if alpha <= 0:
        return
    cx, cy = center
    c = tuple(int(ch * alpha) for ch in color)
    dark = tuple(int(ch * alpha * 0.7) for ch in color)

    cv2.ellipse(overlay, (cx - size, cy), (size, int(size * 0.6)), -20, 0, 360, c, -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (cx - size, cy), (size, int(size * 0.6)), -20, 0, 360, dark, 1, cv2.LINE_AA)
    cv2.ellipse(overlay, (cx + size, cy), (size, int(size * 0.6)), 20, 0, 360, c, -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (cx + size, cy), (size, int(size * 0.6)), 20, 0, 360, dark, 1, cv2.LINE_AA)

    tail1 = np.array([(cx - 4, cy + 6), (cx - 18, cy + 42), (cx - 8, cy + 40), (cx, cy + 10)], dtype=np.int32)
    tail2 = np.array([(cx + 4, cy + 6), (cx + 18, cy + 42), (cx + 8, cy + 40), (cx, cy + 10)], dtype=np.int32)
    cv2.fillConvexPoly(overlay, tail1, dark, cv2.LINE_AA)
    cv2.fillConvexPoly(overlay, tail2, dark, cv2.LINE_AA)

    cv2.circle(overlay, (cx, cy), int(size * 0.45), c, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), int(size * 0.45), dark, 1, cv2.LINE_AA)


def draw_sparkle(overlay, pt, size, color, alpha):
    if alpha <= 0 or size <= 0:
        return
    x, y = int(pt[0]), int(pt[1])
    c = tuple(int(ch * alpha) for ch in color)
    cv2.line(overlay, (x - size, y), (x + size, y), c, 1, cv2.LINE_AA)
    cv2.line(overlay, (x, y - size), (x, y + size), c, 1, cv2.LINE_AA)
    cv2.circle(overlay, (x, y), max(1, size // 3), c, -1, cv2.LINE_AA)


def draw_sun_rays(frame, t, alpha=0.10):
    """A handful of soft, slowly-rotating light-ray wedges from an off-screen sun."""
    h, w = frame.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = int(w * 0.88), int(-h * 0.12)
    n_rays = 6
    length = max(w, h) * 1.5
    for i in range(n_rays):
        ang = t * 0.04 + i * (2 * math.pi / n_rays)
        x2 = cx + math.cos(ang) * length
        y2 = cy + math.sin(ang) * length
        perp = ang + math.pi / 2
        ray_w = 26
        p1 = (int(cx + math.cos(perp) * ray_w), int(cy + math.sin(perp) * ray_w))
        p2 = (int(cx - math.cos(perp) * ray_w), int(cy - math.sin(perp) * ray_w))
        p3 = (int(x2 - math.cos(perp) * ray_w * 3), int(y2 - math.sin(perp) * ray_w * 3))
        p4 = (int(x2 + math.cos(perp) * ray_w * 3), int(y2 + math.sin(perp) * ray_w * 3))
        pts = np.array([p1, p2, p3, p4], dtype=np.int32)
        color = (int(190 * alpha), int(225 * alpha), int(255 * alpha))
        cv2.fillConvexPoly(overlay, pts, color, cv2.LINE_AA)
    cv2.add(frame, overlay, dst=frame)
    return frame

# ============================================================================
# WIND SYSTEM  (shared gentle sway used by every stem/petal/leaf)
# ============================================================================

class WindSystem:
    """A cheap, deterministic breeze made from a couple of summed sine waves."""

    def __init__(self, config: Config):
        self.config = config
        self.phase = random.uniform(0, 10)

    def update(self, dt):
        self.phase += dt

    @property
    def angle(self) -> float:
        if not self.config.enable_wind:
            return 0.0
        return (math.sin(self.phase * 0.6) * 0.10 +
                math.sin(self.phase * 1.7 + 1.3) * 0.05) * self.config.wind_strength

# ============================================================================
# GESTURE-TRIGGERED FLOWER EFFECTS
# ============================================================================

class FullBloomEffect:
    """🌸 Full Bloom — open palm grows one large, color-shifting flower with dew and depth."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.bloom = 0.0
        self.time = 0.0
        self.center: Optional[Tuple[int, int]] = None
        self.base_color = None
        self.rotation_seed = random.uniform(0, 10)
        self.was_active = False
        self.hold_time = 0.0
        self.burst_pending = False

    def update(self, dt, landmarks, wind_angle=0.0):
        self.time += dt
        target = 1.0 if landmarks is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))
        self.bloom += (target - self.bloom) * dt * self.config.bloom_speed
        self.bloom = max(0.0, min(1.0, self.bloom))

        active_now = landmarks is not None
        if active_now and not self.was_active:
            self.base_color = random.choice(self.config.petal_palette)
        self.was_active = active_now

        self.burst_pending = False
        prev_hold = self.hold_time
        if active_now:
            self.hold_time += dt
            if (self.config.enable_rainbow_burst and
                    int(self.hold_time / self.config.rainbow_hold_time) >
                    int(prev_hold / self.config.rainbow_hold_time)):
                self.burst_pending = True
        else:
            self.hold_time = 0.0

        if landmarks is not None:
            self.center = _palm_center(landmarks)
        self._wind = wind_angle

    @property
    def color(self):
        if self.base_color is None:
            return None
        if not self.config.enable_color_shift:
            return self.base_color
        return cycle_color(self.base_color, self.time, self.config.color_cycle_speed)

    def draw(self, frame):
        if self.alpha <= 0.01 or self.center is None or self.base_color is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = self.center
        sway = math.sin(self.time * 1.5 + self.rotation_seed) * 0.15 + getattr(self, '_wind', 0.0)

        if self.config.enable_aura:
            draw_glow(overlay, (cx, cy), 95, self.color, self.alpha, layers=6)

        stem_base = (cx, cy + 70)
        draw_stem(overlay, stem_base, (cx, cy + 18), self.config.colors['stem_green'],
                  thickness=4, alpha=self.alpha)
        draw_leaf(overlay, (cx - 14, cy + 40), math.radians(30) + sway, 14, self.config.colors['leaf_green'], alpha=self.alpha)
        draw_leaf(overlay, (cx + 16, cy + 50), math.radians(-40) + sway, 12, self.config.colors['leaf_green'], alpha=self.alpha)
        draw_flower(overlay, (cx, cy), 60, self.bloom, 7, self.color,
                    self.config.colors['flower_center'], rotation=sway, alpha=self.alpha,
                    dew=self.config.enable_dew)

        cv2.add(frame, overlay, dst=frame)
        return frame


class FlowerBudEffect:
    """🌱 Flower Bud — a closed fist holds a small, gently pulsing bud."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.center: Optional[Tuple[int, int]] = None
        self.base_color = None
        self.was_active = False
        self.hold_time = 0.0
        self.burst_pending = False

    def update(self, dt, landmarks, wind_angle=0.0):
        self.time += dt
        target = 1.0 if landmarks is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))

        active_now = landmarks is not None
        if active_now and not self.was_active:
            self.base_color = random.choice(self.config.petal_palette)
        self.was_active = active_now

        self.burst_pending = False
        prev_hold = self.hold_time
        if active_now:
            self.hold_time += dt
            if (self.config.enable_rainbow_burst and
                    int(self.hold_time / self.config.rainbow_hold_time) >
                    int(prev_hold / self.config.rainbow_hold_time)):
                self.burst_pending = True
        else:
            self.hold_time = 0.0

        if landmarks is not None:
            c = _palm_center(landmarks)
            self.center = (c[0], c[1] - 10)
        self._wind = wind_angle

    @property
    def color(self):
        if self.base_color is None:
            return None
        if not self.config.enable_color_shift:
            return self.base_color
        return cycle_color(self.base_color, self.time, self.config.color_cycle_speed)

    def draw(self, frame):
        if self.alpha <= 0.01 or self.center is None or self.base_color is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = self.center
        pulse = 0.16 + 0.04 * math.sin(self.time * 4.0)
        wind = getattr(self, '_wind', 0.0)

        if self.config.enable_aura:
            draw_glow(overlay, (cx, cy), 55, self.color, self.alpha * 0.8, layers=4)

        for i in range(4):
            ang = i * (math.pi / 2) + 0.4 + wind
            pt = (int(cx + 14 * math.cos(ang)), int(cy + 14 * math.sin(ang)))
            draw_leaf(overlay, pt, ang, 10, self.config.colors['leaf_green'], alpha=self.alpha)
        draw_flower(overlay, (cx, cy), 40, pulse, 6, self.color,
                    self.config.colors['flower_center'], rotation=self.time * 0.3 + wind, alpha=self.alpha)

        cv2.add(frame, overlay, dst=frame)
        return frame


class FlowerTrailEffect:
    """🌷 Flower Trail — pointing and moving plants small blossoms along the path."""

    def __init__(self, config: Config):
        self.config = config
        self.planted = []
        self.last_spawn_pos = None
        self.last_spawn_time = -999.0
        self._t = 0.0

    @property
    def alpha(self):
        return 1.0 if self.planted else 0.0

    def update(self, dt, landmarks, wind_angle=0.0):
        self._t += dt
        if landmarks is not None:
            tip = landmarks[8]
            moved_enough = (self.last_spawn_pos is None or
                            math.hypot(tip[0] - self.last_spawn_pos[0], tip[1] - self.last_spawn_pos[1]) > 28)
            time_ok = (self._t - self.last_spawn_time) > 0.12
            if moved_enough and time_ok:
                self.planted.append({
                    'pos': tip,
                    'base_color': random.choice(self.config.petal_palette),
                    'spawn_time': self._t,
                    'size': random.uniform(22, 34),
                    'rot': random.uniform(0, 2 * math.pi),
                })
                self.last_spawn_pos = tip
                self.last_spawn_time = self._t
        else:
            self.last_spawn_pos = None

        self.planted = [p for p in self.planted if (self._t - p['spawn_time']) < 2.6][-40:]

    def draw(self, frame):
        if not self.planted:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for p in self.planted:
            age = self._t - p['spawn_time']
            bloom_in = min(1.0, age / 0.35)
            fade_out = 1.0
            if age > 1.8:
                fade_out = max(0.0, 1.0 - (age - 1.8) / 0.8)
            a = bloom_in * fade_out
            color = p['base_color']
            if self.config.enable_color_shift:
                color = cycle_color(color, self._t, self.config.color_cycle_speed)
            draw_flower(overlay, p['pos'], p['size'], bloom_in, 6, color,
                        self.config.colors['flower_center'], rotation=p['rot'], alpha=a)
        cv2.add(frame, overlay, dst=frame)
        return frame


class TwinBlossomEffect:
    """🌺 Twin Blossoms — a peace sign grows a flower on each of the two fingertips."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.bloom = 0.0
        self.time = 0.0
        self.pt1: Optional[Tuple[int, int]] = None
        self.pt2: Optional[Tuple[int, int]] = None
        self.base_color1 = None
        self.base_color2 = None
        self.was_active = False
        self.hold_time = 0.0
        self.burst_pending = False

    def update(self, dt, landmarks, wind_angle=0.0):
        self.time += dt
        target = 1.0 if landmarks is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))
        self.bloom += (target - self.bloom) * dt * self.config.bloom_speed
        self.bloom = max(0.0, min(1.0, self.bloom))

        active_now = landmarks is not None
        if active_now and not self.was_active:
            self.base_color1 = random.choice(self.config.petal_palette)
            self.base_color2 = random.choice(self.config.petal_palette)
        self.was_active = active_now

        self.burst_pending = False
        prev_hold = self.hold_time
        if active_now:
            self.hold_time += dt
            if (self.config.enable_rainbow_burst and
                    int(self.hold_time / self.config.rainbow_hold_time) >
                    int(prev_hold / self.config.rainbow_hold_time)):
                self.burst_pending = True
        else:
            self.hold_time = 0.0

        if landmarks is not None:
            self.pt1 = landmarks[8]
            self.pt2 = landmarks[12]
        self._wind = wind_angle

    def _color(self, base):
        if base is None:
            return None
        if not self.config.enable_color_shift:
            return base
        return cycle_color(base, self.time, self.config.color_cycle_speed)

    @property
    def center(self):
        return self.pt1

    def draw(self, frame):
        if self.alpha <= 0.01 or self.pt1 is None or self.base_color1 is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        wind = getattr(self, '_wind', 0.0)
        sway = math.sin(self.time * 2.0) * 0.2 + wind
        c1, c2 = self._color(self.base_color1), self._color(self.base_color2)
        if self.config.enable_aura:
            draw_glow(overlay, self.pt1, 45, c1, self.alpha * 0.7, layers=4)
            draw_glow(overlay, self.pt2, 45, c2, self.alpha * 0.7, layers=4)
        draw_flower(overlay, self.pt1, 32, self.bloom, 6, c1,
                    self.config.colors['flower_center'], rotation=sway, alpha=self.alpha)
        draw_flower(overlay, self.pt2, 32, self.bloom, 6, c2,
                    self.config.colors['flower_center'], rotation=-sway, alpha=self.alpha)
        cv2.add(frame, overlay, dst=frame)
        return frame


class SeedSproutEffect:
    """🌿 Seed Sprout — a pinch grows a small green sprout with a tiny opening bud."""

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.time = 0.0
        self.grow = 0.0
        self.base: Optional[Tuple[int, int]] = None
        self.base_color = None
        self.was_active = False
        self.hold_time = 0.0
        self.burst_pending = False

    def update(self, dt, landmarks, wind_angle=0.0):
        self.time += dt
        target = 1.0 if landmarks is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))
        self.grow += (target - self.grow) * dt * 1.2
        self.grow = max(0.0, min(1.0, self.grow))

        active_now = landmarks is not None
        if active_now and not self.was_active:
            self.base_color = random.choice(self.config.petal_palette)
        self.was_active = active_now

        self.burst_pending = False
        prev_hold = self.hold_time
        if active_now:
            self.hold_time += dt
            if (self.config.enable_rainbow_burst and
                    int(self.hold_time / self.config.rainbow_hold_time) >
                    int(prev_hold / self.config.rainbow_hold_time)):
                self.burst_pending = True
        else:
            self.hold_time = 0.0

        if landmarks is not None:
            self.base = ((landmarks[4][0] + landmarks[8][0]) // 2, (landmarks[4][1] + landmarks[8][1]) // 2)
        self._wind = wind_angle

    @property
    def color(self):
        if self.base_color is None:
            return None
        if not self.config.enable_color_shift:
            return self.base_color
        return cycle_color(self.base_color, self.time, self.config.color_cycle_speed)

    @property
    def center(self):
        return self.base

    def draw(self, frame):
        if self.alpha <= 0.01 or self.base is None or self.base_color is None:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        bx, by = self.base
        wind = getattr(self, '_wind', 0.0)
        stem_h = int(55 * self.grow)
        top = (bx, by - stem_h)

        draw_stem(overlay, (bx, by), top, self.config.colors['stem_green'],
                  thickness=3, alpha=self.alpha, curve=6)
        draw_leaf(overlay, (bx - 6, by - int(stem_h * 0.5)), math.radians(25) + wind, 8,
                  self.config.colors['leaf_green'], alpha=self.alpha)
        draw_leaf(overlay, (bx + 7, by - int(stem_h * 0.6)), math.radians(-30) + wind, 8,
                  self.config.colors['leaf_green'], alpha=self.alpha)
        bud_bloom = max(0.0, min(0.4, (self.grow - 0.6) * 1.0)) + 0.05
        draw_flower(overlay, top, 22, bud_bloom, 5, self.color,
                    self.config.colors['flower_center'], rotation=self.time * 0.4 + wind, alpha=self.alpha)

        cv2.add(frame, overlay, dst=frame)
        return frame


class BouquetEffect:
    """
    💐 Flower Bouquet — the finale. Bring both open palms close together to
    form a full 3D-layered fan of tied flowers, complete with realistic
    curved stems, leaves, an elegant ribbon bow, morning dew, and a soft
    shower of falling petals.
    """

    def __init__(self, config: Config):
        self.config = config
        self.alpha = 0.0
        self.bloom = 0.0
        self.time = 0.0
        self.center: Optional[Tuple[int, int]] = None
        self.base_colors = []
        self.depths = []       # per-flower depth factor for a layered, 3D fan
        self.was_active = False
        self.fall_particles: List[GestureParticle] = []
        self.hold_time = 0.0
        self.burst_pending = False

    def update(self, dt, anchor, wind_angle=0.0):
        self.time += dt
        target = 1.0 if anchor is not None else 0.0
        self.alpha += (target - self.alpha) * dt * self.config.gesture_fade_speed
        self.alpha = max(0.0, min(1.0, self.alpha))
        self.bloom += (target - self.bloom) * dt * self.config.bloom_speed
        self.bloom = max(0.0, min(1.0, self.bloom))

        active_now = anchor is not None
        if active_now and not self.was_active:
            self.base_colors = [random.choice(self.config.petal_palette) for _ in range(7)]
            # depth values spread the fan back-to-front for a layered look
            self.depths = [abs(i - (len(self.base_colors) - 1) / 2) for i in range(len(self.base_colors))]
        self.was_active = active_now

        self.burst_pending = False
        prev_hold = self.hold_time
        if active_now:
            self.hold_time += dt
            if (self.config.enable_rainbow_burst and
                    int(self.hold_time / self.config.rainbow_hold_time) >
                    int(prev_hold / self.config.rainbow_hold_time)):
                self.burst_pending = True
        else:
            self.hold_time = 0.0

        if anchor is not None:
            self.center = anchor
        self._wind = wind_angle

        if self.center is not None and self.alpha > 0.5 and self.base_colors and random.random() < 0.18:
            cx, cy = self.center
            self.fall_particles.append(GestureParticle(
                x=cx + random.uniform(-45, 45), y=cy - 20,
                vx=random.uniform(-8, 8), vy=random.uniform(20, 45),
                life=0.0, max_life=random.uniform(1.0, 1.8),
                size=random.uniform(5, 9),
                color=random.choice(self.base_colors),
                rot=random.uniform(0, 2 * math.pi), rot_speed=random.uniform(-2, 2),
            ))

        alive = []
        for p in self.fall_particles:
            p.life += dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.rot += p.rot_speed * dt
            if p.life < p.max_life:
                alive.append(p)
        self.fall_particles = alive[-60:]

    def _color(self, i):
        base = self.base_colors[i]
        if not self.config.enable_color_shift:
            return base
        return cycle_color(base, self.time + i * 0.6, self.config.color_cycle_speed)

    def draw(self, frame):
        if self.alpha <= 0.01 or self.center is None or not self.base_colors:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = self.center
        n = len(self.base_colors)
        fan_spread = math.radians(78)
        wind = getattr(self, '_wind', 0.0)

        if self.config.enable_aura:
            avg_color = self._color(n // 2)
            draw_glow(overlay, (cx, cy - 20), 130, avg_color, self.alpha * 0.6, layers=7)

        ribbon_pt = (cx, cy + 50)

        # Build each stem/flower's target position first so we can sort by
        # depth and draw back-to-front — this is what sells the 3D fan look.
        stems = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            ang = -math.pi / 2 - fan_spread / 2 + fan_spread * t + wind * 0.6
            depth = self.depths[i] if i < len(self.depths) else 0
            dist = 55 + depth * 4
            fx = int(cx + math.cos(ang) * dist)
            fy = int(cy + math.sin(ang) * dist * 0.6 - depth * 6)
            size = (30 + 6 * math.sin(i * 1.7)) * (1.0 - depth * 0.06)
            stems.append((depth, i, ang, (fx, fy), size))

        # Farthest (largest depth value = center-most, drawn behind) first.
        stems.sort(key=lambda s: -s[0])

        for depth, i, ang, (fx, fy), size in stems:
            draw_stem(overlay, ribbon_pt, (fx, fy + 14), self.config.colors['stem_green'],
                      thickness=3, alpha=self.alpha, curve=(fx - cx) * 0.3)
            draw_leaf(overlay, ((fx + ribbon_pt[0]) // 2 - 6, (fy + ribbon_pt[1]) // 2),
                      ang + math.pi / 2 + 0.3, 9, self.config.colors['leaf_green'], alpha=self.alpha)
            color = self._color(i)
            draw_flower(overlay, (fx, fy), size, self.bloom, 6, color,
                        self.config.colors['flower_center'], rotation=ang + math.pi / 2, alpha=self.alpha,
                        dew=self.config.enable_dew)

        draw_ribbon(overlay, ribbon_pt, self.config.colors['ribbon_color'], self.alpha, size=16)

        for p in self.fall_particles:
            t = 1.0 - (p.life / p.max_life)
            c = tuple(int(ch * self.alpha * t) for ch in p.color)
            axes = (max(1, int(p.size)), max(1, int(p.size * 0.5)))
            cv2.ellipse(overlay, (int(p.x), int(p.y)), axes, math.degrees(p.rot), 0, 360, c, -1, cv2.LINE_AA)

        cv2.add(frame, overlay, dst=frame)
        return frame

# ============================================================================
# AMBIENT EFFECTS
# ============================================================================

class BackgroundAtmosphereEffect:
    """
    Turns the plain webcam feed into a dreamy magical-garden backdrop:
    a soft dusk-to-gold color wash, a gentle vignette that draws focus to
    the center, and a handful of large, soft, slow-drifting bokeh lights
    in varied hues (like out-of-focus fairy lights or distant blossoms).
    Cheap to run — the gradient and vignette masks are cached per frame size.
    """

    def __init__(self, config: Config):
        self.config = config
        self.bokeh = []
        self._vignette = None
        self._tint = None
        self._cache_size = None

    def _build_cache(self, w, h):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt(((xx - cx) / (w * 0.62)) ** 2 + ((yy - cy) / (h * 0.62)) ** 2)
        vig = np.clip(1.0 - dist, 0.0, 1.0) ** 1.4
        self._vignette = vig

        # Soft top-to-bottom color wash: dusk violet-blue up top fading to a
        # warm golden-green near the "ground" — gives the frame a garden-at-
        # golden-hour feel without hiding the person in frame.
        top = np.array([70, 30, 40], dtype=np.float32)     # BGR: deep dusk blue-violet
        bottom = np.array([40, 90, 70], dtype=np.float32)  # BGR: warm mossy gold-green
        row_t = (np.arange(h, dtype=np.float32) / max(1, h - 1))[:, None]
        grad = top[None, None, :] * (1 - row_t)[:, :, None] + bottom[None, None, :] * row_t[:, :, None]
        self._tint = np.repeat(grad, w, axis=1).astype(np.float32)
        self._cache_size = (w, h)

    def _ensure_cache(self, w, h):
        if self._cache_size != (w, h):
            self._build_cache(w, h)

    def update(self, dt, w, h):
        if len(self.bokeh) < self.config.background_bokeh_count and random.random() < 0.03:
            edge = random.choice(['left', 'right', 'top', 'bottom'])
            if edge == 'left':
                x, y = -40, random.uniform(0, h)
            elif edge == 'right':
                x, y = w + 40, random.uniform(0, h)
            elif edge == 'top':
                x, y = random.uniform(0, w), -40
            else:
                x, y = random.uniform(0, w), h + 40
            ang = random.uniform(0, 2 * math.pi)
            self.bokeh.append({
                'x': x, 'y': y,
                'vx': math.cos(ang) * random.uniform(4, 10),
                'vy': math.sin(ang) * random.uniform(4, 10),
                'r': random.uniform(35, 70),
                'color': random.choice(self.config.petal_palette),
                'phase': random.uniform(0, 2 * math.pi),
                'life': 0.0, 'max_life': random.uniform(14, 22),
            })

        alive = []
        for b in self.bokeh:
            b['life'] += dt
            b['phase'] += dt * 0.5
            b['x'] += b['vx'] * dt
            b['y'] += b['vy'] * dt
            if b['life'] < b['max_life']:
                alive.append(b)
        self.bokeh = alive

    def draw(self, frame):
        h, w = frame.shape[:2]
        self._ensure_cache(w, h)

        if self.config.background_tint_strength > 0:
            strength = self.config.background_tint_strength
            frame_f = frame.astype(np.float32)
            frame_f = frame_f * (1 - strength) + self._tint * strength
            frame[:] = np.clip(frame_f, 0, 255).astype(np.uint8)

        if self.bokeh:
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            for b in self.bokeh:
                fade = min(1.0, b['life'] / 1.5) * min(1.0, (b['max_life'] - b['life']) / 1.5)
                glow = 0.5 + 0.5 * math.sin(b['phase'])
                draw_glow(overlay, (b['x'], b['y']), b['r'], b['color'], fade * glow * 0.35, layers=6)
            cv2.add(frame, overlay, dst=frame)

        if self.config.background_vignette_strength > 0:
            strength = self.config.background_vignette_strength
            factor = (1.0 - strength) + strength * self._vignette
            frame_f = frame.astype(np.float32) * factor[:, :, None]
            frame[:] = np.clip(frame_f, 0, 255).astype(np.uint8)

        return frame


class FallingPetalsEffect:
    """A gentle, continuous drift of petals falling across the whole frame."""

    def __init__(self, config: Config):
        self.config = config
        self.petals: List[GestureParticle] = []
        self.spawn_timer = 0.0
        self._t = 0.0

    def update(self, dt, frame_w, frame_h, wind_angle=0.0):
        self._t += dt
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.spawn_timer = random.uniform(0.25, 0.6)
            self.petals.append(GestureParticle(
                x=random.uniform(0, frame_w), y=-10,
                vx=random.uniform(-10, 10), vy=random.uniform(30, 55),
                life=0.0, max_life=random.uniform(6, 9),
                size=random.uniform(5, 10),
                color=random.choice(self.config.petal_palette),
                rot=random.uniform(0, 2 * math.pi), rot_speed=random.uniform(-1.5, 1.5),
            ))

        alive = []
        wind_push = wind_angle * 40
        for p in self.petals:
            p.life += dt
            p.x += (math.sin(p.life * 1.5) * 12 + wind_push) * dt + p.vx * dt * 0.1
            p.y += p.vy * dt
            p.rot += p.rot_speed * dt
            if p.life < p.max_life and p.y < frame_h + 20:
                alive.append(p)
        self.petals = alive[-80:]

    def draw(self, frame):
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for p in self.petals:
            fade = 1.0
            if p.life > p.max_life - 1.0:
                fade = max(0.0, p.max_life - p.life)
            color = p.color
            if self.config.enable_color_shift:
                color = cycle_color(color, self._t + p.x * 0.01, self.config.color_cycle_speed * 0.5)
            axes = (max(1, int(p.size)), max(1, int(p.size * 0.5)))
            c = tuple(int(ch * 0.8 * fade) for ch in color)
            cv2.ellipse(overlay, (int(p.x), int(p.y)), axes, math.degrees(p.rot), 0, 360, c, -1, cv2.LINE_AA)
        cv2.add(frame, overlay, dst=frame)
        return frame


class PollenEffect:
    """Fine, glowing pollen motes drifting up and away from active blooms."""

    def __init__(self, config: Config):
        self.config = config
        self.motes = []
        self.timer = 0.0

    def update(self, dt, active_centers):
        self.timer -= dt
        if active_centers and self.timer <= 0 and len(self.motes) < 45:
            self.timer = random.uniform(0.04, 0.12)
            cx, cy = random.choice(active_centers)
            self.motes.append({
                'x': cx + random.uniform(-30, 30), 'y': cy + random.uniform(-20, 20),
                'vx': random.uniform(-6, 6), 'vy': random.uniform(-20, -6),
                'life': 0.0, 'max_life': random.uniform(1.5, 2.8),
                'phase': random.uniform(0, 2 * math.pi),
            })

        alive = []
        for m in self.motes:
            m['life'] += dt
            m['phase'] += dt * 3.0
            m['x'] += (m['vx'] + math.sin(m['phase']) * 6) * dt
            m['y'] += m['vy'] * dt
            if m['life'] < m['max_life']:
                alive.append(m)
        self.motes = alive

    def draw(self, frame):
        if not self.motes:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for m in self.motes:
            fade = 1.0 - m['life'] / m['max_life']
            glow = 0.5 + 0.5 * math.sin(m['phase'] * 2.0)
            c = (int(140 * fade * glow), int(220 * fade * glow), int(255 * fade * glow))
            cv2.circle(overlay, (int(m['x']), int(m['y'])), 2, c, -1, cv2.LINE_AA)
        cv2.add(frame, overlay, dst=frame)
        return frame


class SparkleDustEffect:
    """Twinkling sparkle motes scattered around active blooms — magical flower dust."""

    def __init__(self, config: Config):
        self.config = config
        self.sparks = []
        self.timer = 0.0

    def update(self, dt, active_centers):
        self.timer -= dt
        if active_centers and self.timer <= 0 and len(self.sparks) < 55:
            self.timer = random.uniform(0.03, 0.08)
            cx, cy = random.choice(active_centers)
            ang = random.uniform(0, 2 * math.pi)
            r = random.uniform(10, 60)
            self.sparks.append({
                'x': cx + math.cos(ang) * r, 'y': cy + math.sin(ang) * r,
                'life': 0.0, 'max_life': random.uniform(0.5, 1.0),
                'size': random.uniform(2, 4), 'phase': random.uniform(0, 2 * math.pi),
            })

        alive = []
        for s in self.sparks:
            s['life'] += dt
            s['phase'] += dt * 10.0
            if s['life'] < s['max_life']:
                alive.append(s)
        self.sparks = alive

    def draw(self, frame):
        if not self.sparks:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for s in self.sparks:
            fade = 1.0 - s['life'] / s['max_life']
            twinkle = 0.4 + 0.6 * abs(math.sin(s['phase']))
            draw_sparkle(overlay, (s['x'], s['y']), int(s['size']), (255, 255, 240), fade * twinkle)
        cv2.add(frame, overlay, dst=frame)
        return frame


class FireflyEffect:
    """A handful of softly glowing fireflies wandering the whole scene."""

    def __init__(self, config: Config, count=6):
        self.config = config
        self.flies = [self._spawn() for _ in range(count)]

    def _spawn(self):
        return {
            'x': random.uniform(0, 640), 'y': random.uniform(0, 480),
            'phase': random.uniform(0, 2 * math.pi),
            'dir': random.uniform(0, 2 * math.pi),
        }

    def update(self, dt, frame_w, frame_h):
        for f in self.flies:
            f['phase'] += dt * 2.0
            f['dir'] += random.uniform(-0.6, 0.6) * dt
            f['x'] += math.cos(f['dir']) * 22 * dt
            f['y'] += math.sin(f['dir']) * 22 * dt
            if frame_w > 0:
                f['x'] %= frame_w
            if frame_h > 0:
                f['y'] %= frame_h

    def draw(self, frame):
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for f in self.flies:
            glow = 0.35 + 0.65 * max(0.0, math.sin(f['phase']))
            pos = (int(f['x']), int(f['y']))
            draw_glow(overlay, pos, 11, (90, 255, 220), glow, layers=3)
            cv2.circle(overlay, pos, 2, (190, 255, 255), -1, cv2.LINE_AA)
        cv2.add(frame, overlay, dst=frame)
        return frame


class BeeEffect:
    """Small striped bees that buzz in tight, fast orbits around active blooms."""

    def __init__(self, config: Config):
        self.config = config
        self.bees = []
        self.spawn_cooldown = 1.5

    def update(self, dt, active_center):
        self.spawn_cooldown -= dt
        if (active_center is not None and self.spawn_cooldown <= 0 and
                len(self.bees) < 2 and random.random() < 0.4):
            ang = random.uniform(0, 2 * math.pi)
            self.bees.append({
                'x': active_center[0] + math.cos(ang) * 70,
                'y': active_center[1] + math.sin(ang) * 70,
                'life': 0.0, 'max_life': random.uniform(3.0, 4.5),
                'wander': random.uniform(0, 2 * math.pi),
            })
            self.spawn_cooldown = random.uniform(3.5, 6.0)

        alive = []
        for b in self.bees:
            b['life'] += dt
            b['wander'] += dt * 3.5
            tx, ty = active_center if active_center else (b['x'], b['y'])
            dx, dy = tx - b['x'], ty - b['y']
            dist = math.hypot(dx, dy) + 1e-5
            orbit_x = math.cos(b['wander']) * 34
            orbit_y = math.sin(b['wander']) * 26
            b['x'] += (dx / dist * 30 + orbit_x * 0.6) * dt
            b['y'] += (dy / dist * 30 + orbit_y * 0.6) * dt
            if b['life'] < b['max_life']:
                alive.append(b)
        self.bees = alive

    def draw(self, frame):
        if not self.bees:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for b in self.bees:
            flap = math.sin(b['life'] * 22.0)
            wing_w = 7 + 3 * flap
            bx, by = int(b['x']), int(b['y'])
            cv2.ellipse(overlay, (bx - 4, by - 3), (int(wing_w), 5), 10, 0, 360, (235, 245, 245), -1, cv2.LINE_AA)
            cv2.ellipse(overlay, (bx + 4, by - 3), (int(wing_w), 5), -10, 0, 360, (235, 245, 245), -1, cv2.LINE_AA)
            cv2.ellipse(overlay, (bx, by), (6, 4), 0, 0, 360, (30, 190, 250), -1, cv2.LINE_AA)
            for k, sx in enumerate((-2, 0, 2)):
                cv2.line(overlay, (bx + sx, by - 3), (bx + sx, by + 3), (20, 20, 20), 1, cv2.LINE_AA)
        cv2.add(frame, overlay, dst=frame)
        return frame


class ButterflyEffect:
    """Occasionally spawns a fluttering butterfly that visits — and can land on — an active bloom."""

    def __init__(self, config: Config):
        self.config = config
        self.butterflies = []
        self.spawn_cooldown = 2.0

    def update(self, dt, active_center):
        self.spawn_cooldown -= dt
        if (active_center is not None and self.spawn_cooldown <= 0 and
                len(self.butterflies) < 2 and random.random() < 0.4):
            ang = random.uniform(0, 2 * math.pi)
            self.butterflies.append({
                'x': active_center[0] + math.cos(ang) * 80,
                'y': active_center[1] + math.sin(ang) * 80,
                'life': 0.0, 'max_life': random.uniform(4.0, 6.5),
                'color': random.choice(self.config.petal_palette),
                'wander': random.uniform(0, 2 * math.pi),
                'landed': False,
                'land_timer': random.uniform(1.5, 3.0),
            })
            self.spawn_cooldown = random.uniform(4.0, 7.0)

        alive = []
        for b in self.butterflies:
            b['life'] += dt
            tx, ty = active_center if active_center else (b['x'], b['y'])
            dist = math.hypot(tx - b['x'], ty - b['y'])

            if not b['landed']:
                b['land_timer'] -= dt
                b['wander'] += dt * 1.3
                dx, dy = tx - b['x'], ty - b['y']
                d = math.hypot(dx, dy) + 1e-5
                orbit_x = math.cos(b['wander']) * 60
                orbit_y = math.sin(b['wander']) * 40
                b['x'] += (dx / d * 20 + orbit_x * 0.4) * dt
                b['y'] += (dy / d * 20 + orbit_y * 0.4) * dt
                if b['land_timer'] <= 0 and dist < 45 and active_center:
                    b['landed'] = True
                    b['land_timer'] = random.uniform(1.0, 2.0)
            else:
                b['land_timer'] -= dt
                b['x'], b['y'] = tx, ty
                if b['land_timer'] <= 0:
                    b['landed'] = False
                    b['wander'] = random.uniform(0, 2 * math.pi)

            if b['life'] < b['max_life']:
                alive.append(b)
        self.butterflies = alive

    def draw(self, frame):
        if not self.butterflies:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for b in self.butterflies:
            flap = 0.15 if b['landed'] else math.sin(b['life'] * 14.0)
            wing_w = 10 + 4 * flap
            c = tuple(int(ch) for ch in b['color'])
            bx, by = int(b['x']), int(b['y'])
            cv2.ellipse(overlay, (bx - 6, by), (int(wing_w), 7), 20, 0, 360, c, -1, cv2.LINE_AA)
            cv2.ellipse(overlay, (bx + 6, by), (int(wing_w), 7), -20, 0, 360, c, -1, cv2.LINE_AA)
            cv2.line(overlay, (bx, by - 6), (bx, by + 6), (40, 40, 40), 2, cv2.LINE_AA)
        cv2.add(frame, overlay, dst=frame)
        return frame


class RainbowBurstEffect:
    """A one-shot radial burst of rainbow petals — triggered when a gesture is held."""

    def __init__(self, config: Config):
        self.config = config
        self.particles: List[GestureParticle] = []
        self._t = 0.0

    def trigger(self, center):
        cx, cy = center
        for _ in range(42):
            ang = random.uniform(0, 2 * math.pi)
            speed = random.uniform(70, 170)
            self.particles.append(GestureParticle(
                x=cx, y=cy, vx=math.cos(ang) * speed, vy=math.sin(ang) * speed,
                life=0.0, max_life=random.uniform(0.6, 1.1),
                size=random.uniform(3, 6), color=(255, 255, 255),
                rot=random.uniform(0, 2 * math.pi), rot_speed=random.uniform(-4, 4),
            ))

    def update(self, dt):
        self._t += dt
        alive = []
        for p in self.particles:
            p.life += dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += 50 * dt
            p.rot += p.rot_speed * dt
            if p.life < p.max_life:
                alive.append(p)
        self.particles = alive

    def draw(self, frame):
        if not self.particles:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for i, p in enumerate(self.particles):
            fade = 1.0 - p.life / p.max_life
            color = rainbow_color(self._t * 0.25 + i * 0.04)
            c = tuple(int(ch * fade) for ch in color)
            axes = (max(1, int(p.size)), max(1, int(p.size * 0.6)))
            cv2.ellipse(overlay, (int(p.x), int(p.y)), axes, math.degrees(p.rot), 0, 360, c, -1, cv2.LINE_AA)
        cv2.add(frame, overlay, dst=frame)
        return frame

# ============================================================================
# ANNOUNCEMENT BANNER
# ============================================================================

class GestureAnnouncer:
    """Slide-in title banner shown briefly whenever a new bloom effect activates."""

    def __init__(self, config: Config):
        self.config = config
        self.last_active_name: Optional[str] = None
        self.text = ""
        self.color: Tuple[int, int, int] = (255, 255, 255)
        self.state = 'idle'
        self.state_time = 0.0
        self.slide_duration = 0.28
        self.hold_duration = 1.1

    def update(self, dt, active_name, text, color):
        self.state_time += dt

        if active_name and active_name != self.last_active_name:
            self.text = text
            self.color = color
            self.state = 'in'
            self.state_time = 0.0

        self.last_active_name = active_name

        if self.state == 'in' and self.state_time >= self.slide_duration:
            self.state = 'hold'
            self.state_time = 0.0
        elif self.state == 'hold' and self.state_time >= self.hold_duration:
            self.state = 'out'
            self.state_time = 0.0
        elif self.state == 'out' and self.state_time >= self.slide_duration:
            self.state = 'idle'
            self.state_time = 0.0

    def draw(self, frame):
        if self.state == 'idle' or not self.text:
            return frame
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        if self.state == 'in':
            t = self.state_time / self.slide_duration
            ease = 1 - (1 - t) ** 3
        elif self.state == 'hold':
            ease = 1.0
        else:
            t = self.state_time / self.slide_duration
            ease = 1.0 - t ** 3
        ease = max(0.0, min(1.0, ease))

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.8
        thickness = 2
        (tw, th), _ = cv2.getTextSize(self.text, font, scale, thickness)
        pad = 20
        box_w = tw + pad * 2
        target_x = w - box_w - 24
        start_x = w + 10
        x = int(start_x + (target_x - start_x) * ease)
        y = 96

        bar_color = tuple(int(c * 0.5 * ease) for c in self.color)
        cv2.rectangle(overlay, (x, y - th - 16), (x + box_w, y + 12), bar_color, -1, cv2.LINE_AA)
        accent_color = tuple(int(c * ease) for c in self.color)
        cv2.rectangle(overlay, (x, y - th - 16), (x + 4, y + 12), accent_color, -1, cv2.LINE_AA)
        text_color = tuple(int(255 * ease) for _ in range(3))
        cv2.putText(overlay, self.text, (x + pad, y), font, scale, text_color, thickness, cv2.LINE_AA)

        cv2.add(frame, overlay, dst=frame)
        return frame

# ============================================================================
# GESTURE MANAGER
# ============================================================================

class GestureEffectManager:
    """Detects gestures each frame and drives the matching flower effect,
    plus every ambient magical-garden system layered on top."""

    def __init__(self, config: Config):
        self.config = config
        self.effects = {
            'open_palm': FullBloomEffect(config),
            'fist':      FlowerBudEffect(config),
            'point':     FlowerTrailEffect(config),
            'peace':     TwinBlossomEffect(config),
            'pinch':     SeedSproutEffect(config),
            'bouquet':   BouquetEffect(config),
        }
        self.announce_colors = {
            'open_palm': (180, 120, 255),
            'fist':      (100, 200, 120),
            'point':     (150, 200, 255),
            'peace':     (200, 140, 255),
            'pinch':     (120, 220, 140),
            'bouquet':   (160, 120, 255),
        }
        self.announcer = GestureAnnouncer(config)
        self.wind = WindSystem(config)
        self.background = BackgroundAtmosphereEffect(config)
        self.falling_petals = FallingPetalsEffect(config)
        self.butterflies = ButterflyEffect(config)
        self.bees = BeeEffect(config)
        self.pollen = PollenEffect(config)
        self.sparkle_dust = SparkleDustEffect(config)
        self.fireflies = FireflyEffect(config)
        self.rainbow_burst = RainbowBurstEffect(config)
        self.smile_detector = SmileDetector(config)
        self.crown = FlowerCrownEffect(config)
        self._active_name: Optional[str] = None
        self._crown_active = False
        self._smile_score = 0.0
        self._sun_time = 0.0

    def update(self, dt, landmarks_list, frame_w, frame_h, face_landmarks=None):
        self._sun_time += dt
        self.wind.update(dt)
        wind_angle = self.wind.angle
        if self.config.enable_background_atmosphere:
            self.background.update(dt, frame_w, frame_h)

        if self.config.enable_smile_crown:
            self._smile_score = self.smile_detector.update(dt, face_landmarks)
            self.crown.update(dt, face_landmarks, self._smile_score, wind_angle)
        else:
            self._smile_score = 0.0

        detected = []
        bouquet_anchor = None

        if landmarks_list:
            if len(landmarks_list) >= 2:
                hand_a, hand_b = landmarks_list[0], landmarks_list[1]
                ga = GestureRecognizer.classify(hand_a)
                gb = GestureRecognizer.classify(hand_b)
                if ga == 'open_palm' and gb == 'open_palm':
                    ca, cb = _palm_center(hand_a), _palm_center(hand_b)
                    gap = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                    hand_scale = max(1.0, math.hypot(hand_a[0][0] - hand_a[9][0], hand_a[0][1] - hand_a[9][1]))
                    if gap < max(90, hand_scale * 3.0):
                        bouquet_anchor = (int((ca[0] + cb[0]) / 2), int((ca[1] + cb[1]) / 2))

            if bouquet_anchor is None:
                for lm in landmarks_list:
                    g = GestureRecognizer.classify(lm)
                    if g:
                        detected.append((g, lm))

        self.effects['bouquet'].update(dt, bouquet_anchor, wind_angle)
        for name, effect in self.effects.items():
            if name == 'bouquet':
                continue
            match = next((lm for g, lm in detected if g == name), None)
            effect.update(dt, match, wind_angle)

        # Fire any rainbow bursts requested by held gestures this frame.
        if self.config.enable_rainbow_burst:
            for name, effect in self.effects.items():
                if getattr(effect, 'burst_pending', False):
                    center = getattr(effect, 'center', None)
                    if center is not None:
                        self.rainbow_burst.trigger(center)
            if self.config.enable_smile_crown and getattr(self.crown, 'burst_pending', False):
                center = self.crown.center
                if center is not None:
                    self.rainbow_burst.trigger(center)
        self.rainbow_burst.update(dt)

        best_name, best_alpha = None, 0.0
        for name, effect in self.effects.items():
            a = getattr(effect, 'alpha', 0.0)
            if a > best_alpha:
                best_alpha = a
                best_name = name
        self._active_name = best_name if best_name and best_alpha > 0.25 else None

        crown_alpha = getattr(self.crown, 'alpha', 0.0) if self.config.enable_smile_crown else 0.0
        self._crown_active = crown_alpha > 0.25 and crown_alpha >= best_alpha

        active_center = None
        if self._active_name == 'point':
            if self.effects['point'].planted:
                active_center = self.effects['point'].planted[-1]['pos']
        elif self._active_name == 'peace':
            active_center = self.effects['peace'].pt1
        elif self._active_name == 'pinch':
            active_center = self.effects['pinch'].base
        elif self._active_name and hasattr(self.effects[self._active_name], 'center'):
            active_center = self.effects[self._active_name].center
        if self._crown_active and self.crown.center is not None:
            active_center = self.crown.center

        # Every currently-active flower center — feeds the aura, pollen, and sparkle dust.
        active_centers = []
        for name, effect in self.effects.items():
            if getattr(effect, 'alpha', 0.0) > 0.3:
                c = getattr(effect, 'center', None)
                if c is not None:
                    active_centers.append(c)
        if self._active_name == 'point' and self.effects['point'].planted:
            active_centers.append(self.effects['point'].planted[-1]['pos'])
        if crown_alpha > 0.3 and self.crown.center is not None:
            active_centers.append(self.crown.center)

        if self.config.enable_falling_petals:
            self.falling_petals.update(dt, frame_w, frame_h, wind_angle)
        if self.config.enable_butterflies:
            self.butterflies.update(dt, active_center)
        if self.config.enable_bees:
            self.bees.update(dt, active_center)
        if self.config.enable_pollen:
            self.pollen.update(dt, active_centers)
        if self.config.enable_sparkle_dust:
            self.sparkle_dust.update(dt, active_centers)
        if self.config.enable_fireflies:
            self.fireflies.update(dt, frame_w, frame_h)

        if self.config.enable_gesture_vfx:
            if self._crown_active:
                announce_key, text, color = 'smile_crown', '🌼 FLOWER CROWN', (210, 150, 255)
            else:
                announce_key = self._active_name
                text = GestureRecognizer.ANNOUNCE_LABELS.get(self._active_name, "")
                color = self.announce_colors.get(self._active_name, (255, 255, 255))
            self.announcer.update(dt, announce_key, text, color)

    def draw(self, frame):
        if self.config.enable_background_atmosphere:
            frame = self.background.draw(frame)
        if self.config.enable_sun_rays:
            frame = draw_sun_rays(frame, self._sun_time)
        if self.config.enable_falling_petals:
            frame = self.falling_petals.draw(frame)
        if self.config.enable_fireflies:
            frame = self.fireflies.draw(frame)
        if self.config.enable_gesture_vfx:
            for effect in self.effects.values():
                frame = effect.draw(frame)
        if self.config.enable_smile_crown:
            frame = self.crown.draw(frame)
        if self.config.enable_pollen:
            frame = self.pollen.draw(frame)
        if self.config.enable_sparkle_dust:
            frame = self.sparkle_dust.draw(frame)
        if self.config.enable_butterflies:
            frame = self.butterflies.draw(frame)
        if self.config.enable_bees:
            frame = self.bees.draw(frame)
        if self.config.enable_rainbow_burst:
            frame = self.rainbow_burst.draw(frame)
        if self.config.enable_gesture_vfx:
            frame = self.announcer.draw(frame)
        return frame

    def active_label(self) -> Optional[str]:
        if self._crown_active:
            return '🌼 Flower Crown'
        if self._active_name:
            return GestureRecognizer.GESTURE_LABELS[self._active_name]
        return None

    def smile_debug(self) -> Tuple[float, float, float]:
        """Returns (smoothed_smile_score, raw_mouth_width_ratio, learned_baseline_ratio)
        for on-screen calibration."""
        baseline = self.smile_detector.baseline_ratio
        return (self._smile_score, self.smile_detector.raw_ratio, baseline if baseline is not None else 0.0)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class HandVFXApp:
    """Main application: webcam capture, hand tracking, and flower VFX."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.tracker = HandTracker(self.config)
        self.face_tracker = FaceTracker(self.config)
        self.skeleton = SkeletonRenderer(self.config)
        self.gesture_vfx = GestureEffectManager(self.config)
        self.last_time = time.time()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        dt = min(0.05, now - self.last_time)
        self.last_time = now

        landmarks_list = self.tracker.detect_hands(frame)
        face_landmarks = self.face_tracker.detect_face(frame) if self.config.enable_smile_crown else None

        frame = self.skeleton.draw(frame, landmarks_list, self.tracker.connections)

        self.gesture_vfx.update(dt, landmarks_list, w, h, face_landmarks)
        frame = self.gesture_vfx.draw(frame)

        self.draw_info(frame)
        return frame

    def draw_info(self, frame: np.ndarray):
        lines = [
            "Blooming Effects  |  q: quit",
            f"Gestures: {'ON' if self.config.enable_gesture_vfx else 'OFF'} (g)  "
            f"Petals: {'ON' if self.config.enable_falling_petals else 'OFF'} (p)  "
            f"Butterflies: {'ON' if self.config.enable_butterflies else 'OFF'} (u)  "
            f"Bees: {'ON' if self.config.enable_bees else 'OFF'} (b)",
            "Hold any gesture ~2s for a rainbow burst  |  f: fireflies  w: wind  s: sun rays  a: background  c: smile crown",
        ]
        if self.config.enable_smile_crown and self.config.enable_face_vfx:
            smoothed, raw_ratio, baseline = self.gesture_vfx.smile_debug()
            lines.append(f"Smile: {smoothed:.2f}  (ratio {raw_ratio:.2f} vs your baseline {baseline:.2f}, "
                         f"rise triggers between {self.config.smile_ratio_low:.2f}-{self.config.smile_ratio_high:.2f})")
        active = self.gesture_vfx.active_label()
        if active:
            lines.append(f"Active: {active}")

        y = 28
        for text in lines:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            y += 24

    def handle_key(self, key: int):
        if key == ord('g'):
            self.config.enable_gesture_vfx = not self.config.enable_gesture_vfx
            print(f"[Bloom] Gestures: {'ON' if self.config.enable_gesture_vfx else 'OFF'}")
        elif key == ord('p'):
            self.config.enable_falling_petals = not self.config.enable_falling_petals
            print(f"[Bloom] Falling petals: {'ON' if self.config.enable_falling_petals else 'OFF'}")
        elif key == ord('u'):
            self.config.enable_butterflies = not self.config.enable_butterflies
            print(f"[Bloom] Butterflies: {'ON' if self.config.enable_butterflies else 'OFF'}")
        elif key == ord('b'):
            self.config.enable_bees = not self.config.enable_bees
            print(f"[Bloom] Bees: {'ON' if self.config.enable_bees else 'OFF'}")
        elif key == ord('f'):
            self.config.enable_fireflies = not self.config.enable_fireflies
            print(f"[Bloom] Fireflies: {'ON' if self.config.enable_fireflies else 'OFF'}")
        elif key == ord('w'):
            self.config.enable_wind = not self.config.enable_wind
            print(f"[Bloom] Wind sway: {'ON' if self.config.enable_wind else 'OFF'}")
        elif key == ord('s'):
            self.config.enable_sun_rays = not self.config.enable_sun_rays
            print(f"[Bloom] Sun rays: {'ON' if self.config.enable_sun_rays else 'OFF'}")
        elif key == ord('a'):
            self.config.enable_background_atmosphere = not self.config.enable_background_atmosphere
            print(f"[Bloom] Background atmosphere: {'ON' if self.config.enable_background_atmosphere else 'OFF'}")
        elif key == ord('c'):
            self.config.enable_smile_crown = not self.config.enable_smile_crown
            print(f"[Bloom] Smile flower crown: {'ON' if self.config.enable_smile_crown else 'OFF'}")
        elif key == ord('d'):
            self.config.enable_face_vfx = not self.config.enable_face_vfx
            print(f"[Bloom] Smile calibration readout: {'ON' if self.config.enable_face_vfx else 'OFF'}")

    def run(self):
        cap = cv2.VideoCapture(0)

        print("[Bloom] Starting. Controls: q=quit  g=gestures  p=falling petals  u=butterflies  b=bees  f=fireflies  w=wind  s=sun rays  a=background  c=smile crown  d=smile readout")
        print("[Bloom] Gestures:")
        print("   Open palm              -> Full Bloom")
        print("   Closed fist            -> Flower Bud")
        print("   Point (index only)     -> Flower Trail")
        print("   Peace sign             -> Twin Blossoms")
        print("   Pinch (thumb+index)    -> Seed Sprout")
        print("   Two open palms close   -> Flower Bouquet")
        print("   Smile 🙂               -> Flower Crown (grows around your head)")
        print("   Hold any gesture ~2s   -> Rainbow Flower Burst")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self.process_frame(frame)
                cv2.imshow("Blooming Effects", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                self.handle_key(key)
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandVFXApp()
    app.run()