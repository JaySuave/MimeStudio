import cv2
from cvzone.HandTrackingModule import HandDetector
import pygame
import numpy as np
import math
import time
import os

# ===================== HELPERS =====================

def get_fingers_robust(hand, detector):
    """
    Returns [thumb, index, middle, ring, pinky].
    Uses geometry for thumb to fix rotation issues.
    """
    fingers = detector.fingersUp(hand)
    
    lm = hand["lmList"]
    x4, y4, _ = lm[4]    # Thumb Tip
    x5, y5, _ = lm[5]    # Index Knuckle
    x17, y17, _ = lm[17] # Pinky Knuckle

    palm_width = math.hypot(x5 - x17, y5 - y17)
    thumb_dist = math.hypot(x4 - x17, y4 - y17)

    # If thumb tip is far from pinky knuckle it's OPEN.
    if thumb_dist > (palm_width * 0.9):
        fingers[0] = 1
    else:
        fingers[0] = 0
        
    return fingers

# ===================== DRUM MACHINE =====================

class DrumMachine:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.kick_sound = self._generate_kick()
        self.snare_sound = self._generate_snare()
        self.piano_sounds = self._load_piano_sounds()
        
        self.is_recording = True
        self.start_time = time.time()
        self.loop_duration = 0
        self.sequence = []
        self.last_check_time = 0
        
        # Cooldown settings
        self.last_trigger_time = 0
        self.cooldown = 0.4  # Slightly faster for punchy beats

    def _generate_kick(self):
        # 1. FREQUENCY: Start at 120Hz dropping to 0Hz
        duration = 0.2
        num_samples = int(self.sample_rate * duration)
        t = np.arange(num_samples) / self.sample_rate
        freq = 120 * np.exp(-8 * t) 
        waveform = np.sin(2 * np.pi * freq * t) * np.exp(-4 * t)
        
        # 2. VOLUME: High gain
        waveform = (waveform * 0.95 * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack((waveform, waveform)))

    def _generate_snare(self):
        duration = 0.15
        num_samples = int(self.sample_rate * duration)
        t = np.arange(num_samples) / self.sample_rate
        
        noise = np.random.uniform(-1, 1, num_samples)
        
        # 3. CLICK FIX: Attack Fade-in
        fade_in_len = 50
        if num_samples > fade_in_len:
            noise[:fade_in_len] *= np.linspace(0, 1, fade_in_len)

        envelope = np.exp(-12 * t)
        waveform = noise * envelope
        
        # 4. VOLUME: High gain
        waveform = (waveform * 0.8 * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack((waveform, waveform)))

    def _load_piano_sounds(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sounds"))
        file_map = {
            0: "polegar.wav",
            1: "indicador.wav",
            2: "medio.wav",
            3: "anelar.wav",
            4: "mindinho.wav",
        }
        sounds = {}
        for idx, filename in file_map.items():
            path = os.path.join(base_dir, filename)
            if not os.path.exists(path):
                continue
            try:
                sounds[idx] = pygame.mixer.Sound(path)
            except pygame.error:
                continue
        return sounds

    def prepare_recording(self):
        self.is_recording = False
        self.sequence = []
        self.loop_duration = 0
        self.last_check_time = 0

    def start_recording(self):
        self.is_recording = True
        self.start_time = time.time()
        self.sequence = []
        self.loop_duration = 0
        self.last_check_time = 0

    def trigger(self, drum_type, finger_idx=None, hand_type=None):
        curr_time = time.time()
        if curr_time - self.last_trigger_time < self.cooldown:
            return False

        if drum_type == "kick":
            self.kick_sound.play()
        elif drum_type == "snare":
            self.snare_sound.play()
        elif drum_type == "piano":
            sound = self.piano_sounds.get(finger_idx)
            if sound is None:
                return False
            sound.play()
        
        if self.is_recording and drum_type in ("kick", "snare"):
            offset = curr_time - self.start_time
            self.sequence.append({"type": drum_type, "time": offset})
        
        self.last_trigger_time = curr_time
        return True

    def finish_recording(self):
        self.is_recording = False
        self.loop_duration = time.time() - self.start_time
        self.start_time = time.time()
        print(f"Loop locked. Duration: {self.loop_duration:.2f}s. Events: {len(self.sequence)}")

    def update_loop(self):
        if self.is_recording or self.loop_duration == 0:
            return

        now = time.time()
        loop_time = (now - self.start_time) % self.loop_duration
        prev_loop_time = self.last_check_time
        
        for event in self.sequence:
            t = event["time"]
            should_play = False
            if prev_loop_time < loop_time:
                if prev_loop_time <= t < loop_time: should_play = True
            else:
                if t >= prev_loop_time or t < loop_time: should_play = True
            
            if should_play:
                if event["type"] == "kick": self.kick_sound.play()
                elif event["type"] == "snare": self.snare_sound.play()

        self.last_check_time = loop_time

# ===================== SYNTH MANAGER =====================

class SynthManager:
    def __init__(self, sample_rate=44100, duration=0.5):
        # Buffer 2048 prevents crackling
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=2048)

        pygame.mixer.set_num_channels(32)

        self.sample_rate = sample_rate
        self.duration = duration
        self.base_freqs = {0: 261.63, 1: 293.66, 2: 329.63, 3: 349.23, 4: 392.00}
        self.bend_factor = 1.0
        self.volume = 1.0
        phases = np.linspace(0, 1, 256, endpoint=False)
        self.shape_table = np.sin(2 * np.pi * phases).astype(np.float32)
        self.sounds = {}
        self.channels = {i: None for i in self.base_freqs.keys()}
        self.active_notes = {i: 0 for i in self.base_freqs.keys()}
        self._regenerate_all_sounds()

    def _build_sound(self, freq: float):
        num_samples = int(self.sample_rate * self.duration)
        t = np.arange(num_samples) / self.sample_rate
        phase = (freq * t) % 1.0
        table_x = np.linspace(0, 1, len(self.shape_table), endpoint=False)
        wave = np.interp(phase, table_x, self.shape_table).astype(np.float32)
        
        # Synth quieter than drums
        wave *= 0.25
        
        audio_mono = (wave * 32767).astype(np.int16)
        audio_stereo = np.column_stack((audio_mono, audio_mono))
        return pygame.sndarray.make_sound(audio_stereo)

    def _regenerate_all_sounds(self):
        for finger, base_f in self.base_freqs.items():
            freq = base_f * self.bend_factor
            self.sounds[finger] = self._build_sound(freq)
        for finger, is_on in self.active_notes.items():
            if is_on:
                ch = self.channels.get(finger)
                if ch is not None: 
                    ch.fadeout(30)
                ch = self.sounds[finger].play(loops=-1)
                ch.set_volume(self.volume)
                self.channels[finger] = ch

    def get_visual_waveform(self, width=400):
        t = np.linspace(0, 0.02, width)
        mix = np.zeros(width)
        active_count = 0
        for finger_idx, is_active in self.active_notes.items():
            if is_active:
                active_count += 1
                freq = self.base_freqs[finger_idx] * self.bend_factor
                phase = (freq * t) % 1.0
                table_x = np.linspace(0, 1, len(self.shape_table), endpoint=False)
                mix += np.interp(phase, table_x, self.shape_table)
        mix *= self.volume
        if active_count > 1: mix = mix / math.sqrt(active_count)
        return mix

    def set_shape(self, shape_table):
        if shape_table is None or len(shape_table) < 2: return
        shape_table = np.asarray(shape_table, dtype=np.float32)
        if self.shape_table.shape == shape_table.shape:
            if np.max(np.abs(self.shape_table - shape_table)) < 1e-2: return
        self.shape_table = shape_table
        self._regenerate_all_sounds()

    def set_pitch_bend(self, semitones: float):
        factor = 2.0 ** (semitones / 12.0)
        if abs(factor - self.bend_factor) < 1e-3: return
        self.bend_factor = factor
        self._regenerate_all_sounds()

    def set_volume(self, volume: float):
        volume = float(np.clip(volume, 0.0, 1.0))
        self.volume = volume
        for ch in self.channels.values():
            if ch is not None: ch.set_volume(volume)

    def note_on(self, finger_idx: int):
        if self.active_notes.get(finger_idx, 0): return
        sound = self.sounds.get(finger_idx)
        if sound is None: return
        ch = sound.play(loops=-1)
        ch.set_volume(self.volume)
        self.channels[finger_idx] = ch
        self.active_notes[finger_idx] = 1

    def note_off(self, finger_idx: int):
        if not self.active_notes.get(finger_idx, 0): return
        ch = self.channels.get(finger_idx)
        if ch is not None: 
            # CLICK FIX: Fade out over 50ms
            ch.fadeout(50) 
        self.channels[finger_idx] = None
        self.active_notes[finger_idx] = 0

    def stop_all(self):
        for ch in self.channels.values():
            if ch is not None: 
                ch.fadeout(100)
        self.channels = {k: None for k in self.channels}
        self.active_notes = {k: 0 for k in self.active_notes}


# ===================== SHAPE HAND (LEFT) =====================

TIP_INDEXES = [4, 8, 12, 16]

class ShapeHandManager:
    def __init__(self, table_size=256, smooth_alpha=0.7):
        self.table_size = table_size
        self.smooth_alpha = smooth_alpha
        self.last_vertices = None
        self.last_table = None

    def update_from_hand(self, img, hand, detector: HandDetector, synth: SynthManager):
        h, w, _ = img.shape
        fingers = get_fingers_robust(hand, detector)
        xs = []
        ys = []
        for i, tip_idx in enumerate(TIP_INDEXES):
            if fingers[i] == 1:
                x_px, y_px, _ = hand["lmList"][tip_idx]
                xs.append(x_px / w)
                ys.append(y_px / h)

        if len(xs) < 2:
            if self.last_vertices is not None:
                self._draw_vertices(img, self.last_vertices[0], self.last_vertices[1])
            if self.last_table is not None:
                synth.set_shape(self.last_table)
            return

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        if self.last_vertices is not None and len(self.last_vertices[0]) == len(xs):
            prev_xs, prev_ys = self.last_vertices
            xs = self.smooth_alpha * prev_xs + (1 - self.smooth_alpha) * xs
            ys = self.smooth_alpha * prev_ys + (1 - self.smooth_alpha) * ys
        self.last_vertices = (xs, ys)

        phases = np.linspace(0, 1, self.table_size, endpoint=False)
        amps_vertices = 1.0 - 2.0 * ys
        table = np.interp(phases, xs, amps_vertices, left=amps_vertices[0], right=amps_vertices[-1]).astype(np.float32)
        self.last_table = table
        synth.set_shape(table)
        self._draw_vertices(img, xs, ys)

    def _draw_vertices(self, img, xs, ys):
        h, w, _ = img.shape
        pts = []
        for x, y in zip(xs, ys):
            px = int(x * w)
            py = int(y * h)
            pts.append((px, py))
        if len(pts) >= 2:
            overlay = img.copy()
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i + 1], (0, 255, 255), 2)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        for p in pts:
            cv2.circle(img, p, 6, (0, 255, 255), -1)


# ===================== PLAY HAND (RIGHT) =====================

class FingerCounter:
    def __init__(self, detection_confidence=0.8, max_hands=2):
        self.detector = HandDetector(detectionCon=detection_confidence, maxHands=max_hands)
        self.synth = SynthManager()
        self.shape_manager = ShapeHandManager()
        
        self.last_pitch_norm = 0.5
        self.last_vol_norm = 0.5
        self.smooth_alpha = 0.8
        self.last_log_time = 0

    def process_synth_mode(self, img, hands):
        left_hand = None
        right_hand = None
        for hand in hands:
            if hand["type"] == "Left": left_hand = hand
            elif hand["type"] == "Right": right_hand = hand

        if left_hand is not None:
            self.shape_manager.update_from_hand(img, left_hand, self.detector, self.synth)
        if right_hand is not None:
            self._update_play_hand(img, right_hand)
        return img

    def _update_play_hand(self, img, hand):
        h, w, _ = img.shape
        lm = hand["lmList"]
        fingers = get_fingers_robust(hand, self.detector)

        # Pitch & Volume
        ix, iy, _ = lm[8]
        x_norm = ix / w
        y_norm = iy / h
        self.last_pitch_norm = self.smooth_alpha * self.last_pitch_norm + (1 - self.smooth_alpha) * x_norm
        self.last_vol_norm = self.smooth_alpha * self.last_vol_norm + (1 - self.smooth_alpha) * y_norm
        bend_semitones = (self.last_pitch_norm - 0.5) * 1.0
        self.synth.set_pitch_bend(bend_semitones)
        volume = 1.0 - self.last_vol_norm
        self.synth.set_volume(volume)

        # Triggers
        tips = [4, 8, 12, 16, 20]
        for i in range(5):
            if fingers[i] == 1:
                self.synth.note_on(i)
                fx, fy, _ = lm[tips[i]]
                cv2.circle(img, (int(fx), int(fy)), 10, (0, 255, 0), -1)
            else:
                self.synth.note_off(i)
                fx, fy, _ = lm[tips[i]]
                cv2.circle(img, (int(fx), int(fy)), 10, (0, 0, 255), 1)


# ===================== WAVEFORM WINDOW =====================

def draw_waveform(samples, width=400, height=200):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if samples is None or len(samples) == 0 or np.max(np.abs(samples)) == 0:
        cv2.putText(img, "Silence", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        return img

    mid = height // 2
    scale = (height // 2) - 10
    pts = []
    for x in range(min(len(samples), width)):
        val = samples[x]
        y = int(mid - (val * scale))
        pts.append((x, y))

    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 2)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (100, 100, 100), 1)
    cv2.putText(img, "Output Wave", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


# ===================== MAIN =====================

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    finger_counter = FingerCounter()
    drum_machine = DrumMachine()

    APP_MODE = "DRUMS"
    countdown_duration = 3.0
    countdown_start_time = 0.0
    pending_mode = "DRUMS"
    countdown_reason = ""
    piano_image = None
    hand_was_in_zone = {}
    last_gesture = {}
    
    # === CONFIG: PROXIMITY / HIT THRESHOLD ===
    # Hand area (w*h) must be larger than this to count as a hit.
    # 25,000 is a good start for 1280x720. Increase if you want to punch closer.
    HIT_THRESHOLD = 35000 

    print("=== MODE: DRUMS ===")
    print("RIGHT HAND (fist) = KICK | LEFT HAND (open) = SNARE")
    print("Press 'R' to start recording")
    print("Press 'F' to toggle fullscreen | Press 'Q' to quit")

    if not cap.isOpened():
        print("Error: could not open camera")
        return

    try:
        while True:
            success, img = cap.read()
            if not success: break
            
            # Do not horizontally flip the camera feed
            hands, img = finger_counter.detector.findHands(img, flipType=True)

            if APP_MODE == "COUNTDOWN":
                elapsed = time.time() - countdown_start_time
                remaining = countdown_duration - elapsed
                
                if remaining > 0:
                    # Display countdown
                    countdown_num = int(remaining) + 1
                    
                    # Large countdown number in center
                    h, w, _ = img.shape
                    text = str(countdown_num)
                    font_scale = 10
                    thickness = 20
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    
                    # Draw countdown with shadow for better visibility
                    cv2.putText(img, text, (text_x + 5, text_y + 5), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 5)
                    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), thickness)
                    
                    # Recording countdown message
                    cv2.putText(img, "RECORDING STARTS SOON!", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(img, "Get ready to play...", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                else:
                    # Countdown finished - return to the mode we were in and start recording
                    APP_MODE = pending_mode
                    drum_machine.start_recording()
                    print(f"Recording started in {APP_MODE} mode!")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elif APP_MODE == "DRUMS":
                cv2.putText(img, "MODE: DRUMS", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
                rec_status = "RECORDING" if drum_machine.is_recording else "NOT RECORDING"
                rec_color = (0, 255, 0) if drum_machine.is_recording else (128, 128, 128)
                cv2.putText(img, f"R: {rec_status}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, rec_color, 2)
                
                # Track which hands are currently detected
                current_hand_ids = set()
                
                if hands:
                    for hand in hands:
                        # 1. GET FINGERS
                        fingers = get_fingers_robust(hand, finger_counter.detector)
                        total_up = sum(fingers)
                        
                        # 2. GET AREA (Z-AXIS / DEPTH)
                        x, y, w, h = hand['bbox']
                        area = w * h
                        
                        # Draw visual indicator for hit strength/proximity
                        color = (0, 0, 255)
                        if area > HIT_THRESHOLD:
                            color = (0, 255, 0) # Green = Close enough
                        
                        # Draw the bounding box to show "Hit Size"
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        
                        # 3. TRIGGER LOGIC
                        # Kick: Fist (0 fingers) AND Close (Area > threshold)
                        if total_up == 0 and area > HIT_THRESHOLD:
                            if drum_machine.trigger("kick"):
                                cv2.putText(img, "KICK!", (x, y - 20), 
                                            cv2.FONT_HERSHEY_TRIPLEX, 2, (255,0,0), 3)
                        
                        # Snare: Open (5 fingers) AND Close (Area > threshold)
                        elif total_up == 5 and area > HIT_THRESHOLD:
                            if drum_machine.trigger("snare"):
                                cv2.putText(img, "SNARE!", (x, y - 20), 
                                            cv2.FONT_HERSHEY_TRIPLEX, 2, (0,255,255), 3)

                key = cv2.waitKey(1)
                if key & 0xFF == ord(' '):
                    drum_machine.finish_recording()
                    APP_MODE = "SYNTH"
                    print("=== MODE: SYNTH ===")
                    print("Left hand: shape waveform | Right hand: play notes")
                    print("PRESS 'R' to toggle recording")
                elif key & 0xFF == ord('r'):
                    if drum_machine.is_recording:
                        drum_machine.finish_recording()
                    else:
                        # Start countdown before recording
                        drum_machine.prepare_recording()
                        countdown_start_time = time.time()
                        countdown_reason = "recording"
                        pending_mode = "DRUMS"
                        APP_MODE = "COUNTDOWN"
                elif key & 0xFF == ord('q'):
                    break

            elif APP_MODE == "SYNTH":
                cv2.putText(img, "MODE: SYNTH", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
                rec_status = "RECORDING" if drum_machine.is_recording else "NOT RECORDING"
                rec_color = (0, 255, 0) if drum_machine.is_recording else (128, 128, 128)
                cv2.putText(img, f"R: {rec_status}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, rec_color, 2)
                drum_machine.update_loop()
                if hands:
                    img = finger_counter.process_synth_mode(img, hands)
                
                live_wave = finger_counter.synth.get_visual_waveform(width=400)
                wave_img = draw_waveform(live_wave)
                cv2.imshow("Waveform", wave_img)
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord(' '):
                    APP_MODE = "PIANO"
                    print("=== MODE: PIANO ===")
                    print("RIGHT HAND: Thumb=Do | Index=Re | Middle=Mi | Ring=Fa | Pinky=Sol")
                    print("LEFT HAND: Thumb=La | Index=Si | Pinky=Do")
                    print("PRESS 'R' to toggle recording")
                elif key & 0xFF == ord('r'):
                    if drum_machine.is_recording:
                        drum_machine.finish_recording()
                    else:
                        # Start countdown before recording
                        drum_machine.prepare_recording()
                        countdown_start_time = time.time()
                        countdown_reason = "recording"
                        pending_mode = "SYNTH"
                        APP_MODE = "COUNTDOWN"
                elif key & 0xFF == ord('q'):
                    break

            elif APP_MODE == "PIANO":
                # Overlay piano image if available
                if piano_image is not None:
                    h, w, _ = img.shape
                    img_h, img_w, _ = piano_image.shape
                    
                    # Resize piano image to fit in top-right corner (25% of screen width)
                    target_width = int(w * 0.25)
                    aspect_ratio = img_h / img_w
                    target_height = int(target_width * aspect_ratio)
                    
                    # Resize the piano image
                    resized_piano = cv2.resize(piano_image, (target_width, target_height))
                    
                    # Position in top-right corner with some padding
                    padding = 20
                    y_offset = padding
                    x_offset = w - target_width - padding
                    
                    # Ensure the image fits within bounds
                    if y_offset >= 0 and x_offset >= 0:
                        # Create semi-transparent overlay
                        overlay = img.copy()
                        overlay[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_piano
                        # Blend with 70% opacity
                        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                
                cv2.putText(img, "MODE: PIANO", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 255), 2)
                rec_status = "RECORDING" if drum_machine.is_recording else "NOT RECORDING"
                rec_color = (0, 255, 0) if drum_machine.is_recording else (128, 128, 128)
                cv2.putText(img, f"R: {rec_status}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, rec_color, 2)
                drum_machine.update_loop()
                
                # Track which hands are currently detected
                current_hand_ids = set()
                
                if hands:
                    for idx, hand in enumerate(hands):
                        # Use hand type as identifier for piano too
                        hand_type = hand["type"]
                        hand_id = f"{hand_type}_piano"
                        current_hand_ids.add(hand_id)
                        
                        fingers = get_fingers_robust(hand, finger_counter.detector)
                        x, y, w, h = hand['bbox']
                        area = w * h
                        
                        color = (0, 0, 255)
                        in_zone = area > HIT_THRESHOLD
                        
                        if in_zone:
                            color = (0, 255, 0)
                        
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        
                        if in_zone:
                            # Define notes based on hand type
                            if hand_type == "Right":
                                note_names = ["Do", "Re", "Mi", "Fa", "Sol"]
                                active_fingers = [0, 1, 2, 3, 4]  # All 5 fingers
                            else:  # Left hand
                                note_names = ["La", "Si", "", "", "Do"]  # Thumb, Index, Pinky
                                active_fingers = [0, 1, 4]  # Only thumb, index, and pinky
                            
                            finger_colors = [
                                (255, 0, 0),    # Thumb - Blue
                                (255, 128, 0),  # Index - Cyan
                                (255, 255, 0),  # Middle - Yellow
                                (0, 255, 0),    # Ring - Green
                                (128, 0, 255),  # Pinky - Purple
                            ]
                            
                            lm = hand["lmList"]
                            finger_tips = [4, 8, 12, 16, 20]
                            
                            # Track which fingers were up last frame for this hand
                            was_in_zone = hand_was_in_zone.get(hand_id, False)
                            prev_fingers = last_gesture.get(hand_id, [0, 0, 0, 0, 0])
                            
                            for i in range(5):
                                tip_x, tip_y, _ = lm[finger_tips[i]]
                                
                                # Only process active fingers for this hand
                                if i in active_fingers:
                                    if fingers[i] == 1:
                                        cv2.circle(img, (tip_x, tip_y), 12, finger_colors[i], -1)
                                        
                                        # Only trigger if:
                                        # 1. Hand just entered zone with finger up OR
                                        # 2. Finger just went up while in zone
                                        should_trigger = False
                                        if not was_in_zone and fingers[i] == 1:
                                            should_trigger = True
                                        elif was_in_zone and prev_fingers[i] == 0 and fingers[i] == 1:
                                            should_trigger = True
                                        
                                        if should_trigger:
                                            if drum_machine.trigger("piano", i, hand_type):
                                                cv2.putText(img, f"{note_names[i]}", 
                                                            (tip_x - 20, tip_y - 20), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, finger_colors[i], 2)
                                    else:
                                        cv2.circle(img, (tip_x, tip_y), 8, finger_colors[i], 2)
                                else:
                                    # Inactive fingers (middle and ring for left hand) - show as disabled
                                    cv2.circle(img, (tip_x, tip_y), 6, (100, 100, 100), 1)
                            
                            # Update state
                            hand_was_in_zone[hand_id] = in_zone
                            last_gesture[hand_id] = fingers.copy()
                        else:
                            # Hand not in zone
                            lm = hand["lmList"]
                            finger_tips = [4, 8, 12, 16, 20]
                            finger_colors = [
                                (255, 0, 0),    # Thumb - Blue
                                (255, 128, 0),  # Index - Cyan
                                (255, 255, 0),  # Middle - Yellow
                                (0, 255, 0),    # Ring - Green
                                (128, 0, 255),  # Pinky - Purple
                            ]
                            for i in range(5):
                                tip_x, tip_y, _ = lm[finger_tips[i]]
                                cv2.circle(img, (tip_x, tip_y), 8, finger_colors[i], 2)
                            
                            hand_was_in_zone[hand_id] = False
                            last_gesture[hand_id] = [0, 0, 0, 0, 0]
                
                # Clean up state for hands that disappeared
                hand_ids_to_remove = [hid for hid in hand_was_in_zone.keys() if hid not in current_hand_ids and "_piano" in hid]
                for hid in hand_ids_to_remove:
                    hand_was_in_zone.pop(hid, None)
                    last_gesture.pop(hid, None)

                key = cv2.waitKey(1)
                if key & 0xFF == ord(' '):
                    APP_MODE = "DRUMS"
                    print("=== MODE: DRUMS ===")
                    print("Make a fist for KICK, open hand for SNARE")
                    print("PRESS 'R' to toggle recording")
                elif key & 0xFF == ord('r'):
                    if drum_machine.is_recording:
                        drum_machine.finish_recording()
                    else:
                        # Start countdown before recording
                        drum_machine.prepare_recording()
                        countdown_start_time = time.time()
                        countdown_reason = "recording"
                        pending_mode = "PIANO"
                        APP_MODE = "COUNTDOWN"
                elif key & 0xFF == ord('q'):
                    break

            cv2.imshow("Wave-Gesture Synth", img)

    finally:
        finger_counter.synth.stop_all()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()