"""
RUKA Hand Teleoperation (MediaPipe keypoint control)

Source: Derived from the original RUKA teleop_gesture implementation by the upstream contributors.
Modifications: English logs/docs, parameter tuning, and minor UX/safety improvements.
Acknowledgements: Full credit to the original authors of the RUKA project. See LICENSE and README for details.
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
from collections import deque
from ruka_hand.control.hand import Hand

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MediaPipe finger tip landmark indices
FINGER_TIPS = {
    'THUMB': 4,
    'INDEX_FINGER': 8,
    'MIDDLE_FINGER': 12,
    'RING_FINGER': 16,
    'PINKY': 20
}

# Mapping from RUKA finger names to MediaPipe names
FINGER_MAPPING = {
    'Thumb': 'THUMB',
    'Index': 'INDEX_FINGER',
    'Middle': 'MIDDLE_FINGER',
    'Ring': 'RING_FINGER',
    'Pinky': 'PINKY'
}

# Normalize joint angles assuming 90-degree max flexion
ANGLE_MAX_DEG = 90.0
ANGLE_MAX_RAD = np.deg2rad(ANGLE_MAX_DEG)
EPS = 1e-8

# Robust base joint indices for each finger (MCP/CMC)
MP_FINGER_BASE_IDX = {
    'THUMB': 1,
    'INDEX_FINGER': 5,
    'MIDDLE_FINGER': 9,
    'RING_FINGER': 13,
    'PINKY': 17,
}

# Position smoothing parameters
SMOOTHING_FACTOR = 0.3  # Between 0-1, smaller = smoother
MAX_COMMAND_RATE = 15   # Max command frequency in Hz
POSITION_HISTORY_SIZE = 5  # Size of position history buffer

class HandController:
    """Controller mapping MediaPipe hand gestures to RUKA robotic hand"""

    def __init__(self, hand_type='right', smoothing=True):
        """
        RUKA Hand Teleoperation (MediaPipe keypoint control)
        
        Source: Derived from the original RUKA teleop_gesture implementation by the upstream contributors.
        Modifications: English logs/docs, parameter tuning, and minor UX/safety improvements.
        Acknowledgements: Full credit to the original authors of the RUKA project. See LICENSE and README for details.
        """
        print("Initializing RUKA Hand Controller...")

        # Initialize robotic hand
        self.hand = Hand(hand_type=hand_type)
        self.hand_type = hand_type
        self.smoothing = smoothing
        self.smoothing_factor = 0.1  # Lower for more smoothing
        self.position_history_size = 10  # Increase history buffer
        self.curl_threshold = 0.05  # Below this = considered straight (lower = more sensitive)

        # Validate calibration data exists
        if not hasattr(self.hand, 'curled_bound') or not hasattr(self.hand, 'tensioned_pos'):
            raise RuntimeError("❌ Error: Hand calibration data missing!\nPlease run calibration first: python calibrate_motors_modify.py -ht " + hand_type)

        # Create position history buffer for smoothing
        if smoothing:
            self.position_history = deque(maxlen=POSITION_HISTORY_SIZE)

        # Frame rate control
        self.last_command_time = 0
        self.command_interval = 1.0 / MAX_COMMAND_RATE

        # Safety parameters
        self.safety_margin = 10  # Position safety margin in encoder units
        self.max_curl = 0.95     # Max allowed curl (prevents over-closing)

        # Calibrate neutral pose at startup to avoid starting at limits
        self.calibrating = True
        self.calib_frames = 20   # ~1 second at 20 FPS
        self.calib_count = 0
        self.neutral_thumb = np.zeros(3, dtype=float)
        self.neutral_fingers = {
            'Index': np.zeros(2, dtype=float),
            'Middle': np.zeros(2, dtype=float),
            'Ring': np.zeros(2, dtype=float),
            'Pinky': np.zeros(2, dtype=float),
        }
        self._sum_thumb = np.zeros(3, dtype=float)
        self._sum_fingers = {
            'Index': np.zeros(2, dtype=float),
            'Middle': np.zeros(2, dtype=float),
            'Ring': np.zeros(2, dtype=float),
            'Pinky': np.zeros(2, dtype=float),
        }
        # Lightweight debug: print thumb raw/adjusted every 10 frames
        self._dbg_frame = 0

        print(f"Hand initialized successfully! Type: {hand_type}")
        print(f"Safety margin: {self.safety_margin} units, Max curl: {self.max_curl:.2f}")
        print("Calibrating neutral pose — please relax your hand naturally for ~1 second...")

    def close(self):
        """Safely close the robotic hand"""
        print("\nSafely shutting down hand...")
        try:
            # Move to open position
            self.hand.set_pos(self.hand.tensioned_pos)
            time.sleep(0.5)
            self.hand.close()
            print("Hand safely closed")
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")

    def _calculate_finger_curl(self, landmarks, finger_name):
        # Robust joint angle calculation, normalized to 90 degrees
        def safe_angle(u, v):
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < EPS or nv < EPS:
                return 0.0
            cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
            return np.arccos(cosang)

        base_idx = MP_FINGER_BASE_IDX[finger_name]
        joint1_idx = base_idx + 1
        joint2_idx = base_idx + 2
        tip_idx = base_idx + 3

        base = np.array([landmarks[base_idx].x, landmarks[base_idx].y, landmarks[base_idx].z])
        joint1 = np.array([landmarks[joint1_idx].x, landmarks[joint1_idx].y, landmarks[joint1_idx].z])
        joint2 = np.array([landmarks[joint2_idx].x, landmarks[joint2_idx].y, landmarks[joint2_idx].z])
        tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])

        # MCP flexion: angle between palm direction and proximal phalanx
        vec_palm = base - wrist
        vec_proximal = joint1 - base
        angle_mcp = safe_angle(vec_palm, vec_proximal)
        curl_mcp = float(np.clip(angle_mcp / ANGLE_MAX_RAD, 0.0, 1.0))

        # PIP and DIP: average normalized angles for robustness
        vec1 = joint1 - base
        vec2 = joint2 - joint1
        vec3 = tip - joint2
        angle_pip = safe_angle(vec1, vec2)
        angle_dip = safe_angle(vec2, vec3)
        curl_pip = float(np.clip(angle_pip / ANGLE_MAX_RAD, 0.0, 1.0))
        curl_dip = float(np.clip(angle_dip / ANGLE_MAX_RAD, 0.0, 1.0))
        curl_distal = 0.5 * (curl_pip + curl_dip)

        return (curl_distal, curl_mcp)

    def _map_curl_to_position(self, curl, motor_index):
        """
        Map curl value (0-1) to motor position.
        motor_index is 0-based, matching tensioned_pos/curled_bound arrays.
        """
        curl = float(np.clip(curl, 0, self.max_curl))
        if curl < self.curl_threshold:
            curl = 0.0
        idx = motor_index  # 0-based index
        if self.hand_type == "right":
            min_pos = int(self.hand.tensioned_pos[idx] + self.safety_margin)
            max_pos = int(self.hand.curled_bound[idx] - self.safety_margin)
        else:  # left hand
            min_pos = int(self.hand.curled_bound[idx] + self.safety_margin)
            max_pos = int(self.hand.tensioned_pos[idx] - self.safety_margin)
        return int(min_pos + curl * (max_pos - min_pos))

    def _calculate_thumb_metrics(self, landmarks):
        """Calculate thumb-specific metrics normalized to 90 degrees.
        Returns: (open_metric, adduct_metric, in_metric) in [0,1]
        """
        def safe_angle(u, v):
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < EPS or nv < EPS:
                return 0.0
            cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
            return np.arccos(cosang)

        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        thumb_cmc = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        thumb_mcp = np.array([landmarks[2].x, landmarks[2].y, landmarks[2].z])
        thumb_ip = np.array([landmarks[3].x, landmarks[3].y, landmarks[3].z])
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])

        # IP flexion (tip curl)
        angle_ip = safe_angle(thumb_tip - thumb_ip, thumb_mcp - thumb_ip)
        angle_mcp = safe_angle(thumb_ip - thumb_mcp, thumb_cmc - thumb_mcp)

        ip_flex = float(np.clip((np.pi - angle_ip) / ANGLE_MAX_RAD, 0.0, 1.0))  # Based on joints 2-3-4 (vertex at 3)
        mcp_flex = float(np.clip((np.pi - angle_mcp) / ANGLE_MAX_RAD, 0.0, 1.0))  # Based on joints 1-2-3 (vertex at 2)

        # Output mapping per user spec:
        #   open_metric & in_metric ← IP flexion (top joints)
        #   adduct_metric ← MCP flexion (base joint)
        metric_1 = ip_flex     # Top joint 1 (2-3-4)
        metric_2 = ip_flex       # Top joint 2 (2-3-4), same as top joint 1
        metric_3 = mcp_flex  # Base joint (1-2-3)
        return metric_1,metric_2, metric_3

    def process_hand_landmarks(self, landmarks, handedness):
        # Calculate curl for each finger
        curls = {}
        finger_map = FINGER_MAPPING  # Same mapping for left/right; angle calc is mirror-agnostic
        for ruka_finger, mp_finger in finger_map.items():
            if ruka_finger == 'Thumb':
                open_m, adduct_m, in_m = self._calculate_thumb_metrics(landmarks)
                curls[ruka_finger] = (open_m, adduct_m, in_m)
            else:
                curl = self._calculate_finger_curl(landmarks, mp_finger)
                curls[ruka_finger] = curl

        # During calibration: accumulate neutral pose and hold open position
        if self.calibrating:
            if 'Thumb' in curls:
                self._sum_thumb += np.array(curls['Thumb'], dtype=float)
            for f in ['Index', 'Middle', 'Ring', 'Pinky']:
                if f in curls:
                    self._sum_fingers[f] += np.array(curls[f], dtype=float)
            self.calib_count += 1
            if self.calib_count % 5 == 0:
                print(f"Calibration progress: {self.calib_count}/{self.calib_frames}")
            # Maintain open pose during calibration, send at rate limit
            self._rate_limited_send(self.hand.tensioned_pos)
            if self.calib_count >= self.calib_frames:
                self.neutral_thumb = self._sum_thumb / float(self.calib_count)
                for f in ['Index', 'Middle', 'Ring', 'Pinky']:
                    self.neutral_fingers[f] = self._sum_fingers[f] / float(self.calib_count)
                self.calibrating = False
                print("Neutral pose calibration complete. Taking control.")
            return np.copy(self.hand.tensioned_pos)

        # Dynamic normalization relative to neutral pose:
        # - Thumb: scale by (value - neutral) / (1 - neutral), then clip [0,1] → avoids zeroing all values
        # - Other fingers: simple baseline subtraction (can be switched to scaling if needed)
        adjusted = {}
        if 'Thumb' in curls:
            t_raw = np.array(curls['Thumb'], dtype=float)
            denom = np.maximum(1e-3, 1.0 - self.neutral_thumb)
            t_adj = (t_raw - self.neutral_thumb) / denom
            t_adj = np.clip(t_adj, 0.0, 1.0)
            adjusted['Thumb'] = tuple(t_adj)
            # Debug print every 10 frames
            self._dbg_frame = (self._dbg_frame + 1) % 10
        for f in ['Index', 'Middle', 'Ring', 'Pinky']:
            if f in curls:
                v = np.array(curls[f], dtype=float) - self.neutral_fingers[f]
                adjusted[f] = tuple(np.clip(v, 0.0, 1.0))

        # Map to motor positions
        target_pos = np.copy(self.hand.tensioned_pos)

        for finger, curl_val in adjusted.items():
            if finger in self.hand.fingers_dict:
                motor_ids = self.hand.fingers_dict[finger]
                if finger == 'Thumb':
                    # Constant order: [CMC, MCP, IP]
                    open_m, adduct_m, in_m = curl_val  # open: MCP, adduct: CMC, in: IP
                    # CMC ← adduct (adduction/abduction)
                    target_pos[motor_ids[0]] = self._map_curl_to_position(adduct_m, motor_ids[0])
                    # MCP ← open (open/close)
                    target_pos[motor_ids[1]] = self._map_curl_to_position(open_m, motor_ids[1])
                    # IP ← in (distal flexion)
                    target_pos[motor_ids[2]] = self._map_curl_to_position(in_m, motor_ids[2])
                else:
                    # Constant order: [DIP/PIP, MCP]
                    distal_curl, mcp_curl = curl_val
                    target_pos[motor_ids[0]] = self._map_curl_to_position(distal_curl, motor_ids[0])
                    target_pos[motor_ids[1]] = self._map_curl_to_position(mcp_curl, motor_ids[1])
            else:
                print(f"Unknown finger: {finger}")

        # Apply safety bounds
        target_pos = self._apply_safety_checks(target_pos)

        # Apply smoothing if enabled
        if self.smoothing:
            target_pos = self._smooth_position(target_pos)

        # Rate-limited command sending
        self._rate_limited_send(target_pos)

        return target_pos

    def _apply_safety_checks(self, target_pos):
        """Apply safety bounds to ensure positions are within safe range"""
        safe_pos = np.copy(target_pos)

        for i in range(len(target_pos)):
            if self.hand_type == "right":
                min_pos = self.hand.tensioned_pos[i] + self.safety_margin
                max_pos = self.hand.curled_bound[i] - self.safety_margin
                safe_pos[i] = int(np.clip(target_pos[i], min_pos, max_pos))
            else:  # left hand
                min_pos = self.hand.curled_bound[i] + self.safety_margin
                max_pos = self.hand.tensioned_pos[i] - self.safety_margin
                safe_pos[i] = int(np.clip(target_pos[i], min_pos, max_pos))

        return safe_pos

    def _smooth_position(self, target_pos):
        """Apply exponential moving average smoothing to position commands"""
        self.position_history.append(target_pos)

        # Weighted average (recent frames have higher weight)
        weights = np.linspace(0.5, 1.5, len(self.position_history))
        weights = weights / np.sum(weights)

        smoothed = np.zeros_like(target_pos, dtype=float)
        for i, weight in enumerate(weights):
            smoothed += self.position_history[i] * weight

        return smoothed.astype(int)

    def _rate_limited_send(self, target_pos):
        """Send position commands at limited rate to avoid overloading bus"""
        current_time = time.time()
        if current_time - self.last_command_time >= self.command_interval:
            try:
                self.hand.set_pos(target_pos)
                self.last_command_time = current_time
            except Exception as e:
                print(f"Error sending position command: {str(e)}")

def main():
    """Main function"""
    print("="*60)
    print("RUKA Robotic Hand - MediaPipe Gesture Control")
    print("Tip: Place your hand in front of the camera to control the robotic hand with gestures")
    print("Press ESC to exit")
    print("="*60)

    # Check camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        print("   Please check:")
        print("   1. Camera connection")
        print("   2. No other application is using the camera")
        print("   3. Camera permissions are granted")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    try:
        # Initialize controller
        controller = HandController(hand_type='right', smoothing=True)

        # Initialize MediaPipe
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)

        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()

        print("\nSystem ready! Please place your hand in front of the camera...")

        while True:
            success, image = cap.read()
            if not success:
                print("Skipping empty frame")
                continue

            # Mirror image horizontally
            image = cv2.flip(image, 1)
            h, w, _ = image.shape

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            results = hands.process(image_rgb)

            # Draw results
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand skeleton
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Annotate landmark indices
                    for i, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.putText(image, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                    # Process gesture and control hand
                    try:
                        handedness = results.multi_handedness[0].classification[0].label if results.multi_handedness else 'Right'
                        target_pos = controller.process_hand_landmarks(hand_landmarks.landmark, handedness)

                        # Visualize: show curl percentage for each finger
                        for i, (finger, _) in enumerate(FINGER_MAPPING.items()):
                            curl = controller._calculate_finger_curl(hand_landmarks.landmark, FINGER_MAPPING[finger])
                            curl_percent = int(curl[0] * 100)

                            # Display curl percentage
                            cv2.putText(image, f"{finger}: {curl_percent}%",
                                      (20, 40 + i*30),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7, (0, 255, 0), 2)

                            # Draw progress bar
                            bar_width = int(curl[0] * 150)
                            cv2.rectangle(image, (120, 35 + i*30), (120 + 150, 55 + i*30), (50, 50, 50), -1)
                            cv2.rectangle(image, (120, 35 + i*30), (120 + bar_width, 55 + i*30), (0, 255, 0), -1)

                    except Exception as e:
                        print(f"Error processing gesture: {str(e)}")

            # Display FPS
            frame_count += 1
            if time.time() - start_time > 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            cv2.putText(image, f"FPS: {fps:.1f}", (w-120, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Display exit instruction
            cv2.putText(image, "Press ESC to exit", (20, h-20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Show image
            cv2.imshow('RUKA Hand Control', image)

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except RuntimeError as e:
        print(f"\nRuntime error: {str(e)}")
        print("   Please run calibration first: python calibrate_motors_modify.py -ht right")

    except Exception as e:
        print(f"\nUnhandled error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up resources...")
        try:
            if 'controller' in locals():
                controller.close()
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            print("Program exited safely")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()