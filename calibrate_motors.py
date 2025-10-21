"""
RUKA Hand Motor Calibration

Source: Derived from the original RUKA calibration implementation (calibrate_motors.py) by the upstream contributors.
Modifications: English logs/docs and minor UX refinements; core calibration logic preserved.
Acknowledgements: Full credit to the original authors of the RUKA project. See LICENSE and README for details.
"""
import os
import sys
import time

import numpy as np

from ruka_hand.control.hand import *
from ruka_hand.utils.file_ops import get_repo_root
from ruka_hand.utils.trajectory import move_to_pos


# Cross-platform single key reader
if sys.platform.startswith("win"):
    import msvcrt

    def get_key():
        """Return standardized key tokens: 'UP','DOWN','LEFT','RIGHT','ENTER','q' on Windows."""
        ch = msvcrt.getch()
        # Handle Ctrl+C gracefully
        if ch in (b"\x03",):
            raise KeyboardInterrupt
        # Enter
        if ch in (b"\r", b"\n"):
            return "ENTER"
        # ASCII letters
        try:
            s = ch.decode("utf-8").lower()
            if s == "q":
                return "q"
        except Exception:
            pass
        # Arrow keys come as two bytes: b"\xe0" or b"\x00" then code
        if ch in (b"\xe0", b"\x00"):
            ch2 = msvcrt.getch()
            code = ch2[0]
            if code == 72:
                return "UP"
            if code == 80:
                return "DOWN"
            if code == 75:
                return "LEFT"
            if code == 77:
                return "RIGHT"
        return None
else:
    import termios
    import tty

    def get_key():
        """Return standardized key tokens on POSIX: 'UP','DOWN','LEFT','RIGHT','ENTER','q'."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch.lower() == "q":
                return "q"
            if ch == "\x1b":
                # escape sequence
                seq = sys.stdin.read(2)
                if seq == "[A":
                    return "UP"
                if seq == "[B":
                    return "DOWN"
                if seq == "[C":
                    return "RIGHT"
                if seq == "[D":
                    return "LEFT"
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class HandCalibrator:
    def __init__(
        self,
        data_save_dir,
        hand_type,
        curr_lim=50,
        testing=False,
        motor_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        dwell=2.0,
        dwell_max=5.0,
        dwell_step=0.5,
        curr_lim_max=300,
        curr_lim_step=50,
        tol=10,
        max_iters=200,
        verbose=True,
        delta_margin=80,
    ):
        self.hand = Hand(hand_type)
        self.curr_lim = curr_lim
        self.testing = testing
        self.motor_ids = motor_ids
        self.data_save_dir = data_save_dir  # directory only
        # new calibration tuning params
        self.dwell_base = float(dwell)
        self.dwell_max = float(dwell_max)
        self.dwell_step = float(dwell_step)
        self.curr_lim_max = int(curr_lim_max)
        self.curr_lim_step = int(curr_lim_step)
        self.tol = int(tol)
        self.max_iters = int(max_iters)
        self.verbose = bool(verbose)
        self.delta_margin = int(delta_margin)
        # Save paths
        self.curled_path = os.path.join(
            self.data_save_dir, f"{hand_type}_curl_limits.npy"
        )
        self.tension_path = os.path.join(
            self.data_save_dir, f"{hand_type}_tension_limits.npy"
        )

    def _move_single_motor(self, motor_id: int, target_pos: int, traj_len: int = 40, sleep_time: float = 0.01):
        try:
            curr = np.array(self.hand.read_pos(), dtype=int)
            des = curr.copy()
            des[motor_id - 1] = int(target_pos)
            move_to_pos(curr_pos=curr, des_pos=des, hand=self.hand, traj_len=traj_len, sleep_time=sleep_time)
            return True
        except Exception as e:
            print(f"[Reset] Motor {motor_id} failed to move to {target_pos}: {e}")
            return False

    def find_bound(self, motor_id):
        """
        Use monotonic binary search (in command space, not feedback position) to find safe boundary.
        - Always update bounds using commanded mid (not feedback pres_pos) to avoid collapse due to compliance/rebound.
        - Sample multiple times within dwell window; use max(f*current) to determine if threshold is crossed (f=+1 right, f=-1 left).
        - Output readable logs in English.
        - First, measure baseline current at a safe "open finger" position to set threshold as max(curr_lim, baseline+delta_margin), avoiding immediate threshold crossing.
        """
        is_right = (self.hand.hand_type == "right")
        f = 1 if is_right else -1

        POS_MIN_CMD = 100
        POS_MAX_CMD = 4000
        POS_HARD_MIN = 10
        POS_HARD_MAX = 4090

        s_low = POS_MIN_CMD if is_right else (4095 - POS_MAX_CMD)
        s_high = POS_MAX_CMD if is_right else (4095 - POS_MIN_CMD)

        pos = np.array(self.hand.read_pos())
        thresh = int(self.curr_lim)
        dwell = float(self.dwell_base)

        # Baseline current measurement (at safe "open finger" position, brief sampling)
        baseline_pos = POS_MIN_CMD + 200 if is_right else POS_MAX_CMD - 200
        baseline_pos = int(max(POS_MIN_CMD, min(POS_MAX_CMD, baseline_pos)))
        pos[motor_id - 1] = baseline_pos
        self.hand.set_pos(pos)
        baseline_dur = min(0.6, dwell)
        t0 = time.time()
        baseline_samples = []
        while time.time() - t0 < baseline_dur:
            cur0 = self.hand.read_single_cur(motor_id)
            if cur0 is not None:
                baseline_samples.append(f * cur0)
            time.sleep(0.05)
        if baseline_samples:
            # Use median to reject outliers
            baseline = float(np.median(baseline_samples))
            # Threshold is max of "absolute minimum" and "baseline + margin"
            # If saturated readings (e.g., 999), use percentile to further suppress anomalies
            if np.percentile(baseline_samples, 95) > 900:
                baseline = float(np.percentile(baseline_samples, 70))
            thresh = max(thresh, int(baseline + self.delta_margin))
        else:
            baseline = None

        if self.testing or self.verbose:
            if baseline is not None:
                print(f"\n[Calibration] >>> Start | Motor={motor_id} | Baseline={baseline:.0f} | Threshold={thresh} | Dwell={dwell:.1f}s | Search range=[{POS_MIN_CMD},{POS_MAX_CMD}]")
            else:
                print(f"\n[Calibration] >>> Start | Motor={motor_id} | Threshold={thresh} | Dwell={dwell:.1f}s | Search range=[{POS_MIN_CMD},{POS_MAX_CMD}] (Baseline sampling failed)")
            print("[Calibration] Iter   Range[s_low,s_high]   Try pos   Actual pos   Peak current   Decision")

        iters = 0
        while s_low < s_high and iters < self.max_iters:
            mid_s = (s_low + s_high) // 2
            com_pos = mid_s if is_right else (4095 - mid_s)
            com_pos = int(max(POS_MIN_CMD, min(POS_MAX_CMD, com_pos)))

            pos[motor_id - 1] = com_pos
            self.hand.set_pos(pos)

            t1 = time.time()
            # Ignore startup current spike: give a "settling window", only sample after
            settle = min(0.35, dwell * 0.3)  # First 0.35s or 30% as settling time
            samples = []
            pres_pos_end = None
            while time.time() - t1 < dwell:
                cur = self.hand.read_single_cur(motor_id)
                if cur is not None:
                    if time.time() - t1 >= settle:
                        samples.append(f * cur)
                pres_pos_end = int(self.hand.read_pos()[motor_id - 1])
                time.sleep(0.05)

            # Use percentile instead of max to suppress spike-induced false positives
            if samples:
                metric_val = float(np.percentile(samples, 80))  # Slightly "conservative" steady-state current
            else:
                metric_val = -1e9

            crossed = (metric_val >= thresh)
            if self.testing or self.verbose:
                verdict = "Crossed" if crossed else "Safe"
                print(f"[Calibration] {iters:03d}  [{s_low:4d},{s_high:4d}]   {com_pos:7d}  {pres_pos_end:7d}   {int(metric_val):7d}  {verdict}")

            if crossed:
                s_high = mid_s
            else:
                s_low = mid_s + 1

            # When approaching boundary without crossing, besides increasing dwell, slightly raise threshold to avoid spike interference
            if (not crossed) and (s_high - s_low) < 40 and dwell >= self.dwell_max:
                # Increase by 5% each time, no floor, avoid abnormal threshold reduction
                thresh = int(thresh * 1.05)

            dwell = min(self.dwell_max, dwell + self.dwell_step)

            iters += 1

        s_bound = s_low
        bound_pos = s_bound if is_right else (4095 - s_bound)
        bound_pos = int(max(POS_HARD_MIN, min(POS_HARD_MAX, bound_pos)))

        if self.testing or self.verbose:
            converged = (s_low >= s_high) or (iters >= self.max_iters)
            print(f"[Calibration] <<< End | Motor={motor_id} | Result pos={bound_pos} | Iterations={iters} | Converged={converged}\n")

        return bound_pos

    def find_curled(self):
        curled = np.zeros(len(self.motor_ids), dtype=int)
        for i, mid in enumerate(self.motor_ids):
            curled[i] = int(self.find_bound(mid))
            # After each motor calibration, immediately move it back to initial position (right=100, left=4000)
            init_pos = 100 if (self.hand.hand_type == "right") else 4000
            print(f"[Reset] Motor {mid} calibration complete, moving to initial position {init_pos} ...")
            self._move_single_motor(mid, init_pos, traj_len=50, sleep_time=0.01)
        return curled

    def estimate_tensioned_from_curled(self, curled):
        """
        First estimate tensioned from curled using fixed offset (1100), then clamp to safe range to avoid negatives or overflow.
        """
        f = 1 if self.hand.hand_type == "right" else -1
        offset = 1100
        est = [int(x - f * offset) for x in curled]
        # Clamp uniformly to avoid negative or exceeding limits
        est = [max(10, min(4090, v)) for v in est]
        return np.array(est, dtype=int)

    def interactive_refine_tensioned(self, tensioned_init, step=10):
        """
        Iterate through motors; arrows adjust ±step; Enter confirms & advances.
        Up/Right arrows: +step, Down/Left arrows: -step relative to servo direction.
        """
        # Ensure we can set positions for all 11 motors
        current_pos = np.array(self.hand.read_pos())

        # Start each motor at its initial tensioned guess
        tensioned = tensioned_init.copy()

        # Direction factor: moves in the "open/tension" direction
        f = 1 if self.hand.hand_type == "right" else -1

        print("\n--- Tensioned Calibration ---")
        print("Use ↑/→ to increase, ↓/← to decrease (±10). Press Enter to confirm motor.\n")

        for mid in self.motor_ids:
            idx = mid - 1
            # Move all motors to safe neutral, then set current motor to its guess
            pos = current_pos.copy()
            pos[idx] = tensioned[idx]
            self.hand.set_pos(pos)
            time.sleep(0.2)

            while True:
                print(f"[Motor {mid}] Current candidate: {pos[idx]}")
                print("Adjust with arrows, Enter to save, 'q' to abort this motor.")

                k = get_key()

                if k == "ENTER":  # Enter confirms
                    tensioned[idx] = int(pos[idx])
                    print(f"Saved Motor {mid} tensioned = {tensioned[idx]}\n")
                    break
                elif k in ("UP", "RIGHT"):
                    pos[idx] = max(min(pos[idx] + step * f, 4090), 10)
                    self.hand.set_pos(pos)
                elif k in ("DOWN", "LEFT"):
                    pos[idx] = max(min(pos[idx] - step * f, 4090), 0)
                    self.hand.set_pos(pos)
                elif k == "q":
                    print(f"Aborted adjustments for Motor {mid}; keeping {tensioned[idx]}\n")
                    break
                else:
                    # ignore other keys
                    pass

        print("Final tensioned array:\n", tensioned)
        return tensioned.astype(int)

    def save_curled_limits(self):
        curled = self.find_curled()
        np.save(self.curled_path, curled)
        print(f"Saved curled limits to {self.curled_path}")
        # After calibration, reset to a safe tensioned position (using estimated values, already clamped)
        try:
            t_init = self.estimate_tensioned_from_curled(curled)
            curr_pos = self.hand.read_pos()
            print("Reset: moving to estimated tensioned position (to avoid staying near boundary)...")
            move_to_pos(curr_pos=curr_pos, des_pos=t_init, hand=self.hand, traj_len=60)
            print("Reset complete.")
        except Exception as e:
            print(f"Reset failed (curl phase), but data is saved: {e}")

    def save_tensioned_limits(self):
        """
        Runs the interactive tension pass.
        Requires curled to exist (either just measured or previously saved).
        """
        if os.path.exists(self.curled_path):
            curled = np.load(self.curled_path)
        else:
            print("Curled limits not found; running curled calibration first...")
            curled = self.find_curled()
            np.save(self.curled_path, curled)
            print(f"Saved curled limits to {self.curled_path}")

        t_init = self.estimate_tensioned_from_curled(curled)
        t_refined = self.interactive_refine_tensioned(t_init, step=10)
        np.save(self.tension_path, t_refined)
        print(f"Saved tensioned limits to {self.tension_path}")
        # After tension calibration, reset to final tensioned position
        try:
            curr_pos = self.hand.read_pos()
            print("Reset: moving to final tensioned position...")
            move_to_pos(curr_pos=curr_pos, des_pos=t_refined, hand=self.hand, traj_len=60)
            print("Reset complete.")
        except Exception as e:
            print(f"Reset failed (tension phase), but data is saved: {e}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate RUKA hand motors")
    parser.add_argument(
        "-ht",
        "--hand-type",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Type of hand to calibrate (right or left)",
    )
    parser.add_argument(
        "--testing",
        type=bool,
        default=True,
        help="Enable testing mode with debug prints",
    )
    parser.add_argument(
        "--curr-lim", type=int, default=100, help="Current limit for calibration"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["curl", "tension", "both"],
        default="curl",
        help="Which calibration(s) to run: curled only, tension only, or both (default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = get_repo_root()
    # Save to motor_limits to align with Hand class expectations
    save_dir = os.path.join(repo_root, "motor_limits")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    calibrator = HandCalibrator(
        data_save_dir=save_dir,
        hand_type=args.hand_type,
        curr_lim=args.curr_lim,
        testing=args.testing,
    )

    if args.mode in ("curl", "both"):
        calibrator.save_curled_limits()
    if args.mode in ("tension", "both"):
        calibrator.save_tensioned_limits()
    # Close hand connection
    try:
        calibrator.hand.close()
    except Exception as e:
        print(f"Failed to close hand: {e}")
