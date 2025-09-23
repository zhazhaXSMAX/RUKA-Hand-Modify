# This script will move the the fingers with order automatically
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
            print(f"[复位] 电机{motor_id}移动到{target_pos}失败：{e}")
            return False

    def find_bound(self, motor_id):
        """
        使用单调二分（在“命令空间”而不是“反馈位置”上）寻找安全边界。
        - 始终用命令的 mid（而不是反馈的 pres_pos）更新上下界，避免因回弹/顺从导致区间塌陷。
        - 在 dwell 窗口内多次采样，取 f*current 的最大值判断是否越阈值（f=+1 右手，f=-1 左手）。
        - 输出中文、可读的日志。
        - 先在安全“开指”位测一段基线电流，阈值取 max(curr_lim, baseline+delta_margin)，避免“一上来就越阈”。
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

        # 基线电流测量（在“开指”侧的安全位置，短暂采样）
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
            # 取中位数抗异常
            baseline = float(np.median(baseline_samples))
            # 阈值采用“绝对最小阈值”和“基线+裕度”的较大者
            # 若出现饱和读数（如999），用分位数替代，进一步抑制异常
            if np.percentile(baseline_samples, 95) > 900:
                baseline = float(np.percentile(baseline_samples, 70))
            thresh = max(thresh, int(baseline + self.delta_margin))
        else:
            baseline = None

        if self.testing or self.verbose:
            if baseline is not None:
                print(f"\n[标定] >>> 开始 | 电机={motor_id} | 基线={baseline:.0f} | 阈值={thresh} | 停留={dwell:.1f}s | 搜索区间=[{POS_MIN_CMD},{POS_MAX_CMD}]")
            else:
                print(f"\n[标定] >>> 开始 | 电机={motor_id} | 阈值={thresh} | 停留={dwell:.1f}s | 搜索区间=[{POS_MIN_CMD},{POS_MAX_CMD}] (基线采样失败)")
            print("[标定] 迭代  区间[s_low,s_high]  试探pos  实测pos  峰值电流  判定")

        iters = 0
        while s_low < s_high and iters < self.max_iters:
            mid_s = (s_low + s_high) // 2
            com_pos = mid_s if is_right else (4095 - mid_s)
            com_pos = int(max(POS_MIN_CMD, min(POS_MAX_CMD, com_pos)))

            pos[motor_id - 1] = com_pos
            self.hand.set_pos(pos)

            t1 = time.time()
            # 忽略起动瞬间的电流尖峰：先给一个“稳定时间窗”，只对其后的样本做统计
            settle = min(0.35, dwell * 0.3)  # 前 0.35s 或 30% 的时间作为稳定过渡
            samples = []
            pres_pos_end = None
            while time.time() - t1 < dwell:
                cur = self.hand.read_single_cur(motor_id)
                if cur is not None:
                    if time.time() - t1 >= settle:
                        samples.append(f * cur)
                pres_pos_end = int(self.hand.read_pos()[motor_id - 1])
                time.sleep(0.05)

            # 使用分位数而非最大值，抑制尖峰导致的误判
            if samples:
                metric_val = float(np.percentile(samples, 80))  # 稍偏“保守”的稳态电流
            else:
                metric_val = -1e9

            crossed = (metric_val >= thresh)
            if self.testing or self.verbose:
                verdict = "越阈" if crossed else "未越"
                print(f"[标定] {iters:03d}  [{s_low:4d},{s_high:4d}]   {com_pos:7d}  {pres_pos_end:7d}   {int(metric_val):7d}  {verdict}")

            if crossed:
                s_high = mid_s
            else:
                s_low = mid_s + 1

            # 当靠近边界、仍未越阈时，除了延长 dwell，还可小步提高阈值以避免尖峰干扰
            if (not crossed) and (s_high - s_low) < 40 and dwell >= self.dwell_max:
                # 每次轻微增加 5%，不向下截断，避免阈值异常降低
                thresh = int(thresh * 1.05)

            dwell = min(self.dwell_max, dwell + self.dwell_step)

            iters += 1

        s_bound = s_low
        bound_pos = s_bound if is_right else (4095 - s_bound)
        bound_pos = int(max(POS_HARD_MIN, min(POS_HARD_MAX, bound_pos)))

        if self.testing or self.verbose:
            converged = (s_low >= s_high) or (iters >= self.max_iters)
            print(f"[标定] <<< 结束 | 电机={motor_id} | 结果pos={bound_pos} | 迭代={iters} | 收敛={converged}\n")

        return bound_pos

    def find_curled(self):
        curled = np.zeros(len(self.motor_ids), dtype=int)
        for i, mid in enumerate(self.motor_ids):
            curled[i] = int(self.find_bound(mid))
            # 每个电机标定完，立即把该电机从极限位挪回初始位（右手=100，左手=4000）
            init_pos = 100 if (self.hand.hand_type == "right") else 4000
            print(f"[复位] 电机{mid} 标定完成，移动到初始位 {init_pos} ...")
            self._move_single_motor(mid, init_pos, traj_len=50, sleep_time=0.01)
        return curled

    def estimate_tensioned_from_curled(self, curled):
        """
        先用固定范围（1100）从 curled 估计 tensioned，随后夹紧到安全范围，避免出现负数或超限。
        """
        f = 1 if self.hand.hand_type == "right" else -1
        offset = 1100
        est = [int(x - f * offset) for x in curled]
        # 统一夹紧，避免减成负数或超过上限
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
        # 标定完成后复位到一个安全的张力位（用估计值，已做边界夹紧）
        try:
            t_init = self.estimate_tensioned_from_curled(curled)
            curr_pos = self.hand.read_pos()
            print("复位：移动到估计张力位（避免停在边界附近）...")
            move_to_pos(curr_pos=curr_pos, des_pos=t_init, hand=self.hand, traj_len=60)
            print("复位完成。")
        except Exception as e:
            print(f"复位失败（curl阶段），但不影响数据已保存：{e}")

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
        # 张力标定结束后，复位到最终张力位
        try:
            curr_pos = self.hand.read_pos()
            print("复位：移动到最终张力位...")
            move_to_pos(curr_pos=curr_pos, des_pos=t_refined, hand=self.hand, traj_len=60)
            print("复位完成。")
        except Exception as e:
            print(f"复位失败（tension阶段），但不影响数据已保存：{e}")


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
    # 关闭手的连接
    try:
        calibrator.hand.close()
    except Exception as e:
        print(f"关闭手失败：{e}")
