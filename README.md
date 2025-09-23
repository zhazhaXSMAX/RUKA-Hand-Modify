# RUKA Hand (Brief Overview)

This repository makes minimal changes on top of the official RUKA project. Specifically, we:
- Updated the motor calibration code: provide `calibrate_motors_modify.py` with adjusted calibration procedure and clearer CLI prompts.
- Added MediaPipe-based keypoint/gesture teleoperation: provide `teleop_gesture.py` to control the robotic hand via camera keypoints.

Acknowledgment & Attribution
- Built on and adapted from the official RUKA project:
  - GitHub: https://github.com/ruka-hand/RUKA
  - Project Page: https://ruka-hand.github.io/

For detailed usage and setup, please refer to the original RUKA documentation and inline code comments.
