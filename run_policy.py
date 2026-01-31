import torch
from pathlib import Path

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.robots.factory import make_robot
# Load trained policy
checkpoint_path = "./outputs/pusht_diffusion_v1/checkpoints/050000/pretrained_model"
print(f"Loading policy from {checkpoint_path}...")
policy = DiffusionPolicy.from_pretrained(checkpoint_path)
policy.eval()
policy = policy.to("cuda:0")
print("Policy loaded successfully!")

# Connect robot with camera
print("Connecting to robot...")
robot_config = {
    "robot_type": "so101_follower",
    "robot_path": "/dev/ttyACM0",
    "cameras": {
        "top": {
            "type": "opencv",
            "index_or_path": "/dev/v4l/by-id/usb-XIFT_HB_Camera_HU12345673-video-index0",
            "width": 640,
            "height": 480,
            "fps": 30
        }
    }
}
robot = make_robot(**robot_config)
robot.connect()
print("Robot connected!")

print("\nRunning policy inference... Press Ctrl+C to stop\n")
step = 0
try:
    while True:
        # Capture observation
        obs = robot.capture_observation()
        
        # Get action from policy
        with torch.no_grad():
            action = policy.select_action(obs)
        
        # Send action to robot
        robot.send_action(action)
        
        step += 1
        if step % 30 == 0:
            print(f"Step {step}")
            
except KeyboardInterrupt:
    print("\n\nStopping policy execution...")
finally:
    robot.disconnect()
    print("Robot disconnected. Done!")
