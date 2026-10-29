import sys
import os
import re
import importlib
import subprocess

# ANSI colors for terminal formatting
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_status(status, message):
    if status == "OK":
        print(f"[{GREEN} OK {RESET}] {message}")
    elif status == "FAIL":
        print(f"[{RED}FAIL{RESET}] {message}")
    elif status == "WARN":
        print(f"[{YELLOW}WARN{RESET}] {message}")

def check_global_dependencies():
    print("========================================")
    print("   Global Dependency Check   ")
    print("========================================\n")

    print("----- 1. Python Environment -----")
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver >= (3, 10):
        print_status("OK", f"Python version\t: {py_str}.")
    else:
        print_status("WARN", f"Python version\t: {py_str} (3.10+ recommended).")

    # 2. Check Python Packages
    print("\n----- 2. Python Packages -----")
    packages1 = {
        "numpy": "NumPy\t\t",
        "matplotlib": "Matplotlib\t",
    }

    installed = {}
    for module, name in packages1.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "Installed")
            print_status("OK", f"{name}: v{version}")
            installed[module] = True
        except ImportError:
            print_status("FAIL", f"{name} is missing! (pip install {module})")
            installed[module] = False

    # 3. Check CV Packages
    print("\n----- 3. CV Packages -----")
    packages2 = {
        "cv2": "OpenCV\t\t",
        "ultralytics": "Ultralytics\t",
    }

    installed = {}
    for module, name in packages2.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "Installed")
            print_status("OK", f"{name}: v{version}")
            installed[module] = True
        except ImportError:
            print_status("FAIL", f"{name} is missing! (pip install {module})")
            installed[module] = False

    # 4. Check RL Packages
    print("\n----- 4. RL Packages -----")
    packages3 = {
        "torch": "PyTorch\t\t",
        "pygame": "Pygame\t\t",
        "gymnasium": "Gymnasium\t",
        "stable_baselines3": "Stable-Baselines3",
        "mujoco": "MuJoCo Physics\t",
        "onnxruntime": "ONNX Runtime\t",
    }
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    installed = {}
    for module, name in packages3.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "Installed")
            print_status("OK", f"{name}: v{version}")
            installed[module] = True
        except ImportError:
            print_status("FAIL", f"{name} is missing! (pip install {module})")
            installed[module] = False
    
    # 5. Check OpenCV Build (GStreamer & GTK)
    print("\n----- 5. OpenCV Build Configuration -----")    
    import cv2
    info = cv2.getBuildInformation()        
    # Check GStreamer
    if re.search(r'GStreamer:\s*YES', info):
        print_status("OK", "OpenCV GStreamer support is ENABLED.")
    else:
        print_status("FAIL", "OpenCV GStreamer support is MISSING. (VideoWriter will fail)")            
    # Check GTK (GUI)
    if re.search(r'GTK.?:\s*YES', info, re.IGNORECASE):
        print_status("OK", "OpenCV GTK (Window) support is ENABLED.")
    else:
        print_status("FAIL", "OpenCV GTK support is MISSING. (cv.imshow will crash)")
    
    # 6. Check System GStreamer
    print("\n----- 6. System GStreamer Tools -----")
    try:
        subprocess.run(["gst-launch-1.0", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print_status("OK", "Ubuntu GStreamer CLI tools are installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("FAIL", "System GStreamer tools are missing! (Run: sudo apt-get install gstreamer1.0-tools)")

    # 7. Check PyTorch CUDA Acceleration
    print("\n----- 7. Hardware Acceleration -----")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_status("OK", f"PyTorch CUDA active.\n       GPU: {gpu_name}")
        else:
            print_status("WARN", "PyTorch CUDA is NOT available. YOLO inference will run on CPU.")
    except ImportError:
        print_status("FAIL", "Skipping CUDA check because PyTorch is not installed.")

    # 8. SB3 $ Gymnasium
    print("\n----- 8. SB3 & Gym Integration -----")
    if installed.get("gymnasium") and installed.get("stable_baselines3"):
        import gymnasium as gym
        from stable_baselines3 import PPO

        try:
            env_name = "Humanoid-v5"
            env = gym.make(env_name)
            model = PPO("MlpPolicy", env, device="cpu", verbose=0)
            print_status(
                "OK",
                f"Environment initialized: {env_name} + {model.__class__.__name__}.",
            )
            env.close()
        except Exception as e:
            print_status(
                "FAIL", f"environment initialization failed: {e}"
            )

    # 9. Low-Level MuJoCo Engine & GL Check
    print("\n----- 9. MuJoCo Dynamic Engine & Rendering -----")
    if installed.get("mujoco"):
        import mujoco

        # Check C++ MJCF Parser
        test_xml = '<mujoco><worldbody><body name="b"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        try:
            m = mujoco.MjModel.from_xml_string(test_xml)
            d = mujoco.MjData(m)
            mujoco.mj_step(m, d)
            print_status("OK", "C++ engine & MJCF active.")
        except Exception as e:
            print_status("FAIL", f"MuJoCo engine failure: {e}")

        # Check Offscreen Rendering Engine
        try:
            renderer = mujoco.Renderer(m, height=240, width=320)
            renderer.update_scene(d)
            _ = renderer.render()
            gl_backend = os.environ.get("MUJOCO_GL", "default")
            print_status(
                "OK", f"Offscreen Renderer functional (Backend: {gl_backend})."
            )
            renderer.close()
        except Exception as e:
            print_status(
                "WARN",
                f"MuJoCo rendering warning: {e}. Try setting 'export MUJOCO_GL=egl' or 'glfw'.",
            )
        
    print("\n========================================")
    print("Diagnosis Complete.")

if __name__ == "__main__":
    check_global_dependencies()