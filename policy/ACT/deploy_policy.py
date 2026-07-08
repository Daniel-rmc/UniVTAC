import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import os
import cv2
import yaml
import numpy as np
import torch
from .act_policy import ACT
# from act_policy import ACT
from torchvision import transforms

class Policy(BasePolicy):
# class Policy:
    def __init__(self, args):
        """Initialize ACT policy for TacArena deployment"""
        # Construct checkpoint directory path
        self.train_config_name = os.environ.get('TRAIN_CONFIG', 'train_config')
        self.ep_num = os.environ.get('EP_NUM', '50')
        ckpt_dir = Path(__file__).parent / "act_ckpt" / f"act-{args['task_name']}" / f"{args['task_config']}-{self.ep_num}" / self.train_config_name
 
        self.task_name = args['task_name']
        with open(Path(__file__).parent.parent / 'task_settings.json', 'r') as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get('camera_type', 'head')
        self.camera_type = os.environ.get("UNIVTAC_ACT_CAMERA_TYPE", self.camera_type)
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")

        with open(Path(__file__).parent / f'{self.train_config_name}.yml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        temporal_agg_override = os.environ.get("UNIVTAC_ACT_TEMPORAL_AGG")
        if temporal_agg_override is not None:
            train_config["temporal_agg"] = temporal_agg_override not in ("0", "false", "False", "no", "NO")
            print(f"Using temporal_agg override from UNIVTAC_ACT_TEMPORAL_AGG: {train_config['temporal_agg']}")
        
        train_config.update({
            'task_name': f"sim-{args['task_name']}-{args['task_config']}-{self.ep_num}",
            'task_config': args['task_config'],
            'ckpt_dir': str(ckpt_dir),
            "seed": args.get('seed', 0),
            "num_epochs": 1
        })
        
        # Initialize ACT model (RoboTwin_Config=None for TacArena)
        self.model = ACT(train_config)
        print(f"ACT policy loaded from {ckpt_dir}")

    def encode_obs(self, observation):
        """
        Encode TacArena observation to ACT input format
        
        Input (TacArena):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_high": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        # Debug: observation structure validated
        # observation['embodiment']['joint'] contains joint state
        def camera_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        
        def tactile_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1)) # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            return img

        if self.camera_type == 'all':
            raw_cam_high = observation["observation"]["head"]["rgb"]
            raw_cam_wrist = observation["observation"]["wrist"]["rgb"]
            cam_high = camera_transform(observation["observation"]["head"]["rgb"])
            cam_wrist = camera_transform(observation["observation"]["wrist"]["rgb"])
        else:
            raw_cam_high = observation["observation"][self.camera_type]["rgb"]
            raw_cam_wrist = None
            cam_high = camera_transform(raw_cam_high)

        tactile_obs = observation["tactile"]
        left_key = "left_gsmini" if "left_gsmini" in tactile_obs else "left_tactile"
        right_key = "right_gsmini" if "right_gsmini" in tactile_obs else "right_tactile"
        raw_left_tac = tactile_obs[left_key]["rgb_marker"]
        raw_right_tac = tactile_obs[right_key]["rgb_marker"]
        left_tac = tactile_transform(raw_left_tac)
        right_tac = tactile_transform(raw_right_tac)
        if os.environ.get("UNIVTAC_POLICY_DEBUG_LOG") == "1" and not getattr(self, "_obs_debug_printed", False):
            def range_msg(name, tensor):
                arr = tensor.detach() if hasattr(tensor, "detach") else tensor
                return f"{name}_dtype={getattr(arr, 'dtype', type(arr))} {name}_min={float(arr.min()):.6f} {name}_max={float(arr.max()):.6f}"
            parts = [
                range_msg("raw_cam_high", raw_cam_high),
                range_msg("raw_left_tac", raw_left_tac),
                range_msg("raw_right_tac", raw_right_tac),
                range_msg("cam_high", cam_high),
                range_msg("left_tac", left_tac),
                range_msg("right_tac", right_tac),
            ]
            if raw_cam_wrist is not None:
                parts.append(range_msg("raw_cam_wrist", raw_cam_wrist))
                parts.append(range_msg("cam_wrist", cam_wrist))
            print("UNIVTAC_OBS_DEBUG " + " ".join(parts))
            self._obs_debug_printed = True
        
        # Extract joint positions (8D: 7 arm + 1 gripper)
        qpos = observation["embodiment"]["joint"][:8]

        ret = {
            "cam_high": cam_high,
            "tac_left": left_tac,
            "tac_right": right_tac,
            "qpos": qpos.cpu().numpy()
        }
        if self.camera_type == 'all':
            ret["cam_wrist"] = cam_wrist
        elif "cam_wrist" in getattr(self.model, "camera_names", []):
            ret["cam_wrist"] = cam_high
        return ret

    def eval(self, task, observation):
        """
        Evaluate ACT policy on TacArena task
        
        Args:
            task: TacArena BaseTask instance
            observation: Current observation from environment
        """
        
        # Get action from ACT model (returns (1, 8) numpy array)
        obs = self.encode_obs(observation)
        if self.model.t % 10 == 0 and os.environ.get("UNIVTAC_DISABLE_POLICY_SNAPSHOTS") != "1":
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        action = self.model.get_action(obs).reshape(-1)
        if os.environ.get("UNIVTAC_POLICY_DEBUG_LOG") == "1" and task.take_action_cnt % 25 == 0:
            qpos_arr = np.asarray(obs["qpos"])
            action_arr = np.asarray(action)
            delta_arr = action_arr - qpos_arr
            print(
                "UNIVTAC_POLICY_DEBUG "
                f"task={self.task_name} step={task.step_count} actions={task.take_action_cnt} "
                f"qpos_min={qpos_arr.min():.6f} qpos_max={qpos_arr.max():.6f} "
                f"action_min={action_arr.min():.6f} action_max={action_arr.max():.6f} "
                f"delta_max_abs={np.max(np.abs(delta_arr)):.6f} "
                f"qpos={np.array2string(qpos_arr, precision=5, separator=',', max_line_width=100000)} "
                f"action={np.array2string(action_arr, precision=5, separator=',', max_line_width=100000)} "
                f"delta={np.array2string(delta_arr, precision=5, separator=',', max_line_width=100000)}"
            )
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter)"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def save(self, img, t):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width-100, obs.height-60), f'{t:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'ACT_{self.task_name}_{self.train_config_name}.png')
