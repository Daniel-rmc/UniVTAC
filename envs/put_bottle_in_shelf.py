from ._base_task import *
import numpy as np
import os

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    adaptive_grasp_depth_threshold = 27.5

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 1
        cfg.sim.physics_material.static_friction = 1
        cfg.uipc_sim.contact.default_friction_ratio = 1
        super().__init__(cfg, mode, render_mode, **kwargs)

    def _debug_pose(self, stage):
        if os.environ.get("UNIVTAC_TASK_STAGE_DEBUG_LOG") != "1":
            return
        bottle_pose = self.bottle.get_pose()
        shelf_pose = self.shelf.get_pose()
        place_target = getattr(self, "place_target", None)
        rel_pose = None
        upright_dot = None
        if place_target is not None:
            rel_pose = bottle_pose.rebase(place_target)
            upright_dot = np.dot(rel_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))
        print(
            "UNIVTAC_TASK_STAGE_DEBUG put_bottle_in_shelf "
            f"stage={stage} step={self.step_count} plan_success={self.plan_success} "
            f"bottle_pose={bottle_pose.tolist()} shelf_pose={shelf_pose.tolist()} "
            f"place_target={place_target.tolist() if place_target is not None else None} "
            f"rel_pose={rel_pose.tolist() if rel_pose is not None else None} "
            f"upright_dot={float(upright_dot) if upright_dot is not None else None}"
        )
    
    def create_actors(self):
        base_pose = Pose([0.9, 0.0, 0.01], [1, 0, 0, 0])
        bottle_pose = Pose([0.5, 0.0, 0.01], [1, 0, 0, 0])

        self.shelf = self._actor_manager.add_from_usd_file(
            name='shelf',
            asset_path="Shelf.usd", 
            pose=base_pose,
        )
        self.bottle = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="BottleLift.usd",
            pose=bottle_pose
        )
        
    def _reset_actors(self):
        base_offset = self.create_noise([0.05, 0.0, 0.0])
        base_pose = Pose([0.9, 0.0, 0.01], [1, 0, 0, 0]).add_offset(base_offset)
        bottle_offset = self.create_noise([0.0, 0.03, 0.0])
        bottle_pose = Pose([0.5, 0.0, 0.01], [1, 0, 0, 0]).add_offset(bottle_offset)

        self.shelf.set_pose(base_pose)
        self.bottle.set_pose(bottle_pose)
 
    def pre_move(self):
        self.delay(10)
        self._debug_pose("pre_move_after_initial_delay")

        bottle_pose = self.bottle.get_pose()
        target_pose = bottle_pose.add_bias([0, 0, 0.11+0.01*self.rng.random()])
        self.grasp_noise = self.create_noise(euler=[0, np.pi/18, 0])
        target_pose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        ).add_offset(self.grasp_noise)
        grasp_idx = self.bottle.register_point(
            pose=target_pose,
            type='contact'
        )
        self.move(self.atom.grasp_actor(
            self.bottle, contact_point_id=grasp_idx, pre_dis=0.0, is_close=False
        ))
        self._debug_pose("pre_move_after_grasp_actor")
        
        base_pose = self.shelf.get_pose()
        self.place_target = base_pose.add_bias([-0.2, 0, 0.21])
        self._debug_pose("pre_move_after_place_target")
        self.move(self.atom.close_gripper())
        self._debug_pose("pre_move_after_close_gripper")

    def _play_once(self):
        lift_height = 0.15 + self.rng.uniform(0.0, 0.05)
        if os.environ.get("UNIVTAC_TASK_STAGE_DEBUG_LOG") == "1":
            print(f"UNIVTAC_TASK_STAGE_DEBUG put_bottle_in_shelf stage=play_lift_start lift_height={float(lift_height):.6f}")
        self.move(self.atom.move_by_displacement(
            z=lift_height
        ), constraint_pose=[1, 1, 1, 0, 1, 0])
        self._debug_pose("play_after_lift")
        self.move(self.atom.move_by_displacement(
            rpy=[0, -np.pi/3-0.2*self.rng.random(), 0]
        ), constraint_pose=[0, 0, 0, 0, 1, 0])
        self._debug_pose("play_after_rotate")

        self.gravity_rotate(self.bottle, target_vec=np.array([0, 0, 1]))
        self._debug_pose("play_after_gravity_rotate")
        self.move(self.atom.place_actor(
            self.bottle,
            pre_dis=0.01,
            dis=0.005,
            target_pose=self.place_target,
            is_open=False
        ), time_dilation_factor=0.3)
        self._debug_pose("play_after_place_actor")
        self.move(self.atom.open_gripper(0.5))
        self._debug_pose("play_after_open_gripper")
        self.delay(20, is_save=False)
        self._debug_pose("play_after_final_delay")

    def check_early_stop(self):
        min_depth = torch.min(self._tactile_manager.get_min_depth()).item()
        
        if min_depth < 20:
            self.metadata['early_stop'] = True
            self.metadata['min_depth'] = float(min_depth)
            return True
        return False

    def check_success(self):
        bottle_pose = self.bottle.get_pose().rebase(self.place_target)
        pos_ok = np.all(np.abs(bottle_pose.p) < np.array([0.02, 0.1, 0.02]))
        upright_dot = np.dot(bottle_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))
        upright_ok = upright_dot > 0.965 # 15°
        if os.environ.get("UNIVTAC_TASK_DEBUG_LOG") == "1":
            print(
                "UNIVTAC_TASK_DEBUG put_bottle_in_shelf "
                f"rel_pos={bottle_pose.p.tolist()} "
                f"abs_rel_pos={np.abs(bottle_pose.p).tolist()} "
                f"pos_ok={bool(pos_ok)} upright_dot={float(upright_dot):.6f} "
                f"upright_ok={bool(upright_ok)}"
            )
        return pos_ok and upright_ok
