import wandb
import numpy as np
import torch
import collections
import tqdm
from gym import spaces
from termcolor import cprint
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from diffusion_policy_3d.env.realworld.realworld_wrapper import RealWorldEnv

class RealWorldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=1024,
                 robot_ip="192.168.20.124"
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    RealWorldEnv(task_name=task_name, device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points, robot_ip=robot_ip)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        #all_traj_rewards = []
        all_success_rates = []
        env = self.env

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in RealWorld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            # Start rollout
            obs = env.reset()
            policy.reset()

            done = False
            #traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
            
                with torch.no_grad():
                    obs_dict_input = {
                        'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                        'agent_pos': obs_dict['agent_pos'].unsqueeze(0),
                    }   
                    
                    action_dict = policy.predict_action(obs_dict_input)

                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
                    action = np_action_dict['action'].squeeze(0)

                    obs, done = env.step(action)

                    #traj_reward += reward
                    done = np.all(done)
                    #is_success = is_success or max(info.get('success', [False]))

            all_success_rates.append(is_success)
            #all_traj_rewards.append(traj_reward)

        log_data = {
            #'mean_traj_rewards': np.mean(all_traj_rewards),
            'mean_success_rates': np.mean(all_success_rates),
            'test_mean_score': np.mean(all_success_rates),
        }

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        return log_data
