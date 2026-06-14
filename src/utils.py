import os
import h5py
import torch
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader

# Force spawn start method for safe multiprocessing with HDF5 on macOS
import sys
if sys.platform == 'darwin':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass




import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = True

        # RAM Pre-loading (Pre-load the entire dataset into memory to avoid disk latency)
        print(f"Pre-loading {len(self.episode_ids)} episodes into memory...")
        self.preloaded_data = []
        dataset_path = os.path.join(self.dataset_dir, 'episodes_500_tuple.hdf5')
        
        with h5py.File(dataset_path, 'r') as root:
            for ep_id in self.episode_ids:
                ep_group = root[f'episode_{ep_id}']
                self.preloaded_data.append({
                    'qpos': ep_group['qpos'][()],
                    'action': ep_group['actions'][()],
                    'episode_len': ep_group['actions'].shape[0]
                })

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        # O(1) Memory Indexing
        data = self.preloaded_data[index]
        qpos_arr = data['qpos']
        action_arr = data['action']
        episode_len = data['episode_len']
        
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        
        qpos = qpos_arr[start_ts]
        action = action_arr[start_ts:]
        action_len = episode_len - start_ts

        padded_action = np.zeros((episode_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # Lazy open of HDF5 per worker process
        if not hasattr(self, 'file_handle'):
            dataset_path = os.path.join(self.dataset_dir, 'episodes_500_tuple.hdf5')
            self.file_handle = h5py.File(dataset_path, 'r')
            
        ep_id = self.episode_ids[index]
        image_uint8 = self.file_handle[f'episode_{ep_id}']['images'][start_ts]
        image_data = torch.from_numpy(image_uint8).float() / 255.0
        
        # CPU-side resizing to match configured resolution
        from config import CONFIG
        target_size = CONFIG.get("image_size")
        if target_size is not None and target_size != (480, 640):
            image_data = torch.nn.functional.interpolate(
                image_data,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # normalize action and qpos
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # task_id detection based on episode attributes
        task_id = 0
        ep_group = self.file_handle[f'episode_{ep_id}']
        if 'task_id' in ep_group.attrs:
            task_id = int(ep_group.attrs['task_id'])
            
        # Generate constant task semantic embedding based on task_id
        seed_val = 42 if task_id == 0 else 1000 + task_id
        state = torch.random.get_rng_state()
        torch.manual_seed(seed_val)
        slow_semantic = torch.randn(512)
        torch.random.set_rng_state(state)

        return image_data, qpos_data, action_data, is_pad, slow_semantic


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    dataset_path = os.path.join(dataset_dir, 'episodes_500_tuple.hdf5')
    with h5py.File(dataset_path, 'r') as root:
        # Dynamically determine the available episodes in the file
        available_episodes = 0
        while f'episode_{available_episodes}' in root:
            available_episodes += 1
        
        num_to_read = min(num_episodes, available_episodes) if available_episodes > 0 else num_episodes
        
        for episode_idx in range(num_to_read):
            ep_group = root[f'episode_{episode_idx}']
            qpos = ep_group['qpos'][()]
            action = ep_group['actions'][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    
    # Golden data loading parameters for Apple Silicon Unified Memory:
    # num_workers=4 (efficient multiprocessing), pin_memory=False (unified memory doesn't need pinned RAM page locks)
    # prefetch_factor=2 (ahead queue), persistent_workers=True (keeps workers active across epoch boundaries)
    # Note: Using spawn start method (configured at the top of utils.py) allows safe HDF5 multiprocessing on macOS.
    import sys
    if sys.platform == 'darwin':
        # Multiprocessing with MPS on macOS is highly prone to queue deadlocks during backpropagation.
        # We use sequential loading (num_workers=0) which runs safely and fast with CPU-side resizing.
        num_workers = 0
        prefetch_factor = None
        persistent_workers = False
    else:
        num_workers = 4
        prefetch_factor = 2
        persistent_workers = True

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        pin_memory=False, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size_val, 
        shuffle=True, 
        pin_memory=False, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def make_policy(policy_class, policy_config):
    from policy import ACTPolicy, CNNMLPPolicy
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return policy

def make_optimizer(policy_class, policy):
    from policy import ACTPolicy, CNNMLPPolicy
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return optimizer

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def get_image(images, camera_names, device='cpu'):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pos2pwm(pos:np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """ 
    return (pos / 3.14 + 1.) * 2048
    
def pwm2pos(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi, pi]
    """
    return (pwm / 2048 - 1) * 3.14

def pwm2vel(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities 
    """
    return pwm * 3.14 / 2048

def vel2pwm(vel:np.ndarray) -> np.ndarray:
    """
    :param vel: numpy array of rad/s joint velocities
    :return: numpy array of pwm/s joint velocities
    """
    return vel * 2048 / 3.14
    
def pwm2norm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of pwm values in range [0, 4096]
    :return: numpy array of values in range [0, 1]
    """
    return x / 4096
    
def norm2pwm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of values in range [0, 1]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return x * 4096