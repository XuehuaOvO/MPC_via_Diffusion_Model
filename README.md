# Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models


Carvalho, J.; Le, A.T.; Baierl, M.; Koert, D.; Peters, J. (2023). **_Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models_**, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

[<img src="https://img.shields.io/badge/arxiv-%23B31B1B.svg?&style=for-the-badge&logo=arxiv&logoColor=white" />](https://arxiv.org/abs/2308.01557)


---

## Installation for cart pole diffusion

Pre-requisites:
- Ubuntu 20.04
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

Clone this repository with
```bash
cd ~
git clone https://github.com/XuehuaOvO/cart_pole_diffusion_based_on_MPD.git
cd cart_pole_diffusion_based_on_MPD
git submodule update --init --recursive # Initialize and update the submodules
```

Download [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) via wget for remote container and extract it under `deps/isaacgym`
```bash
wget https://developer.nvidia.com/isaac-gym-preview-4 
mv ~/isaac-gym-preview-4 /root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/deps/isaac-gym-preview-4
cd ~/cart_pole_diffusion_based_on_MPD/deps
tar -xvf isaac-gym-preview-4
```

Run the bash setup script to install everything.
```bash
cd ~/cart_pole_diffusion_based_on_MPD
bash setup.sh
```

Extra pkg for cart pole diffusion:
```bash
cd ~/cart_pole_diffusion_based_on_MPD
bash setup.sh
conda install -c conda-forge control slycot 
conda install conda-forge::casadi
```

Possible Errors:

If Building wheel for hydra (setup.py) ... error, ...., error: command 'gcc' failed: No such file or directory
```bash
conda install gcc
```

If subprocess.CalledProcessError: Command '['which', 'c++']' returned non-zero exit status 1.
```bash
apt install g++
```

If AttributeError: module 'numpy' has no attribute 'float'
```bash
pip install numpy==1.23.5
```

## linear mpc data collecting
1. Collecting noisy data (only the initial range of position and theta can be set, initial x_dot and theta_dot are always 0. Some parameters and paths in noisy_data_collecting.py should be set manually): 
```bash
conda activate mpd
cd scripts/mpc_data_collecting
python noisy_data_collecting.py
```

2. Collecting data with 4 DoF initial range (Some parameters and paths in 4DoF_data_collecting.py should be set manually):
```bash
conda activate mpd
cd scripts/mpc_data_collecting
python 4DoF_data_collecting.py
```

## cart pole model training
Training Data Path and File Name Setting:
In cart_pole_u.py, 
```python
DATASET_BASE_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data' # training data path of the training data and condition data files

U_DATA_NAME = 'u_tensor_420000-8-1.pt' # training data file name
X0_CONDITION_DATA_NAME = 'x0_tensor_420000-4.pt' # condition data file name
```

Training Launcg setting:
In cart_pole_launch.py,
```python
# training data folder
DATASET_SUBDIR = 'CartPole-LMPC' # the folder of the training data files (location: /root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data/CartPole-LMPC)

# training data amount
TRAINING_DATA_AMOUNT = 420000

# learning parameters
BATCH_SIZE = 512
LEARNING_RATE = 3e-3

EPOCHES = 300 # times that the whole data should be trained

MODEL_SAVED_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/trained_models/420000_training_data'
```

## Running the cart pole inference
Based on a trained diffusion model, the cart pole inference is running via

```bash
conda activate mpd
cd scripts/inference
python Diffusion_MPC_Inference.py  
```
The diffusion & mpc performance plot will be saved in the results_inference folder under the MODEL_PATH set in Diffusion_MPC_Inference.py.

---
## Running the MPD inference

To try out the MPD inference, first download the data and the trained models. 

```bash
conda activate mpd
```

```bash
gdown --id 1mmJAFg6M2I1OozZcyueKp_AP0HHkCq2k
tar -xvf data_trajectories.tar.gz
gdown --id 1I66PJ5QudCqIZ2Xy4P8e-iRBA8-e2zO1
tar -xvf data_trained_models.tar.gz
```

To solve the possible error about pyopensll Version,
```bash
pip install --upgrade pyOpenSSL
```

Run the inference script
```bash
cd scripts/inference
python inference.py
```

Comment out the `model-id` variable in `scripts/inference/inference.py` to try out different models
```python
model_id: str = 'EnvDense2D-RobotPointMass'
model_id: str = 'EnvNarrowPassageDense2D-RobotPointMass'
model_id: str = 'EnvSimple2D-RobotPointMass'
model_id: str = 'EnvSpheres3D-RobotPanda'
```

Depending on the task (`model-id`) you might need to change the weights for collision and smoothness (we will provide an "hyperpameter search" soon.)
```python
weight_grad_cost_collision: float = 3e-2
weight_grad_cost_smoothness: float = 1e-2
```

The results will be saved under `data_trained_models/[model_id]/results_inference/`.

---
## Generate data and train from scratch

We recommend running the follwowing in a SLURM cluster.

```bash
conda activate mpd
```

To regenerate the data:
```bash
cd scripts/generate_data
python launch_generate_trajectories.py
```

To train the model:
```bash
cd scripts/train_diffusion
python launch_train_01.py
```





---
## Citation

If you use our work or code base(s), please cite our article:
```latex
@inproceedings{carvalho2023mpd,
  title={Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models},
  author={Carvalho, J. and  Le, A.T. and  Baierl, M. and  Koert, D. and  Peters, J.},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```


---
## Credits

Parts of this work and software were taken and/or inspired from:
- [https://github.com/jannerm/diffuser](https://github.com/jannerm/diffuser)

