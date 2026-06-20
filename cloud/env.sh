# Source after activating the venv: `source cloud/env.sh`
export MUJOCO_GL=egl            # headless offscreen rendering for MuJoCo/robosuite
export DATALOADER_WORKERS=8     # data-loading parallelism on the cloud box
export PYOPENGL_PLATFORM=egl
# export WANDB_API_KEY=...       # set your key (or run `wandb login`)
