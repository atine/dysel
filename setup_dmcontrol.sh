# deactivate existing virtualenv
if [[ "$VIRTUAL_ENV" != "" ]]
then
    deactivate
fi


# check CUDA versions
CUDA_VERSION=$(cat /usr/local/cuda/version.json | python3 -c "import sys, json; print(json.load(sys.stdin)['cuda']['version'])")
echo "default CUDA:" $CUDA_VERSION

export PATH=/usr/local/cuda-12.0/bin:$PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH
export PATH=/usr/local/cuda-12.2/bin:$PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PATH=/usr/local/cuda-12.5/bin:$PATH

if [ -e "/usr/local/cuda-12.0/version.json" ]; then
    FOUND_CUDA_VERSION=$(cat /usr/local/cuda-12.0/version.json | python3 -c "import sys, json; print(json.load(sys.stdin)['cuda']['version'])")
    echo "found CUDA:" $FOUND_CUDA_VERSION
fi
if [ -e "/usr/local/cuda-12.1/version.json" ]; then
    FOUND_CUDA_VERSION=$(cat /usr/local/cuda-12.1/version.json | python3 -c "import sys, json; print(json.load(sys.stdin)['cuda']['version'])")
    echo "found CUDA:" $FOUND_CUDA_VERSION
fi
if [ -e "/usr/local/cuda-12.2/version.json" ]; then
    FOUND_CUDA_VERSION=$(cat /usr/local/cuda-12.2/version.json | python3 -c "import sys, json; print(json.load(sys.stdin)['cuda']['version'])")
    echo "found CUDA:" $FOUND_CUDA_VERSION
fi
if [ -e "/usr/local/cuda-12.3/version.json" ]; then
    FOUND_CUDA_VERSION=$(cat /usr/local/cuda-12.3/version.json | python3 -c "import sys, json; print(json.load(sys.stdin)['cuda']['version'])")
    echo "found CUDA:" $FOUND_CUDA_VERSION
fi
if [ -e "/usr/local/cuda-12.4/version.json" ]; then
    FOUND_CUDA_VERSION=$(cat /usr/local/cuda-12.4/version.json | python3 -c "import sys, json; print(json.load(sys.stdin)['cuda']['version'])")
    echo "found CUDA:" $FOUND_CUDA_VERSION
fi
if [ -e "/usr/local/cuda-12.5/version.json" ]; then
    FOUND_CUDA_VERSION=$(cat /usr/local/cuda-12.5/version.json | python3 -c "import sys, json; print(json.load(sys.stdin)['cuda']['version'])")
    echo "found CUDA:" $FOUND_CUDA_VERSION
fi

echo "using newest CUDA:" $FOUND_CUDA_VERSION


# setting virutalenv
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "using virtualenv in directory:" $SCRIPT_DIR
source $SCRIPT_DIR/.venv/bin/activate


# avoid timeouts and prints
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=300


# aiming for deterministic
export MUJOCO_GL=egl
export TF_CPP_MIN_LOG_LEVEL=3
export TF_CUDNN_DETERMINISTIC=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"


# for cleaner interfaces
export DISPLAY=:0
export JAX_PLATFORMS=cuda,cpu
