#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="rigvid"
PYTHON_VER="3.10"
CUDA_VER="11.8"
PYTORCH_VER="2.4.1"

# Clone repositories
echo "Cloning RollingDepth and FoundationPose repositories..."
git clone https://github.com/prs-eth/RollingDepth.git
git clone https://github.com/NVlabs/FoundationPose.git

# Create and activate conda environment
echo "Creating conda environment '${ENV_NAME}'..."
conda create -n "${ENV_NAME}" python="${PYTHON_VER}" cudatoolkit="${CUDA_VER}" -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Install numpy and scipy with ABI compatibility
echo "Installing numpy and scipy..."
conda install -y -c conda-forge numpy=1.26.4 scipy=1.12.0 PyJWT

# Install GPU PyTorch
echo "Installing PyTorch, torchvision, and torchaudio..."
pip install \
  torch=="${PYTORCH_VER}+cu118" \
  torchvision=="0.19.1+cu118" \
  torchaudio=="${PYTORCH_VER}+cu118" \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Create constraints file
cat > constraints.txt <<EOF
torch==${PYTORCH_VER}+cu118
torchvision==0.19.1+cu118
torchaudio==${PYTORCH_VER}+cu118
numpy==1.26.4
scipy==1.12.0
EOF

# Install RollingDepth dependencies and diffusers development version
echo "Installing RollingDepth dependencies..."
pushd RollingDepth >/dev/null
pip install -r requirements.txt -c ../constraints.txt
bash script/install_diffusers_dev.sh
popd >/dev/null

# Set PYTHONPATH for RollingDepth
export PYTHONPATH="$(pwd)/RollingDepth:${PYTHONPATH:-}"

# Install FoundationPose dependencies
echo "Installing FoundationPose dependencies..."
conda install -y -c conda-forge eigen=3.4.0

pushd FoundationPose >/dev/null

# Adjust FoundationPose requirements to avoid conflicts
sed -i '/^torch==.*+cu118/d; /^torchvision==.*+cu118/d; /^torchaudio==.*+cu118/d' requirements.txt
pip install -r requirements.txt -c ../constraints.txt

# Additional FoundationPose dependencies
pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
pip install fvcore iopath ninja
pip install git+https://github.com/facebookresearch/pytorch3d.git

# Set CMAKE prefix path for pybind11
export CMAKE_PREFIX_PATH="$CONDA_PREFIX/lib/python${PYTHON_VER}/site-packages/pybind11/share/cmake/pybind11:${CMAKE_PREFIX_PATH:-}"

# Configure setup.py for FoundationPose CUDA extensions
SETUP_PY="bundlesdf/mycuda/setup.py"
cat > "$SETUP_PY" <<'EOF'
from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

code_dir = os.path.dirname(os.path.realpath(__file__))

cxx_flags = ['-O3', '-std=c++17']
nvcc_flags = [
    '-O3',
    '--expt-relaxed-constexpr',
    '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='common',
    ext_modules=[
        CUDAExtension('common', [
            'bindings.cpp',
            'common.cu',
        ], extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}),
        CUDAExtension('gridencoder', [
            f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
            f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
        ], extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}),
    ],
    include_dirs=[
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
EOF

# Build FoundationPose extensions
bash build_all_conda.sh
popd >/dev/null

# Completion message
echo "Setup complete. Activate the environment with:"
echo "  conda activate ${ENV_NAME}"
echo "Then run the projects with:"
echo "  RollingDepth: python run_video.py"
echo "  FoundationPose: cd FoundationPose && python run_demo.py"
