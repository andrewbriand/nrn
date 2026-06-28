# Building NEURON with GPU Support (CoreNEURON + OpenACC)

This guide documents how to build the NEURON simulator from source with GPU acceleration via CoreNEURON using OpenACC and the NVIDIA HPC SDK.

## Prerequisites

### System Dependencies (Ubuntu 24.04)

```bash
sudo apt-get update
sudo apt-get install -y bison cmake flex git \
    libncurses-dev libreadline-dev \
    python3-dev python3-venv
```

### NVIDIA HPC SDK

The GPU build **requires** the NVIDIA HPC SDK (formerly PGI) for OpenACC compilation.

- Download from: https://developer.nvidia.com/hpc-sdk
- Installation path for this guide: `/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/`
- Requires CUDA toolkit >= 9.0 (bundled with NVHPC SDK)

Verify installation:

```bash
nvc --version   # Should report NVHPC 25.7+
nvcc --version  # Should report CUDA 12.x+
```

### Verify GPU

```bash
nvidia-smi      # Confirm GPU is available and note CUDA version
```

For this build: **NVIDIA GeForce RTX 4090** (Ada Lovelace, compute capability 8.9).

## Step 1: Python Virtual Environment

The system Python is externally managed, so create a venv:

```bash
cd /path/to/nrn
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy jinja2 pyyaml sympy find_libpython packaging setuptools
```

## Step 2: Configure with CMake

```bash
mkdir -p build && cd build

# Ensure NVHPC compilers are in PATH
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH"
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9"

cmake .. \
    -DNRN_ENABLE_CORENEURON=ON \
    -DCORENRN_ENABLE_GPU=ON \
    -DNRN_ENABLE_INTERVIEWS=OFF \
    -DNRN_ENABLE_RX3D=OFF \
    -DNRN_ENABLE_MPI=OFF \
    -DCMAKE_INSTALL_PREFIX="$PWD/../install" \
    -DCMAKE_C_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/nvc \
    -DCMAKE_CXX_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/nvc++ \
    -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/nvcc \
    -DCUDAToolkit_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9" \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DPYTHON_EXECUTABLE="/workspace/nrn/venv/bin/python" \
    -DCMAKE_BUILD_TYPE=Release
```

### CMake Options Explained

| Option | Value | Description |
|--------|-------|-------------|
| `NRN_ENABLE_CORENEURON` | ON | Enable CoreNEURON optimized engine |
| `CORENRN_ENABLE_GPU` | ON | Enable GPU support via OpenACC |
| `NRN_ENABLE_INTERVIEWS` | OFF | Disable GUI (not needed for GPU) |
| `NRN_ENABLE_RX3D` | OFF | Disable 3D reaction-diffusion |
| `NRN_ENABLE_MPI` | OFF | Disable MPI (enable if needed) |
| `CMAKE_C_COMPILER` | nvc | NVIDIA HPC C compiler |
| `CMAKE_CXX_COMPILER` | nvc++ | NVIDIA HPC C++ compiler |
| `CMAKE_CUDA_COMPILER` | nvcc | NVIDIA CUDA compiler |
| `CMAKE_CUDA_ARCHITECTURES` | 89 | Compute capability for RTX 4090 |
| `PYTHON_EXECUTABLE` | venv/bin/python | Python from virtual environment |
| `CMAKE_BUILD_TYPE` | Release | Optimized build |

### CUDA Architectures Reference

| GPU | Compute Capability | Flag |
|-----|-------------------|------|
| RTX 4090 | 8.9 | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| RTX 3090 | 8.6 | `-DCMAKE_CUDA_ARCHITECTURES=86` |
| RTX 2080 | 7.5 | `-DCMAKE_CUDA_ARCHITECTURES=75` |
| V100 | 7.0 | `-DCMAKE_CUDA_ARCHITECTURES=70` |
| A100 | 8.0 | `-DCMAKE_CUDA_ARCHITECTURES=80` |
| H100 | 9.0 | `-DCMAKE_CUDA_ARCHITECTURES=90` |
| Multiple | - | `-DCMAKE_CUDA_ARCHITECTURES=60;70;80` |

## Step 3: Build

```bash
cmake --build . --parallel $(nproc) --target install
```

Build time: ~10-20 minutes depending on system speed.

## Step 4: Verify Installation

### Check the build configuration

```bash
export PATH="$(pwd)/../install/bin:$PATH"
nrniv --version
```

Look for `CORENRN_ENABLE_GPU=ON` in the output.

### Check CoreNEURON GPU support

```bash
nrniv-core --help
```

Verify `--gpu` option is listed under `[Option Group: GPU]`.

### Full GPU Integration Test

```bash
# Set up environment
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH"
export PATH="$PWD/../install/bin:$PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH"
export NMODL_PYLIB="/usr/lib/x86_64-linux-gnu/libpython3.12.so"
export NMODLHOME="$PWD/../install"

nrniv -python -c "
from neuron import h, coreneuron

soma = h.Section(name='soma')
soma.L = 5.0
soma.diam = 5.0
soma.insert('pas')
soma.cm = 1.0

ic = h.IClamp(soma(0.5))
ic.delay = 1; ic.dur = 1; ic.amp = 1
h('tstop = 10')
h.CVode().cache_efficient(1)

# CPU run
v = h.Vector().record(soma(0.5)._ref_v)
h.finitialize(-65)
h.ParallelContext().psolve(h('tstop'))
cpu_val = v[int(len(v)/2)]

# GPU run
coreneuron.enable = True
coreneuron.gpu = True
v2 = h.Vector().record(soma(0.5)._ref_v)
h.finitialize(-65)
h.ParallelContext().psolve(h('tstop'))
gpu_val = v2[int(len(v2)/2)]

print(f'CPU: {cpu_val:.6f}  GPU: {gpu_val:.6f}  Diff: {abs(cpu_val-gpu_val):.2e}')
"
# Expected output: CPU: -66.9486xx  GPU: -66.9486xx  Diff: ~1e-13
```

## Step 5: Environment Setup for Development

Add to your `~/.bashrc` or create an activation script:

```bash
# NVHPC SDK
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH"

# NEURON installation
export PATH="/path/to/nrn/install/bin:$PATH"
export LD_LIBRARY_PATH="/path/to/nrn/install/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/path/to/nrn/install/lib/python:$PYTHONPATH"

# Required for nrnivmodl -coreneuron (nmodl needs embedded Python)
export NMODL_PYLIB="/usr/lib/x86_64-linux-gnu/libpython3.12.so"
export NMODLHOME="/path/to/nrn/install"

# Virtual environment
source /path/to/nrn/venv/bin/activate
```

## Using GPU Acceleration

### Building MOD Files with CoreNEURON

Set required environment variables before building MOD files:

```bash
export NMODL_PYLIB="/usr/lib/x86_64-linux-gnu/libpython3.12.so"
export NMODLHOME="/path/to/nrn/install"
```

Build MOD files for GPU:

```bash
nrnivmodl -coreneuron /path/to/mod/files
```

This produces `x86_64/special` (NEURON) and `x86_64/special-core` (CoreNEURON standalone).

### In Python (NEURON 9.0+ API)

```python
from neuron import h, coreneuron

# Build your model...
soma = h.Section(name='soma')
soma.insert('pas')

# Must set tstop via HOC string in NEURON 9.0
h('tstop = 10')

# Enable cache efficiency required by CoreNEURON
h.CVode().cache_efficient(1)

# Run with NEURON (CPU)
pc = h.ParallelContext()
h.finitialize(-65)
pc.psolve(h('tstop'))

# Run with CoreNEURON + GPU
coreneuron.enable = True
coreneuron.gpu = True
coreneuron.verbose = 0
coreneuron.cell_permute = 2  # better GPU utilization

pc = h.ParallelContext()
h.finitialize(-65)
pc.psolve(h('tstop'))
```

### Important: NEURON 9.0 API Changes

NEURON 9.0 introduced breaking API changes vs older versions:

| Old API | NEURON 9.0 API |
|---------|----------------|
| `h.tstop = 10` | `h('tstop = 10')` |
| `h.cvode` | `h.CVode()` |
| `h.stdinit()` | `h.finitialize()` |
| `h.run()` | `h('continuerun(tstop)')` |
| `h.continuerun(t)` | `h('continuerun(t)')` |

### Important Notes for GPU Usage

1. **Must use `special` binary**: GPU execution requires launching through `special`, not `nrniv` or `python` directly:
   ```bash
   nrnivmodl -coreneuron /path/to/mod/files
   ./x86_64/special -python script.py
   ```

2. **MOD file compatibility**: MOD files must be THREADSAFE for CoreNEURON. POINTER variables need conversion to BBCOREPOINTER.

3. **TABLE constructs**: Not supported with GPU execution. Comment out TABLE statements using `:` operator.

4. **Random number generators**: Use Random123 instead of MCellRan4.

5. **Cell permutation**: Enable cell reordering for better GPU performance:
   ```python
   coreneuron.cell_permute = 2  # 1 or 2
   ```

## CUDA Kernel Solver (vs OpenACC)

CoreNEURON has **two GPU backends** for the Hines matrix solver:

| Backend | Source | Kernel Name | Activation |
|---------|--------|-------------|------------|
| OpenACC | `cellorder.cpp:605` | `solve_interleaved1_793` (auto-generated) | Default when `gpu=True` |
| CUDA | `cellorder.cu:66` | `solve_interleaved2_kernel` (hand-written) | `coreneuron.cuda_interface = True` |

### Architecture

```
solve_interleaved(ith)
  ├── permute=1: solve_interleaved1()      → OpenACC kernel only
  └── permute=2: solve_interleaved2()
        ├── cuda_interface=False (default) → OpenACC kernel (solve_interleaved2_loop_body)
        └── cuda_interface=True             → CUDA kernel (solve_interleaved2_kernel)
```

The CUDA kernel (`cellorder.cu:66-113`) is a hand-tuned `__global__` kernel that parallelizes across all threads in one launch. It performs triangular solve + back-substitution for the Hines matrix. The OpenACC path instead relies on compiler auto-parallelization of nested loops.

**Important**: The CUDA kernel path only works with `cell_permute=2` because it's inside `solve_interleaved2()`. Setting `cell_permute=1` will call `solve_interleaved1()` which has no CUDA branch.

### Enabling the CUDA Kernel

```python
from neuron import h, coreneuron

coreneuron.enable = True
coreneuron.gpu = True
coreneuron.cell_permute = 2     # Required: CUDA kernel needs permute type 2
coreneuron.cuda_interface = True  # Switch from OpenACC to CUDA backend
```

### Verifying Which Kernel is Active

```bash
nsys profile --trace=cuda --stats=true ./x86_64/special -python script.py 2>&1 | grep solve
```

- OpenACC active: `solve_interleaved1_793` or `solve_interleaved1_<n>`
- CUDA active: `solve_interleaved2_kernel(NrnThread*, InterleaveInfo*, int)`

### Run via CLI (standalone special-core)

```bash
./x86_64/special-core --gpu --cell-permute 2 --cuda-interface --tstop 100 --datpath .
```

### CUDA Kernel Correctness Tests

The CUDA solver is tested by a C++ unit test that validates all 7 solver implementations produce numerically identical results (tolerance `2e-11`) against the reference CPU path.

#### Build Test Target

The test target requires rebuilding with `-DNRN_ENABLE_TESTS=ON`:

```bash
cd build
cmake .. \
    -DNRN_ENABLE_TESTS=ON \
    <all your existing GPU build flags>
cmake --build . --parallel $(nproc) --target test-solver
```

#### Run

```bash
ctest --test-dir build -R test-solver --output-on-failure -V
```

Or invoke the binary directly:

```bash
./build/bin/test-solver
```

#### Test Cases

| Test Case | Description |
|-----------|-------------|
| `SingleCellAndThread` | 1 cell, 32 segments |
| `UnbalancedCellSingleThread` | 1 cell, 19 segments (non power-of-2) |
| `LargeCellSingleThread` | 1 cell, 4096 segments |
| `ManySmallCellsSingleThread` | 1024 cells, 3 segments each |
| `ManySmallCellsMultiThread` | 1024 cells split across 2 threads |
| `LargeCellSingleThreadRandom` | 4096 segments, random matrix values |
| `ManySmallCellsSingleThreadRandom` | 1024 cells, random matrix values |

#### What Each Test Case Validates

Each case runs **all 7 solver implementations** and compares output:

```
CellPermute0_CPU    (reference, permute=0, CPU)
CellPermute0_GPU    (permute=0, OpenACC)
CellPermute1_CPU    (permute=1, CPU)
CellPermute1_GPU    (permute=1, OpenACC)
CellPermute2_CPU    (permute=2, CPU)
CellPermute2_GPU    (permute=2, OpenACC)
CellPermute2_CUDA   (permute=2, CUDA kernel)   ← verifies solve_interleaved2_kernel
```

All produce bitwise-identical parent indices and numerically identical `d`, `rhs` values.

#### Expected Output (passing)

```
Comparing SolverImplementation::CellPermute2_CUDA to SolverImplementation::CellPermute0_CPU
...
All tests passed (510942 assertions in 7 test cases)
```

## Ringtest GPU Benchmark

The [ringtest](https://github.com/neuronsimulator/ringtest) is the standard benchmark for CoreNEURON GPU performance. It simulates a network of ring-connected cells and scales well to test GPU acceleration.

### Clone and Build

```bash
cd /path/to/nrn/..
git clone https://github.com/neuronsimulator/ringtest.git
cd ringtest

# Set up environment
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH"
export PATH="/path/to/nrn/install/bin:$PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/path/to/nrn/install/lib:$LD_LIBRARY_PATH"
export NMODL_PYLIB="/usr/lib/x86_64-linux-gnu/libpython3.12.so"
export NMODLHOME="/path/to/nrn/install"
export PYTHONPATH="/path/to/nrn/install/lib/python:$PYTHONPATH"

# Build MOD files for CoreNEURON
nrnivmodl -coreneuron mod
```

### Run Benchmarks

```bash
# Small model (256 cells) - CPU
./x86_64/special -python ringtest.py -tstop 100 -nring 32 -ncell 8 -branch 10 20 -coreneuron

# Small model (256 cells) - GPU
./x86_64/special -python ringtest.py -tstop 100 -nring 32 -ncell 8 -branch 10 20 -coreneuron -gpu

# Large model (16384 cells) - CPU
./x86_64/special -python ringtest.py -tstop 100 -nring 256 -ncell 64 -branch 32 64 -coreneuron

# Large model (16384 cells) - GPU
./x86_64/special -python ringtest.py -tstop 100 -nring 256 -ncell 64 -branch 32 64 -coreneuron -gpu
```

### Performance Results (RTX 4090)

| Model | Cells | Compartments | CPU Solver Time | GPU Solver Time | Speedup |
|-------|-------|-------------|-----------------|-----------------|---------|
| Small | 256 | 8,464 | 1.01s | 3.82s | 0.26x |
| Large | 16,384 | ~540,000 | 198.14s | 19.80s | **10.0x** |

**Note**: GPU acceleration is most effective for large models. For small models (< 1000 cells), GPU overhead (data transfer, kernel launch) dominates and CPU may be faster. The crossover point where GPU becomes faster depends on model complexity and hardware.

### Running with the CUDA Kernel Solver

The ringtest CLI (`-gpu`) uses the default OpenACC backend. To use the hand-written CUDA `__global__` kernel, use the Python API:

```bash
./x86_64/special -python -c "
from neuron import h, coreneuron
import ringtest

coreneuron.enable = True
coreneuron.gpu = True
coreneuron.cell_permute = 2      # Required for CUDA kernel path
coreneuron.cuda_interface = True  # Enable CUDA kernel instead of OpenACC

ringtest.create_rings(32, 8, (10,20), 8, method='fixed')
ringtest.runsim(100)
ringtest.write_spikes()
" 2>&1 | grep -E "cuda.interface|Solver|spikes"
# Output: --cuda-interface=true   Solver Time : 24.xxx
```

Or wrap the ringtest in a script that patches the defaults before calling `psolve`:

```python
# ringtest_cuda.py
from neuron import h, coreneuron
import ringtest

coreneuron.enable = True
coreneuron.gpu = True
coreneuron.cell_permute = 2
coreneuron.cuda_interface = True

ringtest.create_rings(32, 8, (10,20), 8)
ringtest.runsim(100)
ringtest.write_spikes()
```

Verify the CUDA kernel is active:

```bash
nsys profile --trace=cuda --stats=true \
    ./x86_64/special -python ringtest_cuda.py 2>&1 | grep solve
# Expected: coreneuron::solve_interleaved2_kernel(NrnThread*, InterleaveInfo*, int)
```

Compare OpenACC vs CUDA solver performance:

```bash
# OpenACC (default)
nsys profile --trace=cuda --stats=true -o /tmp/ringtest_openacc \
    ./x86_64/special -python ringtest.py -tstop 100 -nring 256 -ncell 64 \
    -branch 32 64 -coreneuron -gpu 2>&1 | grep Solver

# CUDA kernel
nsys profile --trace=cuda --stats=true -o /tmp/ringtest_cuda \
    ./x86_64/special -python ringtest_cuda.py 2>&1 | grep Solver
```

### Ringtest CLI Options

| Flag | Description | Example |
|------|-------------|---------|
| `-nring N` | Number of rings | `-nring 256` |
| `-ncell N` | Cells per ring | `-ncell 64` |
| `-branch A B` | Min/max branches per cell | `-branch 32 64` |
| `-tstop F` | Simulation time in ms | `-tstop 100` |
| `-coreneuron` | Enable CoreNEURON | |
| `-gpu` | Run on GPU | |
| `-permute N` | Cell permutation (0/1/2) | `-permute 2` |

### Output

The ringtest writes spike data to `out.dat` and prints runtime statistics:
```
runtime=22.7222  load_balance=0.0%  avg_comp_time=0
Solver Time : 19.8026
Number of cells: 16384
Number of spikes: 11008
```

To verify correctness, check that CPU and GPU runs produce identical spike counts. Spikes can be compared against reference data with:
```bash
sortspike out.dat > out.sort
sortspike reference_data/spk1.100ms.std.ref > ref.sort
diff out.sort ref.sort
```

### Running with MPI

For multi-GPU or multi-node runs, NEURON must be rebuilt with MPI support using the NVHPC SDK's bundled OpenMPI:

```bash
# In cmake configuration, add:
-DNRN_ENABLE_MPI=ON

# Add to PATH before building:
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/12.9/openmpi4/openmpi-4.1.5/bin:$PATH"

# Run with multiple MPI ranks:
mpiexec -n 2 ./x86_64/special -mpi -python ringtest.py -tstop 100 -coreneuron -gpu
```

## Troubleshooting

### CMake fails with NVHPC compiler

```bash
# Delete CMake cache and retry
rm -f CMakeCache.txt
```

### nvcc not found

Ensure NVHPC SDK's `bin` directory is first in PATH:

```bash
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH"
```

### Wrong CUDA version detected

Point CMake to the NVHPC SDK's bundled CUDA:

```bash
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9"
cmake ... -DCUDAToolkit_ROOT="$CUDA_HOME"
```

### Build errors with `libcudart`

The NVHPC SDK dynamically links to its bundled CUDA runtime. Set `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH"
```

### MPI with NVHPC

To enable MPI, use the NVHPC SDK's bundled OpenMPI instead of the system one (ABI incompatibility with GCC-built MPI):

```bash
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/12.9/openmpi4/openmpi-4.1.5/bin:$PATH"
```

Add `-DNRN_ENABLE_MPI=ON` to cmake flags.

### "error while loading shared libraries: /tmp/pgcudafat*.o"

NVHPC stores CUDA fat binaries in `$TMPDIR` at link time. If `/tmp` gets cleaned by `systemd-tmpfiles` (daily on most distros), binaries built with `nrnivmodl` will fail at startup.

**Fix**: Set `TMPDIR` to a persistent directory before building, then keep that directory:

```bash
mkdir -p /home/user/nrn/fatbin
export TMPDIR=/home/user/nrn/fatbin
nrnivmodl -coreneuron mod
```

After `nrnivmodl`, the binary references `pgcudafat*.o` files in `$TMPDIR` — they must not be deleted. Add `TMPDIR` to your environment setup script permanently.

## Build Configuration Summary

```
Build Type:          Release (SHARED)
C Compiler:          nvc (NVHPC 25.7)
C++ Compiler:        nvc++ (NVHPC 25.7)
CUDA Compiler:       nvcc (CUDA 12.9)
GPU Offload:         OpenACC
CUDA Architectures:  89 (Ada Lovelace)
CoreNEURON:          ON
GPU Support:         ON
Python:              3.12.3 (venv)
MPI:                 OFF
Interviews (GUI):    OFF
Rx3D:                OFF
Install Prefix:      ./install
```


diff --git a/ringtest.py b/ringtest.py
index b0e9067..ed5014c 100644
--- a/ringtest.py
+++ b/ringtest.py
@@ -164,6 +164,7 @@ def create_rings():
         coreneuron.file_mode = coreneuron_file_mode
         coreneuron.gpu = coreneuron_gpu
         coreneuron.cell_permute = coreneuron_permute
+        coreneuron.cuda_interface = True

         if args.multisplit is True:
             print("Error: multi-split is not supported with CoreNEURON\n")
