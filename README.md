### ClustSeg

Code for review (instance seg)

## Installation

```shell
conda create -n <env_name> python=3.8 -y
conda activate <env_name>
cd ClustSeg

# compile the pytorch file from the source code
git clone --recursive https://github.com/pytorch/pytorch
conda install astunparse numpy ninja pyyaml setuptools cmake typing_extensions six requests dataclasses mkl mkl-include

conda install -c pytorch magma-cuda110 # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo

# manually replace the torch folder with provided one and then compile the pytorch
cd pytorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

python setup.py develop

# install other requirements
pip install -r requirements.txt
```
<!--
**ClustSeg/ClustSeg** is a āØ _special_ āØ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- š­ Iām currently working on ...
- š± Iām currently learning ...
- šÆ Iām looking to collaborate on ...
- š¤ Iām looking for help with ...
- š¬ Ask me about ...
- š« How to reach me: ...
- š Pronouns: ...
- ā” Fun fact: ...
-->
