# pytorch-gpu-mlproject

Repository containing scaffolding for a Python 3-based data science project with GPU acceleration using the [PyTorch](https://pytorch.org/) ecosystem. 

## Project organization

Project organization is based on ideas from [_Good Enough Practices for Scientific Computing_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510).

1. Put each project in its own directory, which is named after the project.
2. Put external scripts or compiled programs in the `bin` directory.
3. Put raw data and metadata in a `data` directory.
4. Put text documents associated with the project in the `doc` directory.
5. Put all Docker related files in the `docker` directory.
6. Install the Conda environment into an `env` directory. 
7. Put all notebooks in the `notebooks` directory.
8. Put files generated during cleanup and analysis in a `results` directory.
9. Put project source code in the `src` directory.
10. Name all files to reflect their content or function.

PyTorch
=======

PyTorch is a python package that provides two high-level features:

* Tensor computation (like numpy) with strong GPU acceleration
* Deep Neural Networks built on a tape-based autograd system

You can reuse your favorite python packages such as numpy, scipy and Cython to
extend PyTorch when needed.

## Contents of the PyTorch image

This container has the PyTorch framework installed and ready to use. The
pytorch python module is installed as part of a Python 3.5 Conda environment in
/opt/conda/envs/pytorch-py35. 
Both the compiled pytorch libraries and the Python 3.5 environment are included in $PATH. As
a result, running python from the command line executes a Python 3.5
interpreter by default. 
`/opt/pytorch` contains the complete source of this version of PyTorch.

## Running PyTorch

You can choose to use PyTorch as provided by NVIDIA, or you can choose to
customize it. Run pytorch as you would any python program: run a python script,
open interactive session in ipython or jupyter notebook. Start your scripts
or interactive sessions with
```import torch```


## Customizing PyTorch

You can customize PyTorch one of two ways:

(1) Modify the version of the source code in this container and run your
customized version, or (2) use `docker build` to add your customizations on top
of this container if you want to add additional packages.

NVIDIA recommends option 2 for ease of migration to later versions of the
PyTorch container image.

For more information, see https://docs.docker.com/engine/reference/builder/ for
a syntax reference.  Several example Dockerfiles are provided in the container
image in `/workspace/docker-examples`.

## Suggested Reading

For more information about pytorch, see 
 - pytorch documentation http://pytorch.org/docs, 
 - pytorch tutorials https://github.com/pytorch/tutorials
 - pytorch examples https://github.com/pytorch/examples
 - a collection of links to pytorch tutorials, examples, projects and paper implementations https://github.com/ritchieng/the-incredible-pytorch
