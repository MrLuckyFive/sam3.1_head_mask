from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sam31_head_mask",
    version="0.1.0",
    description="Privacy-preserving head/face occlusion for robot datasets using SAM 3.1.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MrLuckyFive/sam3.1_head_mask",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "tests.*", "examples", "scripts")),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7",
        "torchvision",
        "opencv-python>=4.10",
        "numpy>=1.26,<2",
        "h5py>=3.10",
        "einops>=0.7",
        "pycocotools",
        "timm>=1.0.17",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "huggingface_hub>=0.24",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
    ],
)
