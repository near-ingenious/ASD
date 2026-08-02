"""
setup.py — Robust ASD Diagnosis Under Missing Clinical Modalities
Metropolitan University, Sylhet-3104, Bangladesh
"""
from setuptools import setup, find_packages

setup(
    name             = "asd-multimodal",
    version          = "1.0.0",
    description      = (
        "Robust ASD Diagnosis Under Missing Clinical Modalities "
        "Using Cross-Modal Representation Reconstruction and "
        "Explainable Multimodal Learning"
    ),
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    authors          = [
        {"name": "Jarin Alam Prity",   "email": "jarinprity438@gmail.com"},
        {"name": "Popy Rani Boidya",   "email": "popyboidya@gmail.com"},
    ],
    maintainer       = "Jarin Alam Prity",
    maintainer_email = "jarinprity438@gmail.com",
    url              = "https://github.com/YOUR_USERNAME/asd-missing-modalities",
    license          = "MIT",
    packages         = find_packages(where="src"),
    package_dir      = {"": "src"},
    python_requires  = ">=3.10",
    install_requires = [
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "torch>=2.0",
        "xgboost>=1.7",
        "lightgbm>=4.0",
        "shap>=0.42",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    extras_require   = {
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
            "mypy>=1.0",
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
        "harmonise": [
            "neuroHarmonize>=2.3",   # Full ComBat-GAM support
        ],
        "neuroimaging": [
            "nilearn>=0.10",
            "nibabel>=5.0",
        ],
    },
    classifiers      = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords         = (
        "autism ASD fMRI multimodal missing-data explainability fairness "
        "ABIDE functional-connectivity machine-learning"
    ),
)
