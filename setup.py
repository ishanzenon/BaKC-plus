"""
Setup configuration for BaKC-plus package
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = []

# Add PyYAML for config management
if 'pyyaml' not in [r.lower().split('>=')[0].split('==')[0] for r in requirements]:
    requirements.append('pyyaml>=5.4.0')

# Development dependencies
dev_requirements = [
    'pytest>=6.2.0',
    'pytest-cov>=2.12.0',
    'pytest-mock>=3.6.0',
    'black>=21.0',
    'flake8>=3.9.0',
    'mypy>=0.900',
]

setup(
    name="bakc-plus",
    version="0.1.0",
    author="BaKC-plus Development Team",
    description="Bagging and Kernel-based Conformal Prediction for Anomaly Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ishanzenon/BaKC-plus",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            # Will be added in Phase 3
            # "bakc-train=bakc_plus.scripts.train:main",
            # "bakc-evaluate=bakc_plus.scripts.evaluate:main",
        ],
    },
)
