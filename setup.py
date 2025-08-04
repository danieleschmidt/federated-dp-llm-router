#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="federated-dp-llm-router",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    description="Privacy-Preserving Federated LLM Router with Differential Privacy for Healthcare",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/federated-dp-llm-router",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "numpy>=1.24.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "cryptography>=41.0.0",
        "httpx>=0.25.0",
        "python-multipart>=0.0.6",
        "pyjwt>=2.8.0",
        "redis>=5.0.0",
        "prometheus-client>=0.19.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "security": [
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "federated-dp-llm=federated_dp_llm.cli:main",
        ],
    },
)