from setuptools import setup, find_packages

setup(
    name="federated-dp-llm-router",
    version="0.1.0",
    description="Federated, differentially-private LLM request router",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
)
