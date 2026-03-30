from setuptools import setup, find_packages

setup(
    name="valid-framework",
    version="0.1.0",
    description="VALID: Validation Architecture for Learning-based Investment Decisions",
    author="Jaewook Kim",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "matplotlib>=3.7",
    ],
    extras_require={
        "ml": ["catboost>=1.2", "lightgbm>=4.0", "torch>=2.0"],
        "dev": ["pytest>=7.0"],
    },
)
