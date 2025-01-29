import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wincdl",
    version="0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "torch",
        "tqdm",
        "joblib",
        "alphacsc",
    ],
    # python_requires='>=3.8',
    python_requires=">=3.7",
    extras_require={"test": ["pytest"]},
)
