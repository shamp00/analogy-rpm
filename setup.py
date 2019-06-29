import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyAnalogyRpm",
    version="0.1.0",
    author="Robert Anderson",
    author_email="github@nosredna.com",
    description="A contrastive Hebbian learning approach to solving Raven's Progressive Matrices.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shamp00/rpm-analogy",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3.6',
    install_requires=[
            'pycairo',
        ]
)