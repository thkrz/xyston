import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="xyston",
    version="1.0",
    author="Thomas Kreuzer",
    author_email="thomas.kreuzer@uni-wuerzburg.de",
    description="Artificial intelligence toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thkrz/xyston",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operation System :: OS Independent",
    ],
    python_requires=">=3.9",
)
# from numpy.distutils.core import Extension, setup

# setup(
#     name="fstpack",
#     description="A Fortran library for Fast Stockwell transforms",
#     author="Thomas Kreuzer",
#     author_email="thomas.kreuzer@uni-wuerzburg.de",
#     ext_modules=[Extension(name="fst", sources=["fst.f90"])],
# )
