import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="pandemic",
    version="0.0.1",
    author="Chris Jones",
    author_email="cmjone25@ncsu.edu",
    description="Simulation for PoPS Pandemic Model",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/ncsu-landscape-dynamics/Pandemic_Model",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "geopandas",
        "python-dotenv",
        "requests",
        "rasterio",
        "rasterstats",
        "matplotlib",
        "seaborn",
        "fuzzywuzzy",
        "pycountry",
        "haversine",
    ],
)
