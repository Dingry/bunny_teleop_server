import re
from pathlib import Path

from setuptools import setup, find_packages

_here = Path(__file__).resolve().parent
name = "dvp_teleop_server"

# Reference: https://github.com/kevinzakka/mjc_viewer/blob/main/setup.py
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

core_requirements = [
    # General dependency
    "pyzmq",
    "pyyaml",
    "pytransform3d",
    "avp_stream",
    # Our own packages
    "dex_retargeting",
    "sim_web_visualizer",
    # "dvp_teleop",
]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]


def setup_package():
    # Meta information of the project
    author = "DexVisionPro"
    author_email = ""
    description = "Teleoperation server for bimanual robot arms + hands."
    url = ""
    with open(_here / "README.md", "r") as file:
        readme = file.read()

    # Package data
    packages = find_packages(".")
    print(f"Packages: {packages}")

    setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        maintainer=author,
        maintainer_email=author_email,
        description=description,
        long_description=readme,
        long_description_content_type="text/markdown",
        url=url,
        license="MIT",
        license_files=("LICENSE",),
        packages=packages,
        python_requires=">=3.7,<=3.12",
        zip_safe=True,
        install_requires=core_requirements,
        classifiers=classifiers,
        entry_points={
            'console_scripts': [
                'run_robot_server = dvp_teleop_server.main.run_robot_server:main',
                'run_vision_server = dvp_teleop_server.main.run_vision_server:main',
            ],
        },
    )


setup_package()
