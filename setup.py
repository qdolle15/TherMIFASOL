from setuptools import setup, find_packages

# version = subprocess.check_output(['git', 'describe','--tag', '--abbrev=0']).decode('ascii').strip()

setup(
    name="TherMIFASOL",
    version="0.1.0",
    author="Quentin Dollé",
    author_email="quentin.dolle@polytechnique.edu",
    description="TherMIFASOL is a framework developed during Quentin Dollé' P.h.D thesis for microstructure prediction during DED-AM process. \
        It includes python code for sensors monitoring, ",
    packages=find_packages(),  # Check an __init__.py file is in every sub directories
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib"
    ],
)
