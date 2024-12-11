from setuptools import setup, find_packages

setup(
    name="datatools",
    version="0.1.0",
    description="A package for data conversion and analysis tools",
    author="Alberto Mittone",
    author_email="amittone@anl.gov",
    url="https://github.com/mittoalb/Allen",
    packages=find_packages(),  # Automatically find subpackages
    install_requires=[],  # List dependencies here, e.g., ["numpy", "zarr"]
    entry_points={
        'console_scripts': [
            'tiff2zarr=DataTools.DataConversion.tiff2zarr:main',
            'esrf2aps=DataTools.ESRF.ESRF2APS:main',
            'edf2aps=DataTools.ESRF.EDF2APS:main',
            'abscalc=DataTools.Physics.abscalc:main'  
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

