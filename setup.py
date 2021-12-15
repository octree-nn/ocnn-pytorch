from setuptools import setup

setup(
    name='ocnn',
    version='2.0',
    install_requires=["torch", "numpy"],
    packages=['ocnn'],
    package_dir={'ocnn': 'ocnn'},
)
