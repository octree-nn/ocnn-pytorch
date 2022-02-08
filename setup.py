from setuptools import setup, find_packages

__version__ = '2.0.0'

setup(
    name='ocnn',
    version=__version__,
    author='Peng-Shuai Wang',
    author_email='wangps@hotmail.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["torch", "numpy"],
)
