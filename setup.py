from setuptools import setup, find_packages

__version__ = '2.1.0'

with open("README.md", "r", encoding="utf-8") as fid:
  long_description = fid.read()

setup(
    name='ocnn',
    version=__version__,
    author='Peng-Shuai Wang',
    author_email='wangps@hotmail.com',
    description='Octree-based Sparse Convolutional Neural Networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/octree-nn/ocnn-pytorch',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['torch', 'numpy'],
    python_requires='>=3.6',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
