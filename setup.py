from setuptools import setup, find_packages

__version__ = '2.1.5'

with open("README.md", "r", encoding="utf-8") as fid:
  long_description = fid.read()

setup(
    name='ocnn',
    version=__version__,
    author='Peng-Shuai Wang',
    author_email='wangps@hotmail.com',
    url='https://github.com/octree-nn/ocnn-pytorch',
    description='Octree-based Sparse Convolutional Neural Networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    zip_safe=False,
    install_requires=['torch', 'numpy'],
    python_requires='>=3.6',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
