from setuptools import setup, find_packages

setup(
    name='ocnn',
    version='2.0.0',
    author='Peng-Shuai Wang',
    author_email='wangps@hotmail.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["torch", "numpy"],
)
