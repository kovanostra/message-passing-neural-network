from setuptools import setup, find_packages

setup(
    name='protein-folding',
    version='0.1',
    packages=find_packages(exclude=["tests"]),
    url='',
    license='',
    author='Michail Kovanis',
    description='',
    install_requires=[
        'numpy==1.17.4',
        'torch==1.4.0'
    ]
)
