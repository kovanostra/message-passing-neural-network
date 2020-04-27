from setuptools import setup, find_packages

setup(
    name='message-passing-nn',
    version='0.4',
    packages=find_packages(exclude=["tests"]),
    url='',
    license='MIT',
    author='Michail Kovanis',
    description='A message passing neural network with GRU units',
    install_requires=[
        'click',
        'numpy==1.17.4',
        'pandas==1.0.3',
        'torch==1.4.0'
    ],
    entry_points={
        'console_scripts': [
            'message-passing-nn = src.cli:main'
        ],
    },
)
