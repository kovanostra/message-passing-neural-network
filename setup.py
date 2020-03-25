from setuptools import setup, find_packages

setup(
    name='message-passing-nn',
    version='0.1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Michail Kovanis',
    description='',
    install_requires=[
        'click',
        'numpy==1.17.4',
        'torch==1.4.0'
    ],
    entry_points={
        'console_scripts': [
            'message-passing-nn = src.cli:main'
        ],
    },
)
