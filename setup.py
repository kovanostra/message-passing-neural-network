from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

setup(
    name='message-passing-nn',
    version='1.5.0',
    packages=find_packages(exclude=["tests"]),
    url='https://github.com/kovanostra/message-passing-nn',
    download_url='https://github.com/kovanostra/message-passing-nn/archive/1.5.0.tar.gz',
    keywords=['MESSAGE PASSING', 'NEURAL NETWORK', 'GRU'],
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
            'message-passing-nn = message_passing_nn.cli:main'
        ],
    },
    ext_modules=[cpp_extension.CppExtension('rnn_encoder_forward', ['message_passing_nn/model/rnn_encoder_forward.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)