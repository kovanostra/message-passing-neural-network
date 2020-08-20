from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='message-passing-nn',
    version='1.6.0',
    packages=find_packages(exclude=["tests"]),
    url='https://github.com/kovanostra/message-passing-nn',
    download_url='https://github.com/kovanostra/message-passing-nn/archive/1.6.0.tar.gz',
    keywords=['MESSAGE PASSING', 'NEURAL NETWORK', 'RNN', 'GRU'],
    license='MIT',
    author='Michail Kovanis',
    description='A message passing neural network with RNN or GRU units',
    install_requires=[
        'click',
        'numpy==1.17.4',
        'pandas==1.0.3',
        'torch==1.5.0',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'message-passing-nn = message_passing_nn.cli:main'
        ],
    },
    ext_modules=[
        CppExtension('rnn_encoder_cpp',
                     sources=['message_passing_nn/graph/rnn_encoder.cpp',
                              'message_passing_nn/utils/messages.cpp',
                              'message_passing_nn/utils/derivatives.cpp'])],
    cmdclass={'build_ext': BuildExtension},
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
