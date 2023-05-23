import os
import platform

from setuptools import setup

tensorflow = 'tensorflow'
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    tensorflow = 'tensorflow-macos'

    os.environ['GRPC_PYTHON_BUILD_SYSTEM_OPENSSL'] = '1'
    os.environ['GRPC_PYTHON_BUILD_SYSTEM_ZLIB'] = '1'

install_requires = ['numpy', tensorflow]

setup(
    name='keras-ed',
    version='1.0.0',
    description='Keras Encoder-Decoder',
    author='Who ducking cares...',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['ed'],
    install_requires=install_requires
)
