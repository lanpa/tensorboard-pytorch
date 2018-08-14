#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        # Dynamically compile protos after installation
        subprocess.call(['./compile_protos.sh'])


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Dynamically compile protos after installation
        subprocess.call(['./compile_protos.sh'])

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'protobuf >= 3.2.0',
    'six',
]

test_requirements = [
    'pytest',
    'matplotlib'
]

setup(
    name='tensorboardX',
    version='1.4',
    description='TensorBoardX lets you watch Tensors Flow without Tensorflow',
    long_description=history,
    author='Tzu-Wei Huang',
    author_email='huang.dexter@gmail.com',
    url='https://github.com/lanpa/tensorboardX',
    packages=find_packages(exclude=['docs', 'tests', 'examples']),
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    scripts=['compile_protos.sh'],
    test_suite='tests',
    tests_require=test_requirements
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*
