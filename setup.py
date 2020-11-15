from codecs import open
from os import path
from setuptools import setup, find_packages


root = path.abspath(path.dirname(__file__))

with open(path.join(root, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(root, 'LICENSE.txt'), encoding='utf-8') as f:
    license = f.read()

with open(path.join(root, 'VERSION.txt'), encoding='utf-8') as f:
    version = f.read().strip()


setup(
    name='total_space',
    version=version,
    description='Investigate the total state space of communicating finite state machines',
    long_description=long_description,
    url='https://github.com/orenbenkiki/total_space',
    author='Oren Ben-Kiki',
    author_email='oren@ben-kiki.org',
    license=license,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='sample setuptools development',
    packages=find_packages(),
    extras_require={
        '': ['pytest'],
    },
    package_data={
        '': ['VERSION.txt', 'LICENSE.txt']
    },
)
