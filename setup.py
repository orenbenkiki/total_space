from codecs import open
from os import path
from setuptools import setup, find_packages


setup(
    name='total_space',
    version='0.2.5',
    description='Investigate the total state space of communicating finite state machines',
    long_description_content_type='text/plain',
    long_description='''
        Investigate the total state space of communicating finite state machines. Specifically,
        given a model of a system comprising of multiple agents, where each agent is a
        non-deterministic state machine, which responds to either time or receiving a message with
        one of some possible actions, where each such action can change the agent state and/or send
        messages to other agents; Then this package will generate the total possible state space of
        the overall system, validate the model for completeness, validate each system state for
        additional arbitrary correctness criteria, and visualize the states and transitions in
        various ways.
    ''',
    url='https://github.com/orenbenkiki/total_space',
    author='Oren Ben-Kiki',
    author_email='oren@ben-kiki.org',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='sample setuptools development',
    packages=['total_space'],
    extras_require={
        '': ['pytest'],
    },
    package_data={
        '': ['VERSION.txt', 'LICENSE.txt']
    },
)
