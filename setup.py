"""This is the setup for the spmcmm package"""

from setuptools import setup

setup(
    name='spmcmm',
    version="1.0",
    license='',
    zip_safe=False,
    setup_requires=['numpy>=1.7.1'],
    tests_require=['numpy>=1.7.1', 'nose>=1.3'],
    install_requires=['numpy>=1.7.1', 'scipy', 'matplotlib'],
    packages=['clustering', 'estimation', 'analysis'],
    test_suite='discover_tests')
