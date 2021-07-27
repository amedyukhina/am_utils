from setuptools import setup


setup(
    name='am_utils',
    version='0.1',
    url="https://github.com/amedyukhina/am_utils",
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['am_utils'],
    license='Apache License Version 2.0',
    include_package_data=True,

    test_suite='am_utils.tests',

    install_requires=[
        'scikit-image',
        'pandas',
        'numpy',
        'ddt',
        'natsort',
        'tqdm',
        'scipy',
        'ddt',
        'pytest'
      ],
 )
