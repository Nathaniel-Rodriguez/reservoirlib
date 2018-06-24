from setuptools import setup

setup(name='reservoirlib',
      version='0.1',
      description='Python 3 library that provides utilities for creating and'
                  ' training reservoir computers.',
      author='Nathaniel Rodriguez',
      packages=['reservoirlib'],
      url='https://github.com/Nathaniel-Rodriguez/reservoirlib.git',
      install_requires=[
          'numpy',
          'scipy'
      ],
      include_package_data=True)
