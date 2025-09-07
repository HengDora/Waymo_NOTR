from setuptools import setup, find_packages

setup(name='deeplab2-for-wod',
      version='0.0.1',
      description='A minimal copy of DeepLab2 for Waymo Open Dataset',
      url='https://github.com/google-research/deeplab2',
      author='Waymo Open Dataset Authors',
      author_email='open-dataset@waymo.com',
      license='Apache License 2.0',
      packages=find_packages(),
      zip_safe=False
)
