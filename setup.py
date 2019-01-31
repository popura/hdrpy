from setuptools import setup, find_packages

setup(
    name='hdrpy',
    version='0.0.2',
    description='A package for handling high dynamic range images',
    long_description=readme,
    author='Yuma Kinoshita',
    url='https://github.com/popura/hdrpy',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'scipy', 'opencv-python',
                      'OpenEXR', 'colour-science', 'colour-hdri']
)
