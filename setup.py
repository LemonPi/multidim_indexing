from setuptools import setup, find_packages

setup(
    name='multidim-indexing',
    version='0.0.2',
    packages=find_packages(),
    url='',
    license='',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='',
    test_suite='pytest',
    tests_require=[
        'pytest',
    ], install_requires=['numpy', 'torch']
)
