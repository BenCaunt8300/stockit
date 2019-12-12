from setuptools import setup

setup(
    name='stockit',
    url='https://github.com/BenCaunt8300/stockit',
    author='Ben Caunt',
    author_email='bdcaunt@gmail.com',
    author_instagram='@bencaunt1232',
    packages=['stockit'],
    install_requires=['pandas','numpy','matplotlib','tqdm','sklearn'],
    version='1.0',
    license='Apache License 2.0',
    description='python library for easy stock analysis and prediction'
)

