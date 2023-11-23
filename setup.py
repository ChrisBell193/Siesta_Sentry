from setuptools import setup, find_packages

setup(
    name='Siesta_Sentry',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Specify the dependencies from requirements.txt
        line.strip() for line in open('requirements.txt')
    ],
)
