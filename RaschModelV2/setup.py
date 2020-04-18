from setuptools import setup

setup(
    name='RaschModel',
    version='1.0',
    author='thaide',
    author_email='thuichap@students.kennesaw.edu',
    description='Rasch Model implementetion with tenserflow',
    install_requires=[
         'tensorflow', 'edward', 'numpy']
)
