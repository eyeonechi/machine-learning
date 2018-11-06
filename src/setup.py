from setuptools import setup

def readme():
    with open('../README.md') as f:
        return f.read();

def license():
    with open('../LICENSE') as f:
        return f.read();

setup(
    name='machine-learning',
    version='0.0.1',
    description='Implementation of machine learning algorithms',
    long_description=readme(),
    url='https://github.com/eyeonechi/machine-learning',
    author='Ivan Ken Weng Chee',
    author_email='ichee@student.unimelb.edu.au',
    license=license(),
    keywords=[
        'machine learning'
    ],
    scripts=[],
    packages=[],
    zip_safe=False,
    include_package_data=True
)
