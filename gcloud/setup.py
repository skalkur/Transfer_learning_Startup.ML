from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['pyyaml','scipy==0.18.1','scikit-learn','numpy']
setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classifier test'
)