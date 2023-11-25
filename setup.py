from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install',
             'torch', 'torchvision', 'torchaudio',
             '--index-url', 'https://download.pytorch.org/whl/cu118']
        )

def read_requirements():
    with open('requirements.txt', 'r') as req:
        return [line.strip() for line in req.readlines() if not line.startswith('#')]

required_packages = read_requirements()


setup(
    name= 'soccer-utils',
    version='0.1',
    packages=find_packages(),
    description='some tools to segment soccer into shorts',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Miguel Esteve',
    author_email='estevebrotons@hotmail.com',
    cmdclass={
        'install': CustomInstallCommand,
    },
    install_requires=required_packages,
    python_requires = '>= 3.9.6',
    include_package_data=True,
    zip_safe=False,
)
