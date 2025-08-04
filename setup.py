from setuptools import find_packages, setup
import os
from glob import glob
from setuptools import setup

package_name = 'overtaking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include license
        (os.path.join('share', package_name), ['LICENSE']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='George_Halim',
    maintainer_email='georgehany064@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "bestFS_exe = overtaking.best_first_search.best_first_search:main"
        ],
    },
)
