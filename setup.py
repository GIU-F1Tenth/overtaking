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
    maintainer='George Halim',
    maintainer_email='georgehany064@gmail.com',
    description='Python package for high speed overtaking',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "bestFS_exe = best_first_search.best_first_search_node:main",
            "potential_field_exe = potential_field.potential_field_node:main",
            "dwa_exe = dynamic_window_approach.dwa:main"
        ],
    },
)
