from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'roa_vision_pipeline'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),

        # install config and launch files into share
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='robit',
    maintainer_email='leokim0503@kw.ac.kr',
    description='Detection refinement and ball tracking for the RO:BIT vision pipeline',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'detection_refiner_node = roa_vision_pipeline.detection_refiner.node:main',
            'ball_tracker_node = roa_vision_pipeline.ball_tracker.node:main',
        ],
    },
)
