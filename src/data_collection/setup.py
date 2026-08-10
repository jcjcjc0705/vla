from setuptools import find_packages, setup

package_name = 'data_collection'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='screamlab',
    maintainer_email='a0905256272@gmail.com',
    description='Run the pick-cube expert against Isaac over ROS, and record it.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'expert = data_collection.expert:main',
            'record = data_collection.expert:record_main',
        ],
    },
)
