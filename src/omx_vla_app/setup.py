from setuptools import find_packages, setup

package_name = 'omx_vla_app'

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
    description='Run the pick-cube expert against Isaac over ROS.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'expert = omx_vla_app.expert_node:main',
            'record = omx_vla_app.expert_node:record_main',
        ],
    },
)
