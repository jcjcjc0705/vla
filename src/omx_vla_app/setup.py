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
    description='Shared task layer: config, kinematics, cube sampling.',
    license='Apache-2.0',
    # No console_scripts on purpose -- this package is a library. The programs
    # live in data_collection (expert, record) and ml/ (convert, eval), and all
    # of them import from here so the task is described in exactly one place.
)
