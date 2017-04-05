from setuptools import setup

setup(
        name='pymalis',
        version='0.5',
        description='The MALIS loss as a simple python module.',
        url='https://github.com/funkey/pymalis',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        requires=['cython','numpy'],
        packages=['pymalis'],
        package_data={
            '': [
                'pymalis/*.h',
                'pymalis/*.hpp',
                'pymalis/*.cpp',
                'pymalis/*.pyx',
            ]
        },
        include_package_data=True,
        zip_safe=False
)
