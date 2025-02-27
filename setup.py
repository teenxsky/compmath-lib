from setuptools import setup, find_packages


def getLongDescription():
    with open("README.md") as file:
        return file.read()


setup(
    name="compmath",
    version="0.0.1",
    description="Another library for complex mathematical calculations!",
    long_description=getLongDescription(),
    long_description_content_type="text/markdown",
    author="Ruslan Kutorgin",
    author_email="kutorgin2002@gmail.com",
    url="https://github.com/teenxsky/compmath-lib",
    project_urls={
        "GitHub Project": "https://github.com/teenxsky/compmath-lib",
        "Issue Tracker": "https://github.com/teenxsky/compmath-lib/issues",
    },
    packages=find_packages(
        include=["compmath"],
    ),
    package_data={
        "compmath": ["src/"],
    },
    # install_requires=[
    #     "requests==2.27.1",
    # ],
    # setup_requires=[
    #     "pytest-runner",
    #     "flake8==4.0.1",
    # ],
    # tests_require=[
    #     "pytest==7.1.2",
    #     "requests-mock==1.9.3",
    # ],
    python_requires=">=3.6",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "Math",
        "Computer Science",
        "Numerical Analysis",
        "NumPy",
        "SciPy",
        "Faults",
        "Conditional Numbers",
        "Computational Mathematics",
    ],
    license="MIT",
)
