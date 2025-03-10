import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()
    
setuptools.setup(
    name="lollms_client",
    version="0.9.1",
    author="ParisNeo",
    author_email="parisneoai@gmail.com",
    description="A client library for LoLLMs generate endpoint",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ParisNeo/lollms_client",
    packages=setuptools.find_packages(),
    install_requires=[
        'requests',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
