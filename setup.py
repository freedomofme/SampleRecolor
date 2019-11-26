import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires=[
"cycler==0.10.0",
"decorator==4.4.1",
"httplib2==0.14.0",
"imageio==2.6.1",
"joblib==0.14.0",
"kiwisolver==1.1.0",
"matplotlib==3.1.1",
"networkx==2.4",
"nltk==3.4.5",
"numpy==1.17.3",
"opencv-python==4.1.2.30",
"pandas==0.25.3",
"Pillow==6.2.1",
"pyparsing==2.4.4",
"python-dateutil==2.8.1",
"pytz==2019.3",
"PyWavelets==1.1.1",
"scikit-image==0.16.2",
"scikit-learn==0.21.3",
"scipy==1.3.1",
"six==1.13.0",
"sklearn==0.0",
]


setuptools.setup(
    name="LinearRecolor", # Replace with your own username
    version="1.0.0",
    author="yelhuang",
    author_email="xiegeixiong@gmail.com",
    description="基于线性映射模板的色彩转移(Based on Image recoloring using linear template mapping)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/freedomofme/SampleRecolor",
    project_urls={
        "Bug Reports": "https://github.com/freedomofme/SampleRecolor/issues",
        "Source": "https://github.com/freedomofme/SampleRecolor",
    },
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires='>=3.6',
)