from setuptools import setup, find_packages

# README 파일을 UTF-8 인코딩으로 읽어옵니다.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='YOLOv7_with_depthmap',
    version='1.0.0',
    description='inference YOLOv7 with scratch very simply and caculate depth also if depth_map is for input',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Na-I-Eon',
    author_email='112fkdldjs@naver.com',
    url='https://github.com/Nyan-SouthKorea/YOLOv7_with_depthmap',
    packages=find_packages(),
    install_requires=[
        'ultralytics'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
