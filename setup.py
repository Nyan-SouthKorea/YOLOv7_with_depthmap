from setuptools import setup, find_packages

# README 파일을 UTF-8 인코딩으로 읽어옵니다.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='YOLOv7_with_depthmap',
    version='1.0.1',
    description='inference YOLOv7 with scratch very simply and caculate depth also if depth_map is for input',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Na-I-Eon',
    author_email='112fkdldjs@naver.com',
    url='https://github.com/Nyan-SouthKorea/YOLOv7_with_depthmap',
    packages=find_packages(include=['YOLOv7_with_depthmap', 'YOLOv7_with_depthmap.*']),
    install_requires=[
        'matplotlib>=3.2.2',
        'numpy>=1.18.5,<1.24.0',
        'opencv-python>=4.1.1',
        'Pillow>=7.1.2',
        'PyYAML>=5.3.1',
        'requests>=2.23.0',
        'scipy>=1.4.1',
        'torch>=1.7.0,!=1.12.0',
        'torchvision>=0.8.1,!=0.13.0',
        'tqdm>=4.41.0',
        'protobuf<4.21.3',
        'tensorboard>=2.4.1',
        # 'wandb',
        'pandas>=1.1.4',
        'seaborn>=0.11.0',
        'ipython',
        'psutil',
        'thop',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        '': ['models/*', 'utils/*', 'hubconf.py', 'YOLOv7_with_depthmap.py'],
    },
)