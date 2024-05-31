import sys

try:
    from skbuild import setup
except ImportError:
    print('Please update pip, you need pip 10 or greater,\n'
          ' or you need to install the PEP 518 requirements in pyproject.toml yourself', file=sys.stderr)
    raise

setup(
    name="vss-vision-lib",
    description="Isolated vision segmentation from vss-vision",
    packages=['vss_vision'],
    package_dir={'': 'build'},
    cmake_install_dir='build/vss_vision'
)



