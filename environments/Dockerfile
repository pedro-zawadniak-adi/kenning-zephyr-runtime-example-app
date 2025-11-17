FROM ubuntu:22.04

ENV INST 'apt-get install -y --no-install-recommends'
ENV PIPINST 'python3 -m pip install --no-cache-dir --upgrade'

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && $INST \
    ccache \
    cmake \
    curl \
    device-tree-compiler \
    dfu-util \
    file \
    fonts-lato \
    gcc \
    gcc-multilib \
    git \
    g++-multilib \
    jq \
    libdbus-1-3 \
    libfontconfig \
    libgl1 \
    libgssapi-krb5-2 \
    libhidapi-hidraw0 \
    libhidapi-libusb0 \
    libmagic1 \
    libncurses5 \
    libsdl2-dev \
    libusb-0.1-4 \
    libxcb1 \
    libxcb-glx0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-util1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    make \
    ninja-build \
    patch \
    picocom \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-tk \
    python3-venv \
    python3-wheel \
    wget \
    xxd \
    xz-utils \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb

RUN apt-get update && $INST dotnet-sdk-9.0 && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/renode-portable-dotnet.tar.gz https://builds.renode.io/renode-1.16.0+20250808gitb64a2d67c.linux-portable-dotnet.tar.gz && \
    mkdir -p /opt/renode/ && \
    tar -C /opt/renode --strip-components=1 -xzvf /tmp/renode-portable-dotnet.tar.gz && rm /tmp/renode-portable-dotnet.tar.gz

ENV PYRENODE_RUNTIME=coreclr
ENV PYRENODE_BIN=/opt/renode/renode

RUN $PIPINST \
    pip \
    setuptools \
    west \
    pyelftools

RUN wget -O /tmp/prepare_zephyr_env.sh https://raw.githubusercontent.com/antmicro/kenning-zephyr-runtime/389361d59ea42a0f6eda36c6743b15ff5a01044d/scripts/prepare_zephyr_env.sh && \
    chmod +x /tmp/prepare_zephyr_env.sh && \
    ZEPHYR_SDK_ONLY=1 ZEPHYR_SDK_PATH=/opt/zephyr-sdk /tmp/prepare_zephyr_env.sh && \
    rm /tmp/prepare_zephyr_env.sh

RUN wget -O /tmp/requirements.txt https://raw.githubusercontent.com/zephyrproject-rtos/zephyr/d330e86e033f94c8b0a1a7a3be55c2553713a350/scripts/requirements-base.txt && \
    $PIPINST -r /tmp/requirements.txt && \
    wget -O /tmp/requirements.txt https://raw.githubusercontent.com/zephyrproject-rtos/zephyr/d330e86e033f94c8b0a1a7a3be55c2553713a350/scripts/requirements-build-test.txt && \
    $PIPINST -r /tmp/requirements.txt && \
    wget -O /tmp/requirements.txt https://raw.githubusercontent.com/zephyrproject-rtos/zephyr/d330e86e033f94c8b0a1a7a3be55c2553713a350/scripts/requirements-run-test.txt && \
    $PIPINST -r /tmp/requirements.txt && \
    $PIPINST \
        cmake \
        importlib_metadata \
        pyelftools \
        pytest \
        pytest-reportlog \
        west && \
    rm /tmp/requirements.txt

ENV ZEPHYR_SDK_INSTALL_DIR /opt/zephyr-sdk
ENV ZEPHYR_SDK_PATH /opt/zephyr-sdk

RUN $PIPINST \
    "kenning @ git+https://github.com/antmicro/kenning.git@6c6d48a8d4846b0e0eeba363da269de1e7ce65c0" \
    pandas==2.3.1 \
    pyod==2.0.5 \
    iree-compiler==20230209.425 \
    iree-runtime==20230209.425 \
    iree-tools-tf==20230209.425 \
    iree-tools-tflite==20230209.425 \
    apache-tvm==0.14.dev273 \
    onnx-tf==1.10.0 \
    tensorflow-addons==0.23.0 \
    tensorflow-model-optimization==0.8.0 \
    tensorflow==2.11.1 \
    tensorflow-probability==0.19.0 \
    tf2onnx==1.14.0 \
    tflite==2.18.0 \
    onnx2torch==1.5.15 \
    torch==2.3.1 \
    torchvision==0.18.1 \
    Jinja2==3.1.6 \
    "servis[bokeh,matplotlib] @ git+https://github.com/antmicro/servis.git@e774d246b791b8223bd0bef45305a0e092b42b09" \
    "antmicro_sphinx_utils @ git+https://github.com/antmicro/antmicro-sphinx-utils.git@8907a689189de91ec2fcedd1ab8088f123bcf0ab" \
    docutils==0.21.2 \
    sphinxcontrib-napoleon==0.7 \
    Sphinx==8.1.3 \
    "pyrenode3 @ git+https://github.com/antmicro/pyrenode3.git@0b23dca4324f41682efd0c96bef4bd3ac8cb984f" \
    "renode-run @ git+https://github.com/antmicro/renode-run.git@b00d691a51ea4a1ab0dc07f13911bd7032678ea1" \
    pyserial==3.5 \
    "autoPyTorch @ git+https://github.com/antmicro/auto-pytorch.git@69e3649ef34090e4fb806579858f558a84eb1499" \
    ConfigSpace==1.2.1 \
    smac==1.4.0
    # "dts2repl @ git+https://github.com/antmicro/dts2repl@83de3bb32677dc10472c33c4372d0402f1b28911" \
    # "pyrfr @ git+https://github.com/antmicro/random-forest-run.git@502afa7d3e0c7591df1cd9ef028a3971177491a0"
