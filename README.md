# Sample Zephyr App using Kenning Zephyr Runtime for inference

Copyright (c) 2025 [Antmicro](https://www.antmicro.com)

This is a sample standalone [Zephyr](https://www.zephyrproject.org/) app that uses Kenning Zephyr Runtime library for inference of any model.

## Building the project

### Cloning the project

First, clone the project to provide all the necessary files for building the application and Docker image:

```
git clone git@github.com:antmicro/kenning-zephyr-runtime-example-app.git
cd kenning-zephyr-runtime-example-app
```

### Installing dependencies

One way to obtain dependencies is to use the Docker image:

```
docker build -t kenning-zephyr-runtime-app -f environments/Dockerfile .
docker run --rm -it -v $(pwd):$(pwd) -w $(pwd) kenning-zephyr-runtime-app /bin/bash
```

To set things locally, install:

* [Zephyr dependencies](https://docs.zephyrproject.org/latest/develop/getting_started/index.html#install-dependencies)
* `jq`
* `curl`
* `west`
* `patch`
* `CMake`

On Debian-based systems, run:

```bash
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  ccache cmake colorized-logs curl device-tree-compiler dfu-util file fonts-lato \
  g++-multilib gcc gcc-multilib git jq libmagic1 libsdl2-dev make ninja-build \
  python3-dev python3-pip python3-setuptools python3-tk python3-wheel python3-venv \
  mono-complete wget xxd xz-utils patch

python3 -m venv .venv
source .venv/bin/activate
pip3 install west pyelftools
```

### Downloading Zephyr SDK and necessary build dependencies

With all the system dependencies satisfied, you need to obtain all Zephyr repositories and additional tools.
To do so, initialize west and install Kenning Zephyr Runtime dependencies:

```bash
west init -l app
west update
west zephyr-export
```

Then, prepare KZR modules and dependencies:

```bash
python3 -m pip install -r zephyr/scripts/requirements.txt
python3 -m pip install -r kenning-zephyr-runtime/requirements.txt
pushd ./kenning-zephyr-runtime
./scripts/prepare_modules.sh
popd
```

In the end, set up the Zephyr SDK (this step can be omitted in the Docker image).

```bash
ZEPHYR_SDK_ONLY=1 ./kenning-zephyr-runtime/scripts/prepare_zephyr_env.sh
```

### Building the application

In addition to providing a platform, the app requires:

* A runtime - `-DEXTRA_CONF_FILE` option, can be `tvm.conf` or `tflite.conf`
* A model - `-DCONFIG_KENNING_MODEL_PATH` option, can be URL to model or existing local path.

To build an app using TVM as backend, run:

```bash
west build \
  -p always \
  -b max32690evkit/max32690/m4 app -- \
  -DEXTRA_CONF_FILE="tvm.conf" \
  -DCONFIG_KENNING_MODEL_PATH=\"https://dl.antmicro.com/kenning/models/classification/magic_wand.h5\"
west build -t board-repl
```

To build an app using TFLite as backend, run:

```bash
west build \
  -p always \
  -b max32690evkit/max32690/m4 app -- \
  -DEXTRA_CONF_FILE="tflite.conf" \
  -DCONFIG_KENNING_MODEL_PATH=\"https://dl.antmicro.com/kenning/models/classification/magic_wand.h5\"
west build -t board-repl
```

### Running the app

To run the app on a simulated platform, first download [Renode](https://github.com/renode/renode) portable (this step can be omitted in the Docker container):

```bash
source kenning-zephyr-runtime/scripts/prepare_renode.sh
```

To start the simulation, run:

```bash skip
python3 kenning-zephyr-runtime/scripts/run_renode.py --no-kcomm
```

To stop the simulation, use `Ctrl-C`.

### Profiling the app with Zephelin

[Zephelin](https://antmicro.github.io/zephelin) is a library for profiling Zephyr applications.

To use it, first, install dependencies and prepare modules:

```bash
python3 -m pip install -r zephelin/requirements.txt
west forall -c "git reset HEAD --hard && git clean -fd"
west patch -sm zephelin apply
```

Then, build the app with additional `zpl.conf` configuration file and a flag enabling tracing for a specific framework:

```bash
west build \
  -p always \
  -b max32690fthr/max32690/m4 app -- \
  -DEXTRA_CONF_FILE="tflite.conf;zpl.conf" \
  -DCONFIG_KENNING_MODEL_PATH=\"https://dl.antmicro.com/kenning/models/classification/magic_wand.h5\" \
  -DCONFIG_ZPL_TFLM_PROFILER=y
west build -t board-repl
```

If the platform is simulated, ensure [Renode](https://github.com/renode/renode) is prepared (this step is not required if you did it previously in the current environment):

```bash
source kenning-zephyr-runtime/scripts/prepare_renode.sh
```

For example, capturing the trace for 10 seconds can be using the following command:

```bash
python zephelin/scripts/run_renode.py --trace-output ./channel0_0 --timeout 10 --repl build/max32690fthr.repl
```

Using Zephelin CLI, you can create a [TEF](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw) trace from the output:

```bash
west zpl-prepare-trace ./channel0_0 \
  --tflm-model-path build/modules/kenning-zephyr-runtime/lib/kenning_inference_lib/runtimes/tflite/generated/model.tflite \
  -o ./tef_tflm_profiler.json
```

This trace can be visualized in [Zephelin Trace Viewer](https://github.com/antmicro/zephelin-trace-viewer).
