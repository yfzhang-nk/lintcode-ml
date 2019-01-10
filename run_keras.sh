#!/bin/bash

docker run -it --rm \
  $(ls /dev/nvidia* | xargs -I{} echo '--device={}') \
  $(ls /usr/lib/nvidia-390/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') \
  $(ls /usr/lib/*-linux-gnu/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') \
  -v $(pwd):/srv gw000/keras:2.1.4-py2-tf-gpu bash
