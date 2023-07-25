#!/bin/bash

bwrap --dev-bind / / --bind /shared/home/ethanbro/tmp /tmp "$@"
