#!/bin/bash
# Environment variables to match Docker build environment
export CC=clang-12
export CXX=clang++-12
export LLVM_CONFIG_PATH=/usr/bin/llvm-config-12
export LIBCLANG_PATH=/usr/lib/llvm-12/lib
export LIBCLANG_STATIC_PATH=/usr/lib/llvm-12/lib
export CLANG_PATH=/usr/bin/clang-12

echo "Environment variables set to match Docker build environment:"
echo "  CC=$CC"
echo "  CXX=$CXX"
echo "  LLVM_CONFIG_PATH=$LLVM_CONFIG_PATH"
echo "  LIBCLANG_PATH=$LIBCLANG_PATH"
echo "  LIBCLANG_STATIC_PATH=$LIBCLANG_STATIC_PATH"
echo "  CLANG_PATH=$CLANG_PATH"
echo ""
echo "Usage:"
echo "  source ./build-env.sh && cargo build"
echo "  or"
echo "  ./build-env.sh cargo build"