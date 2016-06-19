libcacc
-------------------------------------------------------------------------------
This project ports parts of the [libacc](https://github.com/nmoehrle/libacc)
library to cuda. While construction of the acceleration structures is out of
scope, the library provides constructs to upload and use the libacc data
structures within cuda as well as some data structures to handle kernel in
and output.

Usage
-------------------------------------------------------------------------------
Add relevant headers and `.cu` files to your project and compile with
`--relocatable-device-code=true`

The location of data structures is encoded as template argument.
Uploading data to the GPU is done via the assignment operator:
```
Tree<HOST> htree = ...
Tree<DEVICE> dtree = htree;
```

Since it is preferable to access some data structures as textures it is
necessary to bind those textures before executing the kernel.
```
tracing::bind_textures(dtree.cdata());
```

Per default parts of the traversal stack are cached in shared memory. Be sure
to reduce the block size or short stack size
([tracing.h](https://github.com/nmoehrle/libcacc/blob/master/tracing.h)
`SSTACK_SIZE`) if you want to use shared memory within your kernel.

For very large trees one might have to increase the overall stack size
([tracing.h](https://github.com/nmoehrle/libcacc/blob/master/tracing.h)
`GSTACK_SIZE` + `SSTACK_SIZE`)

Requirements
-------------------------------------------------------------------------------
NVCC 7.x - C++11 support

License
-------------------------------------------------------------------------------
The software is licensed under the BSD 3-Clause license,
for more details see the LICENSE.txt file.
