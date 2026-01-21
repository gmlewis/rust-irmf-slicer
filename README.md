# Rust [IRMF Shader](https://irmf.io) Slicer

## Summary

IRMF is a file format used to describe
[GLSL ES](https://en.wikipedia.org/wiki/OpenGL_ES) or
[WGSL](https://www.w3.org/TR/WGSL/) shaders that define the
materials in a 3D object with infinite resolution. IRMF
eliminates the need for [software slicers](https://en.wikipedia.org/wiki/Slicer_(3D_printing)),
[STL](https://en.wikipedia.org/wiki/STL_(file_format)), and
[G-code](https://en.wikipedia.org/wiki/G-code) files used in
[3D printers](https://en.wikipedia.org/wiki/3D_printing).

I believe that IRMF shaders will some day revolutionize the 3D-printing industry.

See [irmf.io](https://irmf.io) for more details.

## LYGIA support

As of 2022-10-27, support has been added for using the LYGIA Shader Library
at: https://lygia.xyz !

This means that you can add lines to your IRMF shaders like this:

```glsl
#include "lygia/math/decimation.glsl"
```

and the source will be retrieved from the LYGIA server.

Congratulations and thanks go to [Patricio Gonzalez Vivo](https://github.com/sponsors/patriciogonzalezvivo)
for making the LYGIA server available for anyone to use, and also
for the amazing tool [glslViewer](https://github.com/patriciogonzalezvivo/glslViewer)!

## About the IRMF Shader Slicer

This Rust crate is a port of the [Go IRMF Slicer](https://github.com/gmlewis/irmf-slicer)
and is a technology demonstration of how to embed an IRMF Slicer into the firmware of
a 3D printer. Included in this repo is a standalone command-line version that demonstrates
the usage of this crate.

The technology stack used is Rust and OpenGL or WebGPU.

This crate can be used by 3D printer hobbyists and manufacturers to natively support
IRMF shaders in addition to G-Code or voxel slices.

The standalone command-line program `irmf_slicer` is a program that
slices an IRMF shader model into either STL files
or into voxel slices (with various output file formats).
For STL files, it outputs one STL file per material.
(Note that some STL files can become enormous, way larger than any online
service bureau currently supports. The resolution can be reduced to limit
the STL file sizes, but at the expense of detail loss.)

For voxel slices, `irmf_slicer` can write them out to ZIP files (one ZIP file per material).
These slices can then be fed to 3D printer software that accepts
voxel slices as input for printing (such as [NanoDLP](https://www.nano3dtech.com/)).

For resin printers using either the [ChiTuBox](https://www.chitubox.com/) or
[AnyCubic](https://store.anycubic.com/collections/resin-3d-printer) slicer
(such as the [Elegoo Mars](https://us.elegoo.com/collections/mars-series)),
the `-dlp` option will output the voxel slices to the `.cbddlp` file
format (which is identical to the `.photon` file format).

Once 3D printers support IRMF shader model files directly for printing,
however, this standalone slicer will no longer be needed.

# FAQ

## How do I install it?

After you have a recent version of [Rust](https://rustup.rs/) installed,
run the following command in a terminal window:

```sh
$ cargo install irmf-slicer
```

Then you might want to try it out on some of the [example IRMF shaders](/examples).

To slice one or more `.irmf` files, just list them on the command line,
like this:

```sh
$ irmf_slicer -view -stl examples/*/*.irmf
```

The output files will be saved in the same directory as the original
input IRMF files.

## How does it work?

This slicer dices up your model (the IRMF shader) into slices (planes)
that are perpendicular (normal) to the Z (up) axis. The slices are very
thin and when stacked together, represent your solid model.

Using the `-zip` option, the result is one ZIP file per model material
with all the slices in the root of the ZIP so as to be compatible
with NanoDLP. When using the `-zip` option, the resolution is set
to X: 65, Y: 60, Z: 30 microns (unless the `-res` option is used to
override this) in order to support the `MCAST + Sylgard / 65 micron`
option of NanoDLP.

Using the `-dlp` option, the result is one `.cbddlp` file per model material
that can be loaded into the [ChiTuBox](https://www.chitubox.com/) or
[AnyCubic](https://store.anycubic.com/collections/resin-3d-printer)
slicer directly (`.cbddlp` is identical to the `.photon` file format).

Using the `-stl` option, the result is one STL file per model material.

Using the `-binvox` option, it will write one `.binvox` file per model material.

----------------------------------------------------------------------

# License

Copyright 2019 Glenn M. Lewis. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
