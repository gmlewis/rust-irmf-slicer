# 037-stanford-bunny

The Stanford Bunny is a famous 3D model and here is one version of it:
https://www.thingiverse.com/thing:88208/files

Here it is converted to IRMF using the command:

```bash
$ stl-to-irmf Bunny.stl -o bunny.irmf --fourier --res 128 --language wgsl
```

## bunny.irmf

![bunny.png](/examples/assets/037-stanford-bunny/bunny.png)

* Try loading [bunny.irmf](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/rust-irmf-slicer/blob/master/examples/037-stanford-bunny/bunny.irmf) now in the experimental IRMF editor!

----------------------------------------------------------------------

# License

Copyright 2026 Glenn M. Lewis. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
