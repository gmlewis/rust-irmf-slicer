# 035-the-thinker

The mascott for the amazing Berkeley Computed Axial Lithography (CAL)
Volumetric Additive Manufacturing (VAM) 3D printer project is
Rodin's "The Thinker".

Here is their mascott converted to IRMF using the command:

```bash
$ stl-to-irmf -o the-thinker-uncompressed.irmf the-thinker.stl
$ compress-irmf --base64 -o the-thinker.irmf the-thinker-uncompressed.irmf
```

## the-thinker.irmf

![the-thinker.png](/examples/assets/035-the-thinker/the-thinker.png)

* Try loading [the-thinker.irmf](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/rust-irmf-slicer/blob/master/examples/035-the-thinker/the-thinker.irmf) now in the experimental IRMF editor!

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
