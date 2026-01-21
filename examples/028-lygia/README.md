# 028-lygia

Here are examples of using the [Lygia Shader Library](https://lygia.xyz/) with [IRMF](https://irmf.io/).

## lygia-01.irmf

![lygia-01.png](/examples/assets/028-lygia/lygia-01.png)

```glsl
/*{
  irmf: "1.0",
  materials: ["jasper","sapphire","agate","emerald"],
  max: [3,3,3],
  min: [-3,-3,-3],
  units: "mm",
}*/

#define FNC_SATURATE
#include "lygia/sdf.glsl"

float sdf2mat(in float a) {
  return smoothstep(0.010, 0.009, a);
}

vec4 sdfs(in vec3 pos) {
  vec4 res = vec4(0);

  res.x += sdf2mat(sphereSDF(pos - vec3(0.0, 0.60, 0.0), 0.5));
  res.y += sdf2mat(boxSDF(pos - vec3(2.0, 0.5, 0.0), vec3(0.4)));
  res.z += sdf2mat(torusSDF(pos - vec3(0.0, 0.5, 2.0), vec2(0.4, 0.1)));
  res.w += sdf2mat(capsuleSDF(pos, vec3(-2.3, 0.4, - 0.2), vec3(-1.6, 0.75, 0.2), 0.2));
  res.x += sdf2mat(triPrismSDF(pos - vec3(-2.0, 0.50, - 2.0), vec2(0.5, 0.1)));
  res.y += sdf2mat(cylinderSDF(pos - vec3(2.0, 0.50, - 2.0), vec2(0.2, 0.4)));
  res.z += sdf2mat(coneSDF(pos - vec3(0.0, 0.75, - 2.0), vec3(0.8, 0.6, 0.6)));
  res.w += sdf2mat(hexPrismSDF(pos - vec3(-2.0, 0.60, 2.0), vec2(0.5, 0.1)));
  res.x += sdf2mat(pyramidSDF(pos - vec3(2.0, 0.10, 2.0), 1.0));

  return res;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials = sdfs(xyz);
}
```

* Try loading [lygia-01.irmf](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/028-lygia/lygia-01.irmf) now in the experimental IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

----------------------------------------------------------------------

# License

Copyright 2022 Glenn M. Lewis. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
