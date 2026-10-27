# 023-infill

3D printing infill can also be easily accomplished with an IRMF shader.

## gyroid-1.irmf

This is a gyroid structure as defined on [Wikipedia](https://en.wikipedia.org/wiki/Gyroid):

![sin x cos y + sin y cos z + sin z cos x = 0](gyroid-equation.png)

![gyroid-1.png](gyroid-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA"],
  max: [5,5,5],
  min: [-5,-5,-5],
  units: "mm",
}*/

float gyroid(in float scale, in vec3 xyz) {
  xyz /= scale;
  float v = sin(xyz.x) * cos(xyz.y) + sin(xyz.y) * cos(xyz.z) + sin(xyz.z) * cos(xyz.x);
  return abs(v) < 0.2 ? 1.0 : 0.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials[0] = gyroid(0.5, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA"],
  max: [5,5,5],
  min: [-5,-5,-5],
  units: "mm",
}*/

fn gyroid(scale: f32, xyz_in: vec3f) -> f32 {
  let xyz = xyz_in / scale;
  let v = sin(xyz.x) * cos(xyz.y) + sin(xyz.y) * cos(xyz.z) + sin(xyz.z) * cos(xyz.x);
  return select(0.0, 1.0, abs(v) < 0.2);
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = gyroid(0.5, xyz);
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA"],
  max: [5,5,5],
  min: [-5,-5,-5],
  units: "mm",
}*/

const gyroid = (scale, xyz) => {
  const p = xyz.div(scale);
  const v = sin(p.x).mul(cos(p.y))
    .add(sin(p.y).mul(cos(p.z)))
    .add(sin(p.z).mul(cos(p.x)));
  return select(abs(v).lessThan(0.2), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => [gyroid(0.5, xyz)];
```

* Try loading [gyroid-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/023-infill/gyroid-1.irmf) now in the IRMF editor!
* Try loading [gyroid-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/023-infill/gyroid-1-wgsl.irmf) now in the IRMF editor!
* Try loading [gyroid-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/023-infill/gyroid-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

----------------------------------------------------------------------

# License

Copyright 2020 Glenn M. Lewis. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
