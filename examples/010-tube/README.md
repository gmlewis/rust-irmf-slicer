# 010-tube

## tube-1.irmf

Another surprisingly-simple model is a tube which is a cylinder with an inner
radius and an outer radius.

![tube-1.png](tube-1.png)

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

float tube(float innerRadius, float outerRadius, float height, in vec3 xyz) {
  // First, trivial reject on the two ends of the tube.
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  
  // Then, constrain the tube to the inner and outer radii.
  float rxy = length(xyz.xy);
  if (rxy < innerRadius || rxy > outerRadius) { return 0.0; }
  
  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 5.0;
  materials[0] = tube(4.0, 5.0, 10.0, xyz);
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

fn tube(innerRadius: f32, outerRadius: f32, height: f32, xyz: vec3f) -> f32 {
  // First, trivial reject on the two ends of the tube.
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  
  // Then, constrain the tube to the inner and outer radii.
  let rxy = length(xyz.xy);
  if (rxy < innerRadius || rxy > outerRadius) { return 0.0; }
  
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 5.0;
  var materials = vec4f(0.0);
  materials[0] = tube(4.0, 5.0, 10.0, xyz);
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

const tube = (innerRadius, outerRadius, height, xyz) => {
  // First, trivial reject on the two ends of the tube.
  const zOK = xyz.z.greaterThanEqual(0.0).and(xyz.z.lessThanEqual(height));
  // Then, constrain the tube to the inner and outer radii.
  const rxy = length(xyz.xy);
  const rOK = rxy.greaterThanEqual(innerRadius).and(rxy.lessThanEqual(outerRadius));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => [tube(4.0, 5.0, 10.0, xyz.add(vec3(0, 0, 5.0)))];
```

* Try loading [tube-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/010-tube/tube-1.irmf) now in the IRMF editor!
* Try loading [tube-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/010-tube/tube-1-wgsl.irmf) now in the IRMF editor!
* Try loading [tube-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/010-tube/tube-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

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
