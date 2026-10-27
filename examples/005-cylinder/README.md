# 005-cylinder

## cylinder-1.irmf

It is time to make some Platonic solids for building blocks.

![cylinder-1.png](cylinder-1.png)

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

float cylinder(float radius, float height, in vec3 xyz) {
  // First, trivial reject on the two ends of the cylinder.
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  
  // Then, constrain radius of the cylinder:
  float rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  
  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 2.5;
  materials[0] = cylinder(2.5, 5.0, xyz);
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

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  // First, trivial reject on the two ends of the cylinder.
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  
  // Then, constrain radius of the cylinder:
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 2.5;
  var materials = vec4f(0.0);
  materials[0] = cylinder(2.5, 5.0, xyz);
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

const cylinder = (radius, height, xyz) => {
  // First, trivial reject on the two ends of the cylinder.
  const zOK = xyz.z.greaterThanEqual(0.0).and(xyz.z.lessThanEqual(height));
  // Then, constrain radius of the cylinder:
  const rxy = length(xyz.xy);
  return select(zOK.and(rxy.lessThanEqual(radius)), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => [cylinder(2.5, 5.0, xyz.add(vec3(0, 0, 2.5)))];
```

* Try loading [cylinder-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/005-cylinder/cylinder-1.irmf) now in the IRMF editor!
* Try loading [cylinder-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/005-cylinder/cylinder-1-wgsl.irmf) now in the IRMF editor!
* Try loading [cylinder-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/005-cylinder/cylinder-1-tsl.irmf) now in the IRMF editor!

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
