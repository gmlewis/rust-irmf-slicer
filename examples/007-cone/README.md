# 007-cone

## cone-1.irmf

Here is a unit cone.

![cone-1.png](cone-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA"],
  max: [1,1,0.5],
  min: [-1,-1,-0.5],
  units: "mm",
}*/

float cone(in vec3 xyz) {
  // Trivially reject above and below the cone.
  if (xyz.z < 0.0 || xyz.z > 1.0) { return 0.0; }
  
  // Calculate the new size based on the height.
  float zsize = mix(1.0, 0.0, xyz.z);
  float r = length(xyz.xy);
  
  if (r > zsize) { return 0.0; }
  
  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 0.5;
  materials[0] = cone(xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA"],
  max: [1,1,0.5],
  min: [-1,-1,-0.5],
  units: "mm",
}*/

fn cone(xyz: vec3f) -> f32 {
  // Trivially reject above and below the cone.
  if (xyz.z < 0.0 || xyz.z > 1.0) { return 0.0; }
  
  // Calculate the new size based on the height.
  let zsize = mix(1.0, 0.0, xyz.z);
  let r = length(xyz.xy);
  
  if (r > zsize) { return 0.0; }
  
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 0.5;
  var materials = vec4f(0.0);
  materials[0] = cone(xyz);
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
  max: [1,1,0.5],
  min: [-1,-1,-0.5],
  units: "mm",
}*/

const cone = (xyz) => {
  // Trivially reject above and below the cone.
  const zOK = xyz.z.greaterThanEqual(0.0).and(xyz.z.lessThanEqual(1.0));
  // Calculate the new size based on the height.
  const zsize = mix(float(1.0), float(0.0), xyz.z);
  const r = length(xyz.xy);
  return select(zOK.and(r.lessThanEqual(zsize)), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => [cone(xyz.add(vec3(0, 0, 0.5)))];
```

* Try loading [cone-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/007-cone/cone-1.irmf) now in the IRMF editor!
* Try loading [cone-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/007-cone/cone-1-wgsl.irmf) now in the IRMF editor!
* Try loading [cone-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/007-cone/cone-1-tsl.irmf) now in the IRMF editor!

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
