# 019-full-color

## full-color-1.irmf

Here is one way a full-color 3D printer could interpret the materials
to represent color. In my [irmf-editor](https://github.com/gmlewis/irmf-editor),
I have added HSV, HSL, and RGB color model support. If the same material name
is listed three times with the triplet suffixes `["name.H","name.S","name.V"]`,
`["name.H","name.S","name.L"]`, or `["name.R","name.G","name.B"]`, then those
three material channels will be interpreted by the editor as three components
of a single full-color material.

Here is an example using the HSV color space.
(See [HSL and HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) for more
information about color spaces.)

![full-color-1.png](full-color-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA.H","PLA.S","PLA.V"],
  max: [5,5,3.5],
  min: [-5,-5,-3.5],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float coilSquareFace(float radius, float size, float gap, float nTurns, in vec3 xyz) {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > nTurns * (size + gap) + 0.5 * size) { return 0.0; }
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  float rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  
  // If the current point is between the coils, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  float dz = mod(xyz.z, size + gap);
  
  float lastHelixZ = angle * (size + gap);
  float coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);  // center of current coil.
    coilNum = -1.0;
  }
  float nextHelixZ = lastHelixZ + (size + gap);  // center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }

  return 1.0;
}


void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 3.0;
  float coil = coilSquareFace(4.0, 1.0, 2.0, 2.0, xyz);
  materials = vec4(xyz.z/7.0,coil,coil,1);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA.H","PLA.S","PLA.V"],
  max: [5,5,3.5],
  min: [-5,-5,-3.5],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: f32, xyz: vec3f) -> f32 {
  if (xyz.z < -0.5 * size || xyz.z > nTurns * (size + gap) + 0.5 * size) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; }
  let dz = xyz.z - (size + gap) * floor(xyz.z / (size + gap));
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap);
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 3.0;
  let coil = coilSquareFace(4.0, 1.0, 2.0, 2.0, xyz);
  return vec4f(xyz.z/7.0, coil, coil, 1.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA.H","PLA.S","PLA.V"],
  max: [5,5,3.5],
  min: [-5,-5,-3.5],
  units: "mm",
}*/

const coilSquareFace = (radius, size, gap, nTurns, xyz) => {
  // First, trivial reject on the two vertical ends of the coil.
  const zOK = xyz.z.greaterThanEqual(-0.5 * size).and(
    xyz.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );

  // Then, constrain the coil to the cylinder with wall thickness "size":
  const rxy = length(xyz.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );

  // If the current point is between the coils, return no material:
  let angle = atan(xyz.y, xyz.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle); // 0 <= angle <= 1
  const dz = mod(xyz.z, size + gap);

  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dz);
  const lastHelixZ = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextHelixZ = lastHelixZ.add(size + gap);

  const inGap = dz.greaterThan(lastHelixZ.add(0.5 * size)).and(
    dz.lessThan(nextHelixZ.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(
    floor(xyz.z.add(0.5 * size).sub(lastHelixZ).div(size + gap))
  );
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => {
  const p = xyz.add(vec3(0, 0, 3.0));
  const coil = coilSquareFace(4.0, 1.0, 2.0, 2.0, p);
  return [p.z.div(7.0), coil, coil];
};
```

* Try loading [full-color-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/019-full-color/full-color-1.irmf) now in the IRMF editor!
* Try loading [full-color-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/019-full-color/full-color-1-wgsl.irmf) now in the IRMF editor!
* Try loading [full-color-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/019-full-color/full-color-1-tsl.irmf) now in the IRMF editor!

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
