# 004-coil-circle-face

## coil-circle.irmf

Once the square-face helical coil was done, it is a rather simple matter
to round the edges.

![coil-circle.png](coil-circle.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA"],
  max: [5,5,1.5],
  min: [-5,-5,-1.5],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float coilCircleFace(float radius, float size, float gap, float nTurns, in vec3 xyz) {
  // First, trivial reject on the two ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > nTurns * (size + gap) + 0.5 * size) { return 0.0; }
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  float rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  
  // If the current point is between the coils, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils
  float dz = mod(xyz.z, size + gap); // 0 <= dz <= (size+gap) between coils.
  
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
  
  // At this point, we are within the square cross-section face, so let's round the edge.
  vec3 lastHelixCenter = vec3(radius * cos(angle * 2.0 * M_PI), radius * sin(angle * 2.0 * M_PI), lastHelixZ);
  vec3 nextHelixCenter = vec3(radius * cos(angle * 2.0 * M_PI), radius * sin(angle * 2.0 * M_PI), nextHelixZ);
  vec3 testPt = vec3(xyz.xy, dz);
  float r1 = length(testPt - lastHelixCenter);
  float r2 = length(testPt - nextHelixCenter);
  if (r1 > 0.5 * size && r2 > 0.5 * size) { return 0.0; }
  
  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 1.0;
  materials[0] = coilCircleFace(3.0, 0.85, 0.15, 2.0, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA"],
  max: [5,5,1.5],
  min: [-5,-5,-1.5],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn coilCircleFace(radius: f32, size: f32, gap: f32, nTurns: f32, xyz: vec3f) -> f32 {
  // First, trivial reject on the two ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > nTurns * (size + gap) + 0.5 * size) { return 0.0; }
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  
  // If the current point is between the coils, return no material:
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils
  let dz = xyz.z - (size + gap) * floor(xyz.z / (size + gap)); // 0 <= dz <= (size+gap) between coils.
  
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);  // center of current coil.
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap);  // center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }
  
  // At this point, we are within the square cross-section face, so let's round the edge.
  let lastHelixCenter = vec3f(radius * cos(angle * 2.0 * M_PI), radius * sin(angle * 2.0 * M_PI), lastHelixZ);
  let nextHelixCenter = vec3f(radius * cos(angle * 2.0 * M_PI), radius * sin(angle * 2.0 * M_PI), nextHelixZ);
  let testPt = vec3f(xyz.xy, dz);
  let r1 = length(testPt - lastHelixCenter);
  let r2 = length(testPt - nextHelixCenter);
  if (r1 > 0.5 * size && r2 > 0.5 * size) { return 0.0; }
  
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 1.0;
  var materials = vec4f(0.0);
  materials[0] = coilCircleFace(3.0, 0.85, 0.15, 2.0, xyz);
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
  max: [5,5,1.5],
  min: [-5,-5,-1.5],
  units: "mm",
}*/

const coilCircleFace = (radius, size, gap, nTurns, xyz) => {
  const zOK = xyz.z.greaterThanEqual(-0.5 * size).and(
    xyz.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );
  const rxy = length(xyz.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );

  let angle = atan(xyz.y, xyz.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
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

  // Round the edge of the square cross-section face.
  const ang = angle.mul(2.0 * Math.PI);
  const lastHelixCenter = vec3(cos(ang).mul(radius), sin(ang).mul(radius), lastHelixZ);
  const nextHelixCenter = vec3(cos(ang).mul(radius), sin(ang).mul(radius), nextHelixZ);
  const testPt = vec3(xyz.xy, dz);
  const r1 = length(testPt.sub(lastHelixCenter));
  const r2 = length(testPt.sub(nextHelixCenter));
  const roundOK = r1.lessThanEqual(0.5 * size).or(r2.lessThanEqual(0.5 * size));

  return select(
    zOK.and(rOK).and(inGap.not()).and(coilOK).and(roundOK),
    float(1.0),
    float(0.0)
  );
};

const mainModel4 = (xyz) => [coilCircleFace(3.0, 0.85, 0.15, 2.0, xyz.add(vec3(0, 0, 1.0)))];
```

* Try loading [coil-circle.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/004-coil-circle-face/coil-circle.irmf) now in the IRMF editor!
* Try loading [coil-circle-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/004-coil-circle-face/coil-circle-wgsl.irmf) now in the IRMF editor!
* Try loading [coil-circle-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/004-coil-circle-face/coil-circle-tsl.irmf) now in the IRMF editor!

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
