# 003-coil-square-face

## coil-1.irmf

Another surprisingly-simple model is a helical coil with a square cross-section
face.

![coil-1.png](coil-1.png)

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
  xyz.z += 1.0;
  materials[0] = coilSquareFace(3.0, 0.85, 0.15, 2.0, xyz);
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

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: i32, trimStartAngle: f32, trimEndAngle: f32, xyz: vec3f) -> f32 {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > f32(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  
  // If the current point is between the coils, return no material:
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  let dz = xyz.z - (size + gap) * floor(xyz.z / (size + gap));
  
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap); // center of current coil.
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap); // center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  
  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= f32(nTurns)) { return 0.0; }
  
  // For the last two checks, convert angle back to radians.
  angle *= (2.0 * M_PI);
  
  // If the current point is in the first coil, use the trimStartAngle.
  if (coilNum < 1.0) { return select(0.0, 1.0, angle >= trimStartAngle); }
  
  // If the current point is in the last coil, use the trimEndAngle.
  if (coilNum >= f32(nTurns) - 1.0 && trimEndAngle > 0.0) { return select(0.0, 1.0, angle <= trimEndAngle); }
  
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 1.0;
  var materials = vec4f(0.0);
  materials[0] = coilSquareFace(3.0, 0.85, 0.15, 2, 0.5, (2.0 * M_PI) - 0.5, xyz);
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

const coilSquareFace = (radius, size, gap, nTurns, xyz) => {
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
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => [coilSquareFace(3.0, 0.85, 0.15, 2.0, xyz.add(vec3(0, 0, 1.0)))];
```

* Try loading [coil-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/003-coil-square-face/coil-1.irmf) now in the IRMF editor!
* Try loading [coil-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/003-coil-square-face/coil-1-wgsl.irmf) now in the IRMF editor!
* Try loading [coil-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/003-coil-square-face/coil-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## coil-2.irmf

It can be useful to add a `trimStartAngle` and `trimEndAngle`:

![coil-2.png](coil-2.png)

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

float coilSquareFace(in float radius, in float size, in float gap, in int nTurns, in float trimStartAngle, in float trimEndAngle, in vec3 xyz) {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > float(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  
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
    lastHelixZ -= (size + gap); // center of current coil.
    coilNum = -1.0;
  }
  float nextHelixZ = lastHelixZ + (size + gap); // center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  
  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= float(nTurns)) { return 0.0; }
  
  // For the last two checks, convert angle back to radians.
  angle *= (2.0 * M_PI);
  
  // If the current point is in the first coil, use the trimStartAngle.
  if (coilNum < 1.0) { return angle >= trimStartAngle ? 1.0 : 0.0; }
  
  // If the current point is in the last coil, use the trimEndAngle.
  if (coilNum >= float(nTurns) - 1.0 && trimEndAngle > 0.0) { return angle <= trimEndAngle ? 1.0 : 0.0; }
  
  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 1.0;
  materials[0] = coilSquareFace(3.0, 0.85, 0.15, 2, 0.5, (2.0 * M_PI) - 0.5, xyz);
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

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: i32, trimStartAngle: f32, trimEndAngle: f32, xyz: vec3f) -> f32 {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > f32(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  
  // If the current point is between the coils, return no material:
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  let dz = xyz.z - (size + gap) * floor(xyz.z / (size + gap));
  
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap); // center of current coil.
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap); // center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  
  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= f32(nTurns)) { return 0.0; }
  
  // For the last two checks, convert angle back to radians.
  let rad_angle = angle * (2.0 * M_PI);
  
  // If the current point is in the first coil, use the trimStartAngle.
  if (coilNum < 1.0) { return select(0.0, 1.0, rad_angle >= trimStartAngle); }
  
  // If the current point is in the last coil, use the trimEndAngle.
  if (coilNum >= f32(nTurns) - 1.0 && trimEndAngle > 0.0) { return select(0.0, 1.0, rad_angle <= trimEndAngle); }
  
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 1.0;
  var materials = vec4f(0.0);
  materials[0] = coilSquareFace(3.0, 0.85, 0.15, 2, 0.5, (2.0 * M_PI) - 0.5, xyz);
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

const coilSquareFace = (radius, size, gap, nTurns, trimStartAngle, trimEndAngle, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(
    p.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );
  const rxy = length(p.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );

  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dz = mod(p.z, size + gap);

  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dz);
  const lastHelixZ = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextHelixZ = lastHelixZ.add(size + gap);

  const inGap = dz.greaterThan(lastHelixZ.add(0.5 * size)).and(
    dz.lessThan(nextHelixZ.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(floor(p.z.add(0.5 * size).sub(lastHelixZ).div(size + gap)));
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));

  // For the last two checks, convert angle back to radians.
  const angleRad = angle.mul(2.0 * Math.PI);

  const startTrim = select(angleRad.greaterThanEqual(trimStartAngle), float(1.0), float(0.0));
  let trim = select(coilNum.lessThan(1.0), startTrim, float(1.0));
  if (trimEndAngle > 0.0) {
    const endTrim = select(angleRad.lessThanEqual(trimEndAngle), float(1.0), float(0.0));
    trim = select(coilNum.lessThan(1.0), startTrim,
      select(coilNum.greaterThanEqual(nTurns - 1.0), endTrim, float(1.0)));
  }

  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), trim, float(0.0));
};

const mainModel4 = (xyz) => [
  coilSquareFace(3.0, 0.85, 0.15, 2, 0.5, 2.0 * Math.PI - 0.5, xyz.add(vec3(0, 0, 1.0))),
];
```

* Try loading [coil-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/003-coil-square-face/coil-2.irmf) now in the IRMF editor!
* Try loading [coil-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/003-coil-square-face/coil-2-wgsl.irmf) now in the IRMF editor!
* Try loading [coil-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/003-coil-square-face/coil-2-tsl.irmf) now in the IRMF editor!

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
