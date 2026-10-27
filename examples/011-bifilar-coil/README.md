# 011-bifilar-coil

## bifilar-coil-1.irmf

Another surprisingly-simple model is a bifilar-coil which is two spirals.

![bifilar-coil-1.png](bifilar-coil-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper","copper"],
  max: [22,22,0.425],
  min: [-22,-22,-0.425],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float spiralSquareFace(float startRadius, float size, float gap, float nTurns, in vec3 xyz) {
  // First, trivial reject above and below the spiral.
  if (xyz.z < -0.5 * size || xyz.z > 0.5 * size) { return 0.0; }
  
  float r = length(xyz.xy);
  if (r < startRadius - 0.5 * size || r > startRadius + 0.5 * size + (size + gap) * nTurns) { return 0.0; }
  
  // If the current point is between the spirals, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between spirals from center to center.
  float dr = mod(r - startRadius, size + gap); // 0 <= dr <= (size+gap) between spirals from center to center.
  
  float coilNum = 0.0;
  float lastSpiralR = angle * (size + gap);
  if (lastSpiralR > dr) {
    lastSpiralR -= (size + gap);  // center of current coil.
    coilNum = -1.0;
  }
  float nextSpiralR = lastSpiralR + (size + gap);  // center of next outer coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dr > lastSpiralR + 0.5 * size && dr < nextSpiralR - 0.5 * size) { return 0.0; }
  
  coilNum += floor((r - startRadius + (0.5 * size) - lastSpiralR) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }

  return 1.0;
}

mat3 rotAxis(vec3 axis, float a) {
  // This is from: http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
  float s = sin(a);
  float c = cos(a);
  float oc = 1.0 - c;
  vec3 as = axis * s;
  mat3 p = mat3(axis.x * axis, axis.y * axis, axis.z * axis);
  mat3 q = mat3(c, - as.z, as.y, as.z, c, - as.x, - as.y, as.x, c);
  return p * oc + q;
}

vec2 bifilarCoil(float startRadius, float size, float gap, float nTurns, in vec3 xyz) {
  float coil1 = spiralSquareFace(startRadius, size, (size + 2.0 * gap), nTurns, xyz);
  mat4 rot180Z = mat4(rotAxis(vec3(0, 0, 1), M_PI));
  xyz = (vec4(xyz, 1.0) * rot180Z).xyz;
  float coil2 = spiralSquareFace(startRadius, size, (size + 2.0 * gap), nTurns, xyz);
  return vec2(coil1, coil2);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials.xy = bifilarCoil(3.0, 0.85, 0.15, 9.0, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper","copper"],
  max: [22,22,0.425],
  min: [-22,-22,-0.425],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn spiralSquareFace(startRadius: f32, size: f32, gap: f32, nTurns: f32, xyz: vec3f) -> f32 {
  // First, trivial reject above and below the spiral.
  if (xyz.z < -0.5 * size || xyz.z > 0.5 * size) { return 0.0; }
  
  let r = length(xyz.xy);
  if (r < startRadius - 0.5 * size || r > startRadius + 0.5 * size + (size + gap) * nTurns) { return 0.0; }
  
  // If the current point is between the spirals, return no material:
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between spirals from center to center.
  let dr = (r - startRadius) - (size + gap) * floor((r - startRadius) / (size + gap)); // 0 <= dr <= (size+gap) between spirals from center to center.
  
  var coilNum = 0.0;
  var lastSpiralR = angle * (size + gap);
  if (lastSpiralR > dr) {
    lastSpiralR -= (size + gap);  // center of current coil.
    coilNum = -1.0;
  }
  let nextSpiralR = lastSpiralR + (size + gap);  // center of next outer coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dr > lastSpiralR + 0.5 * size && dr < nextSpiralR - 0.5 * size) { return 0.0; }
  
  coilNum += floor((r - startRadius + (0.5 * size) - lastSpiralR) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }

  return 1.0;
}

fn rotAxis(axis: vec3f, a: f32) -> mat3x3f {
  let s = sin(a);
  let c = cos(a);
  let oc = 1.0 - c;
  let as_ = axis * s;
  let p = mat3x3f(axis.x * axis, axis.y * axis, axis.z * axis);
  let q = mat3x3f(c, as_.z, -as_.y, -as_.z, c, as_.x, as_.y, -as_.x, c);
  return p * oc + q;
}

fn mat3ToMat4(m: mat3x3f) -> mat4x4f {
  return mat4x4f(
    vec4f(m[0], 0.0),
    vec4f(m[1], 0.0),
    vec4f(m[2], 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0)
  );
}

fn bifilarCoil(startRadius: f32, size: f32, gap: f32, nTurns: f32, xyz: vec3f) -> vec2f {
  let coil1 = spiralSquareFace(startRadius, size, (size + 2.0 * gap), nTurns, xyz);
  let rot180Z = mat3ToMat4(rotAxis(vec3f(0, 0, 1), M_PI));
  let xyz180Z = (rot180Z * vec4f(xyz, 1.0)).xyz;
  let coil2 = spiralSquareFace(startRadius, size, (size + 2.0 * gap), nTurns, xyz180Z);
  return vec2f(coil1, coil2);
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  let coils = bifilarCoil(3.0, 0.85, 0.15, 9.0, xyz);
  materials[0] = coils.x;
  materials[1] = coils.y;
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper","copper"],
  max: [22,22,0.425],
  min: [-22,-22,-0.425],
  units: "mm",
}*/


const spiralSquareFace = (startRadius, size, gap, nTurns, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(p.z.lessThanEqual(0.5 * size));
  const r = length(p.xy);
  const rOK = r.greaterThanEqual(startRadius - 0.5 * size).and(
    r.lessThanEqual(startRadius + 0.5 * size + (size + gap) * nTurns)
  );
  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dr = mod(r.sub(startRadius), size + gap);
  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dr);
  const lastSpiralR = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextSpiralR = lastSpiralR.add(size + gap);
  const inGap = dr.greaterThan(lastSpiralR.add(0.5 * size)).and(
    dr.lessThan(nextSpiralR.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(
    floor(r.sub(startRadius).add(0.5 * size).sub(lastSpiralR).div(size + gap))
  );
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), float(1.0), float(0.0));
};

const bifilarCoil = (startRadius, size, gap, nTurns, p) => {
  const g = size + 2.0 * gap;
  const c1 = spiralSquareFace(startRadius, size, g, nTurns, p);
  // 180-degree rotation about Z: (x,y,z) -> (-x,-y,z).
  const p180 = vec3(p.x.negate(), p.y.negate(), p.z);
  const c2 = spiralSquareFace(startRadius, size, g, nTurns, p180);
  return [c1, c2];
};

const mainModel4 = (xyz) => bifilarCoil(3.0, 0.85, 0.15, 9.0, xyz);
```

* Try loading [bifilar-coil-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/011-bifilar-coil/bifilar-coil-1.irmf) now in the IRMF editor!
* Try loading [bifilar-coil-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/011-bifilar-coil/bifilar-coil-1-wgsl.irmf) now in the IRMF editor!
* Try loading [bifilar-coil-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/011-bifilar-coil/bifilar-coil-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## bifilar-coil-2.irmf

Let's take bifilar-coil-1 above and add in a dielectric between the copper wires.

![bifilar-coil-2.png](bifilar-coil-2.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper","copper","dielectric"],
  max: [22,22,0.425],
  min: [-22,-22,-0.425],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float spiralSquareFace(float startRadius, float size, float gap, float nTurns, in vec3 xyz) {
  // First, trivial reject above and below the spiral.
  if (xyz.z < -0.5 * size || xyz.z > 0.5 * size) { return 0.0; }
  
  float r = length(xyz.xy);
  if (r < startRadius - 0.5 * size || r > startRadius + 0.5 * size + (size + gap) * nTurns) { return 0.0; }
  
  // If the current point is between the spirals, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between spirals from center to center.
  float dr = mod(r - startRadius, size + gap); // 0 <= dr <= (size+gap) between spirals from center to center.
  
  float coilNum = 0.0;
  float lastSpiralR = angle * (size + gap);
  if (lastSpiralR > dr) {
    lastSpiralR -= (size + gap);  // center of current coil.
    coilNum = -1.0;
  }
  float nextSpiralR = lastSpiralR + (size + gap);  // center of next outer coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dr > lastSpiralR + 0.5 * size && dr < nextSpiralR - 0.5 * size) { return 0.0; }
  
  coilNum += floor((r - startRadius + (0.5 * size) - lastSpiralR) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }

  return 1.0;
}

mat3 rotAxis(vec3 axis, float a) {
  // This is from: http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
  float s = sin(a);
  float c = cos(a);
  float oc = 1.0 - c;
  vec3 as = axis * s;
  mat3 p = mat3(axis.x * axis, axis.y * axis, axis.z * axis);
  mat3 q = mat3(c, - as.z, as.y, as.z, c, - as.x, - as.y, as.x, c);
  return p * oc + q;
}

float cylinder(float radius, float height, in vec3 xyz) {
  // First, trivial reject on the two ends of the cylinder.
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  
  // Then, constrain radius of the cylinder:
  float rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  
  return 1.0;
}

vec3 bifilarCoilWithDielectric(float startRadius, float size, float gap, float nTurns, in vec3 xyz) {
  float coil1 = spiralSquareFace(startRadius, size, (size + 2.0 * gap), nTurns, xyz);
  mat4 rot180Z = mat4(rotAxis(vec3(0, 0, 1), M_PI));
  vec3 xyz180Z = (vec4(xyz, 1.0) * rot180Z).xyz;
  float coil2 = spiralSquareFace(startRadius, size, (size + 2.0 * gap), nTurns, xyz180Z);
  float dielectric = cylinder(startRadius + size + 2.0 * (size + gap) * nTurns, size, xyz+vec3(0,0,.5*size));
  dielectric -= (coil1 + coil2); // Don't put dielectric where the metal is.
  return vec3(coil1, coil2, dielectric);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials.xyz = bifilarCoilWithDielectric(3.0, 0.85, 0.15, 9.0, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper","copper","dielectric"],
  max: [22,22,0.425],
  min: [-22,-22,-0.425],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn spiralSquareFace(startRadius: f32, size: f32, gap: f32, nTurns: f32, xyz: vec3f) -> f32 {
  // First, trivial reject above and below the spiral.
  if (xyz.z < -0.5 * size || xyz.z > 0.5 * size) { return 0.0; }
  
  let r = length(xyz.xy);
  if (r < startRadius - 0.5 * size || r > startRadius + 0.5 * size + (size + gap) * nTurns) { return 0.0; }
  
  // If the current point is between the spirals, return no material:
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between spirals from center to center.
  let dr = (r - startRadius) - (size + gap) * floor((r - startRadius) / (size + gap)); // 0 <= dr <= (size+gap) between spirals from center to center.
  
  var coilNum = 0.0;
  var lastSpiralR = angle * (size + gap);
  if (lastSpiralR > dr) {
    lastSpiralR -= (size + gap);  // center of current coil.
    coilNum = -1.0;
  }
  let nextSpiralR = lastSpiralR + (size + gap);  // center of next outer coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dr > lastSpiralR + 0.5 * size && dr < nextSpiralR - 0.5 * size) { return 0.0; }
  
  coilNum += floor((r - startRadius + (0.5 * size) - lastSpiralR) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }

  return 1.0;
}

fn rotAxis(axis: vec3f, a: f32) -> mat3x3f {
  let s = sin(a);
  let c = cos(a);
  let oc = 1.0 - c;
  let as_ = axis * s;
  let p = mat3x3f(axis.x * axis, axis.y * axis, axis.z * axis);
  let q = mat3x3f(c, as_.z, -as_.y, -as_.z, c, as_.x, as_.y, -as_.x, c);
  return p * oc + q;
}

fn mat3ToMat4(m: mat3x3f) -> mat4x4f {
  return mat4x4f(
    vec4f(m[0], 0.0),
    vec4f(m[1], 0.0),
    vec4f(m[2], 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0)
  );
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  // First, trivial reject on the two ends of the cylinder.
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  
  // Then, constrain radius of the cylinder:
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  
  return 1.0;
}

fn bifilarCoilWithDielectric(startRadius: f32, size: f32, gap: f32, nTurns: f32, xyz: vec3f) -> vec3f {
  let coil1 = spiralSquareFace(startRadius, size, (size + 2.0 * gap), nTurns, xyz);
  let rot180Z = mat3ToMat4(rotAxis(vec3f(0, 0, 1), M_PI));
  let xyz180Z = (rot180Z * vec4f(xyz, 1.0)).xyz;
  let coil2 = spiralSquareFace(startRadius, size, (size + 2.0 * gap), nTurns, xyz180Z);
  let dielectric = cylinder(startRadius + size + 2.0 * (size + gap) * nTurns, size, xyz+vec3f(0,0,0.5*size));
  return vec3f(coil1, coil2, clamp(dielectric - (coil1 + coil2), 0.0, 1.0));
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  let coils = bifilarCoilWithDielectric(3.0, 0.85, 0.15, 9.0, xyz);
  materials[0] = coils.x;
  materials[1] = coils.y;
  materials[2] = coils.z;
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper","copper","dielectric"],
  max: [22,22,0.425],
  min: [-22,-22,-0.425],
  units: "mm",
}*/


const spiralSquareFace = (startRadius, size, gap, nTurns, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(p.z.lessThanEqual(0.5 * size));
  const r = length(p.xy);
  const rOK = r.greaterThanEqual(startRadius - 0.5 * size).and(
    r.lessThanEqual(startRadius + 0.5 * size + (size + gap) * nTurns)
  );
  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dr = mod(r.sub(startRadius), size + gap);
  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dr);
  const lastSpiralR = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextSpiralR = lastSpiralR.add(size + gap);
  const inGap = dr.greaterThan(lastSpiralR.add(0.5 * size)).and(
    dr.lessThan(nextSpiralR.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(
    floor(r.sub(startRadius).add(0.5 * size).sub(lastSpiralR).div(size + gap))
  );
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), float(1.0), float(0.0));
};

const cylinder = (radius, height, p) => {
  const zOK = p.z.greaterThanEqual(0.0).and(p.z.lessThanEqual(height));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const bifilarCoilWithDielectric = (startRadius, size, gap, nTurns, p) => {
  const g = size + 2.0 * gap;
  const c1 = spiralSquareFace(startRadius, size, g, nTurns, p);
  const p180 = vec3(p.x.negate(), p.y.negate(), p.z);
  const c2 = spiralSquareFace(startRadius, size, g, nTurns, p180);
  let dielectric = cylinder(
    startRadius + size + 2.0 * (size + gap) * nTurns,
    size,
    p.add(vec3(0, 0, 0.5 * size))
  );
  dielectric = dielectric.sub(c1).sub(c2); // Don't put dielectric where the metal is.
  return [c1, c2, dielectric];
};

const mainModel4 = (xyz) => bifilarCoilWithDielectric(3.0, 0.85, 0.15, 9.0, xyz);
```

* Try loading [bifilar-coil-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/011-bifilar-coil/bifilar-coil-2.irmf) now in the IRMF editor!
* Try loading [bifilar-coil-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/011-bifilar-coil/bifilar-coil-2-wgsl.irmf) now in the IRMF editor!
* Try loading [bifilar-coil-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/011-bifilar-coil/bifilar-coil-2-tsl.irmf) now in the IRMF editor!

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
