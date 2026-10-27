# 014-chain-link

## chain-link-1.irmf

Once we have the torus with start and end parameters, we can make a chain
link out of a couple partial toroids and a couple cylinders:

![chain-link-1.png](chain-link-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [21,15,3],
  min: [-15,-15,-3],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float torus(in float majorRadius, in float minorRadius, in float fromDeg, in float toDeg, in vec3 xyz) {
  float r = length(xyz);
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  float angleDeg = mod(360.0 + atan(xyz.y, xyz.x) * 180.0 / M_PI, 360.0);
  if (fromDeg < toDeg &&(angleDeg < fromDeg || angleDeg > toDeg)) { return 0.0; }
  if (fromDeg > toDeg && angleDeg < fromDeg && angleDeg > toDeg) { return 0.0; }
  
  return 1.0;
}

float cylinder(in float radius, in float height, in vec3 xyz) {
  // First, trivial reject on the two ends of the cylinder.
  if (xyz.x < 0.0 || xyz.x > height) { return 0.0; }
  
  // Then, constrain radius of the cylinder:
  float r = length(xyz.yz);
  if (r > radius) { return 0.0; }
  
  return 1.0;
}

float chainLink(in float majorRadius, in float minorRadius, in float separator, in vec3 xyz) {
  float result = torus(majorRadius, minorRadius, 90.0, 270.0, xyz);
  result += torus(majorRadius, minorRadius, 270.0, 90.0, xyz-vec3(separator, 0, 0));
  result += cylinder(minorRadius, separator, xyz - vec3(0, majorRadius, 0));
  result += cylinder(minorRadius, separator, xyz + vec3(0, majorRadius, 0));
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials[0] = chainLink(9.0, 3.0, 6.0, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [21,15,3],
  min: [-15,-15,-3],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn torus(majorRadius: f32, minorRadius: f32, fromDeg: f32, toDeg: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  var angleDeg = (360.0 + atan2(xyz.y, xyz.x) * 180.0 / M_PI) % 360.0;
  if (fromDeg < toDeg && (angleDeg < fromDeg || angleDeg > toDeg)) { return 0.0; }
  if (fromDeg > toDeg && angleDeg < fromDeg && angleDeg > toDeg) { return 0.0; }
  
  return 1.0;
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  // First, trivial reject on the two ends of the cylinder.
  if (xyz.x < 0.0 || xyz.x > height) { return 0.0; }
  
  // Then, constrain radius of the cylinder:
  let r = length(xyz.yz);
  if (r > radius) { return 0.0; }
  
  return 1.0;
}

fn chainLink(majorRadius: f32, minorRadius: f32, separator: f32, xyz: vec3f) -> f32 {
  var result = torus(majorRadius, minorRadius, 90.0, 270.0, xyz);
  result += torus(majorRadius, minorRadius, 270.0, 90.0, xyz - vec3f(separator, 0.0, 0.0));
  result += cylinder(minorRadius, separator, xyz - vec3f(0.0, majorRadius, 0.0));
  result += cylinder(minorRadius, separator, xyz + vec3f(0.0, majorRadius, 0.0));
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = chainLink(9.0, 3.0, 6.0, xyz);
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [21,15,3],
  min: [-15,-15,-3],
  units: "mm",
}*/

const torus = (majorRadius, minorRadius, fromDeg, toDeg, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const angleDeg = mod(
    float(360.0).add(atan(p.y, p.x).mul(180.0).div(Math.PI)),
    float(360.0)
  );
  const angleOK = fromDeg < toDeg
    ? angleDeg.greaterThanEqual(fromDeg).and(angleDeg.lessThanEqual(toDeg))
    : angleDeg.greaterThanEqual(fromDeg).or(angleDeg.lessThanEqual(toDeg));
  return select(ringOK.and(r2.lessThanEqual(minorRadius)).and(angleOK), float(1.0), float(0.0));
};

const cylinderX = (radius, height, p) => {
  const xOK = p.x.greaterThanEqual(0.0).and(p.x.lessThanEqual(height));
  return select(xOK.and(length(p.yz).lessThanEqual(radius)), float(1.0), float(0.0));
};

const chainLink = (majorRadius, minorRadius, separator, p) => {
  return torus(majorRadius, minorRadius, 90.0, 270.0, p)
    .add(torus(majorRadius, minorRadius, 270.0, 90.0, p.sub(vec3(separator, 0, 0))))
    .add(cylinderX(minorRadius, separator, p.sub(vec3(0, majorRadius, 0))))
    .add(cylinderX(minorRadius, separator, p.add(vec3(0, majorRadius, 0))));
};

const mainModel4 = (xyz) => [chainLink(9.0, 3.0, 6.0, xyz)];
```

* Try loading [chain-link-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/014-chain-link/chain-link-1.irmf) now in the IRMF editor!
* Try loading [chain-link-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/014-chain-link/chain-link-1-wgsl.irmf) now in the IRMF editor!
* Try loading [chain-link-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/014-chain-link/chain-link-1-tsl.irmf) now in the IRMF editor!

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
