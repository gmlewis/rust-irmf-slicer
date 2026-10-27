# 013-torus

## torus-1.irmf

It turns out that a torus is also relatively easy to model:

![torus-1.png](torus-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [15,15,3],
  min: [-15,-15,-3],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float torus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  const float radius = 5.0;
  materials[0] = torus(9.0, 3.0, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [15,15,3],
  min: [-15,-15,-3],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn torus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = torus(9.0, 3.0, xyz);
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
  max: [15,15,3],
  min: [-15,-15,-3],
  units: "mm",
}*/

const torus = (majorRadius, minorRadius, xyz) => {
  const r = length(xyz);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(xyz.y, xyz.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(xyz.sub(center));
  return select(ringOK.and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => [torus(9.0, 3.0, xyz)];
```

* Try loading [torus-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/013-torus/torus-1.irmf) now in the IRMF editor!
* Try loading [torus-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/013-torus/torus-1-wgsl.irmf) now in the IRMF editor!
* Try loading [torus-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/013-torus/torus-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## torus-2.irmf

Sometimes you just want a slice of a torus, and I find it easier to think
in terms of degrees when sectioning things with a 'from' and 'to', so I
made the parameters degrees in this case.

![torus-2.png](torus-2.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1","PLA2"],
  max: [17,15,3],
  min: [-15,-15,-3],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float torus(float majorRadius, float minorRadius, float fromDeg, float toDeg, in vec3 xyz) {
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

void mainModel4(out vec4 materials, in vec3 xyz) {
  const float radius = 5.0;
  materials[0] = torus(9.0, 3.0, 45.0, 315.0, xyz);
  materials[1] = torus(9.0, 3.0, 315.0, 45.0, xyz - vec3(2,0,0));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1","PLA2"],
  max: [17,15,3],
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

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = torus(9.0, 3.0, 45.0, 315.0, xyz);
  materials[1] = torus(9.0, 3.0, 315.0, 45.0, xyz - vec3f(2,0,0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1","PLA2"],
  max: [17,15,3],
  min: [-15,-15,-3],
  units: "mm",
}*/

const torus = (majorRadius, minorRadius, fromDeg, toDeg, xyz) => {
  const r = length(xyz);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(xyz.y, xyz.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(xyz.sub(center));
  const angleDeg = mod(
    float(360.0).add(atan(xyz.y, xyz.x).mul(180.0).div(Math.PI)),
    float(360.0)
  );
  const angleOK = fromDeg < toDeg
    ? angleDeg.greaterThanEqual(fromDeg).and(angleDeg.lessThanEqual(toDeg))
    : angleDeg.greaterThanEqual(fromDeg).or(angleDeg.lessThanEqual(toDeg));
  return select(
    ringOK.and(r2.lessThanEqual(minorRadius)).and(angleOK),
    float(1.0),
    float(0.0)
  );
};

const mainModel4 = (xyz) => [
  torus(9.0, 3.0, 45.0, 315.0, xyz),
  torus(9.0, 3.0, 315.0, 45.0, xyz.sub(vec3(2, 0, 0))),
];
```

* Try loading [torus-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/013-torus/torus-2.irmf) now in the IRMF editor!
* Try loading [torus-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/013-torus/torus-2-wgsl.irmf) now in the IRMF editor!
* Try loading [torus-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/013-torus/torus-2-tsl.irmf) now in the IRMF editor!

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
