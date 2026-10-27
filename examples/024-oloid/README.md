# 024-oloid

## oloid-1.irmf

This is an oloid solid as defined on [Wikipedia](https://en.wikipedia.org/wiki/Oloid):

![oloid-1.png](oloid-1.png)

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

#define M_PI 3.1415926535897932384626433832795

float oloid(in vec3 xyz) {
  if (xyz.x >= 0.0) {
    float r = length(xyz.xy - vec2(0.5, 0));
    if (r > 1.0) { return 0.0; }
    float a = abs(atan(xyz.y, xyz.x - 0.5));
    float y = abs(xyz.y);
    float h = sin(120.0 * M_PI / 180.0) * (1.0 - xyz.x / 1.5);
    float sy = sqrt((1.0 - (xyz.x - 0.5) * (xyz.x - 0.5)));
    float sz = mix(h, 0.0, y / sy);
    float z = abs(xyz.z);
    if (z > sz) { return 0.0; }
  } else {
    float r = length(xyz.xz + vec2(0.5, 0));
    if (r > 1.0) { return 0.0; }
    float a = abs(atan(xyz.z, xyz.x + 0.5));
    float z = abs(xyz.z);
    float h = sin(120.0 * M_PI / 180.0) * (1.0 + xyz.x / 1.5);
    float sz = sqrt((1.0 - (xyz.x + 0.5) * (xyz.x + 0.5)));
    float sy = mix(h, 0.0, z / sz);
    float y = abs(xyz.y);
    if (y > sy) { return 0.0; }
  }
  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials[0] = oloid(xyz / (5.0 / 1.5));
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

const M_PI = 3.1415926535897932384626433832795;

fn oloid(xyz: vec3f) -> f32 {
  if (xyz.x >= 0.0) {
    let r = length(xyz.xy - vec2f(0.5, 0.0));
    if (r > 1.0) { return 0.0; }
    let y = abs(xyz.y);
    let h = sin(120.0 * M_PI / 180.0) * (1.0 - xyz.x / 1.5);
    let sy = sqrt((1.0 - (xyz.x - 0.5) * (xyz.x - 0.5)));
    let sz = mix(h, 0.0, y / sy);
    let z = abs(xyz.z);
    if (z > sz) { return 0.0; }
  } else {
    let r = length(xyz.xz + vec2f(0.5, 0.0));
    if (r > 1.0) { return 0.0; }
    let z = abs(xyz.z);
    let h = sin(120.0 * M_PI / 180.0) * (1.0 + xyz.x / 1.5);
    let sz = sqrt((1.0 - (xyz.x + 0.5) * (xyz.x + 0.5)));
    let sy = mix(h, 0.0, z / sz);
    let y = abs(xyz.y);
    if (y > sy) { return 0.0; }
  }
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = oloid(xyz_in / (5.0 / 1.5));
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

const oloid = (p) => {
  const right = p.x.greaterThanEqual(0.0);

  const hConst = float(Math.sin((120.0 * Math.PI) / 180.0));

  // Right half.
  const rR = length(p.xy.sub(vec2(0.5, 0)));
  const y = p.y.abs();
  const hR = hConst.mul(float(1.0).sub(p.x.div(1.5)));
  const syR = sqrt(float(1.0).sub(p.x.sub(0.5).mul(p.x.sub(0.5))));
  const szR = mix(hR, float(0.0), y.div(syR));
  const z = p.z.abs();
  const okR = rR.lessThanEqual(1.0).and(z.lessThanEqual(szR));

  // Left half.
  const rL = length(p.xz.add(vec2(0.5, 0)));
  const hL = hConst.mul(float(1.0).add(p.x.div(1.5)));
  const szL = sqrt(float(1.0).sub(p.x.add(0.5).mul(p.x.add(0.5))));
  const syL = mix(hL, float(0.0), z.div(szL));
  const okL = rL.lessThanEqual(1.0).and(p.y.abs().lessThanEqual(syL));

  return select(right, select(okR, float(1.0), float(0.0)), select(okL, float(1.0), float(0.0)));
};

const mainModel4 = (xyz) => [oloid(xyz.div(5.0 / 1.5))];
```

* Try loading [oloid-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/024-oloid/oloid-1.irmf) now in the IRMF editor!
* Try loading [oloid-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/024-oloid/oloid-1-wgsl.irmf) now in the IRMF editor!
* Try loading [oloid-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/024-oloid/oloid-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## oloid-2.irmf

Just like [sphericon](https://github.com/gmlewis/irmf/tree/master/examples/022-superquadrics#sphericon-2irmf),
oloid is easier to visualize with two materials.

![oloid-2.png](oloid-2.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["Red","Green"],
  max: [5,5,5],
  min: [-5,-5,-5],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

vec2 oloid2(in float slices, in vec3 xyz) {
  if (xyz.x >= 0.0) {
    float r = length(xyz.xy - vec2(0.5, 0));
    if (r > 1.0) { return vec2(0); }
    float a = abs(atan(xyz.y, xyz.x - 0.5));
    float y = abs(xyz.y);
    float h = sin(120.0 * M_PI / 180.0) * (1.0 - xyz.x / 1.5);
    float sy = sqrt((1.0 - (xyz.x - 0.5) * (xyz.x - 0.5)));
    float sz = mix(h, 0.0, y / sy);
    float z = abs(xyz.z);
    if (z > sz) { return vec2(0); }
    if (mod(abs(xyz.z) * slices, 1.0) <= 0.5) { return vec2(1, 0); }
    return vec2(0, 1);
  } else {
    float r = length(xyz.xz + vec2(0.5, 0));
    if (r > 1.0) { return vec2(0); }
    float a = abs(atan(xyz.z, xyz.x + 0.5));
    float z = abs(xyz.z);
    float h = sin(120.0 * M_PI / 180.0) * (1.0 + xyz.x / 1.5);
    float sz = sqrt((1.0 - (xyz.x + 0.5) * (xyz.x + 0.5)));
    float sy = mix(h, 0.0, z / sz);
    float y = abs(xyz.y);
    if (y > sy) { return vec2(0); }
    if (mod(-abs(xyz.y) * slices, 1.0) <= 0.5) { return vec2(1, 0); }
    return vec2(0, 1);
  }
  return vec2(0);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials.xy = oloid2(5.79, xyz / (5.0 / 1.5));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["Red","Green"],
  max: [5,5,5],
  min: [-5,-5,-5],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn oloid2(slices: f32, xyz: vec3f) -> vec2f {
  if (xyz.x >= 0.0) {
    let r = length(xyz.xy - vec2f(0.5, 0.0));
    if (r > 1.0) { return vec2f(0.0); }
    let y = abs(xyz.y);
    let h = sin(120.0 * M_PI / 180.0) * (1.0 - xyz.x / 1.5);
    let sy = sqrt((1.0 - (xyz.x - 0.5) * (xyz.x - 0.5)));
    let sz = mix(h, 0.0, y / sy);
    let z = abs(xyz.z);
    if (z > sz) { return vec2f(0.0); }
    if (wgsl_mod(abs(xyz.z) * slices, 1.0) <= 0.5) { return vec2f(1.0, 0.0); }
    return vec2f(0.0, 1.0);
  } else {
    let r = length(xyz.xz + vec2f(0.5, 0.0));
    if (r > 1.0) { return vec2f(0.0); }
    let z = abs(xyz.z);
    let h = sin(120.0 * M_PI / 180.0) * (1.0 + xyz.x / 1.5);
    let sz = sqrt((1.0 - (xyz.x + 0.5) * (xyz.x + 0.5)));
    let sy = mix(h, 0.0, z / sz);
    let y = abs(xyz.y);
    if (y > sy) { return vec2f(0.0); }
    if (wgsl_mod(-abs(xyz.y) * slices, 1.0) <= 0.5) { return vec2f(1.0, 0.0); }
    return vec2f(0.0, 1.0);
  }
  return vec2f(0.0);
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  let res = oloid2(5.79, xyz_in / (5.0 / 1.5));
  return vec4f(res, 0.0, 0.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["Red","Green"],
  max: [5,5,5],
  min: [-5,-5,-5],
  units: "mm",
}*/

const oloid2 = (slices, p) => {
  const right = p.x.greaterThanEqual(0.0);
  const hConst = float(Math.sin((120.0 * Math.PI) / 180.0));

  const z = p.z.abs();
  const y = p.y.abs();

  const rR = length(p.xy.sub(vec2(0.5, 0)));
  const hR = hConst.mul(float(1.0).sub(p.x.div(1.5)));
  const syR = sqrt(float(1.0).sub(p.x.sub(0.5).mul(p.x.sub(0.5))));
  const szR = mix(hR, float(0.0), y.div(syR));
  const okR = rR.lessThanEqual(1.0).and(z.lessThanEqual(szR));
  const pickR = mod(z.mul(slices), 1.0).lessThanEqual(0.5);

  const rL = length(p.xz.add(vec2(0.5, 0)));
  const hL = hConst.mul(float(1.0).add(p.x.div(1.5)));
  const szL = sqrt(float(1.0).sub(p.x.add(0.5).mul(p.x.add(0.5))));
  const syL = mix(hL, float(0.0), z.div(szL));
  const okL = rL.lessThanEqual(1.0).and(y.lessThanEqual(syL));
  const pickL = mod(y.mul(slices).negate(), 1.0).lessThanEqual(0.5);

  return [
    select(right, select(okR, select(pickR, float(1.0), float(0.0)), float(0.0)),
      select(okL, select(pickL, float(1.0), float(0.0)), float(0.0))),
    select(right, select(okR, select(pickR, float(0.0), float(1.0)), float(0.0)),
      select(okL, select(pickL, float(0.0), float(1.0)), float(0.0))),
  ];
};

const mainModel4 = (xyz) => oloid2(5.79, xyz.div(5.0 / 1.5));
```

* Try loading [oloid-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/024-oloid/oloid-2.irmf) now in the IRMF editor!
* Try loading [oloid-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/024-oloid/oloid-2-wgsl.irmf) now in the IRMF editor!
* Try loading [oloid-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/024-oloid/oloid-2-tsl.irmf) now in the IRMF editor!

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
