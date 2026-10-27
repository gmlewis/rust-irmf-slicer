# 027-libfive

I recently came across this excellent CSG CAD system called
[libfive](https://libfive.com)
by Matthew Keeter and was amazed how well his ideas mesh (:smile:)
with IRMF.

Here is an adaptation of [libfive examples](https://libfive.com/examples)
in IRMF.

## libfive-1.irmf - Scheme (high-level)

This [libfive example](https://libfive.com/examples/#stdlib):

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [1,1,1],
  min: [-1,-1,-1],
  units: "mm",
}*/

float sphere(in float radius, in vec3 center, in vec3 xyz) {
  float r = length(xyz-center);
  if (r > radius) { return 0.0; }
  return 1.0;
}

float cylinder_z(in float radius, in float height, in vec3 center, in vec3 xyz) {
  float h = length(xyz.z - center.z);
  if (h > height) { return 0.0; }
  float r = length(xyz.xy - center.xy);
  if (r > radius) { return 0.0; }
  return 1.0;
}

void mainModel4(out vec4 m, in vec3 xyz) {
  m[0] = sphere(1.0, vec3(0), xyz);
  m[0] -= sphere(0.6, vec3(0), xyz);
  m[0] -= cylinder_z(0.6, 2.0, vec3(0,0,-1), xyz);
  m[0] -= cylinder_z(0.6, 2.0, vec3(0,0,-1), xyz.zyx);
  m[0] -= cylinder_z(0.6, 2.0, vec3(0,0,-1), xyz.xzy);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [1,1,1],
  min: [-1,-1,-1],
  units: "mm",
}*/

fn sphere(radius: f32, center: vec3f, xyz: vec3f) -> f32 {
  let r = length(xyz-center);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn cylinder_z(radius: f32, height: f32, center: vec3f, xyz: vec3f) -> f32 {
  let h = abs(xyz.z - center.z);
  if (h > height) { return 0.0; }
  let r = length(xyz.xy - center.xy);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var m = vec4f(0.0);
  m[0] = sphere(1.0, vec3f(0.0), xyz);
  m[0] -= sphere(0.6, vec3f(0.0), xyz);
  m[0] -= cylinder_z(0.6, 2.0, vec3f(0.0, 0.0, -1.0), xyz);
  m[0] -= cylinder_z(0.6, 2.0, vec3f(0.0, 0.0, -1.0), xyz.zyx);
  m[0] -= cylinder_z(0.6, 2.0, vec3f(0.0, 0.0, -1.0), xyz.xzy);
  return m;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [1,1,1],
  min: [-1,-1,-1],
  units: "mm",
}*/

const sphere = (radius, center, xyz) => {
  return select(length(xyz.sub(center)).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinder_z = (radius, height, center, xyz) => {
  const h = xyz.z.sub(center.z).abs();
  const r = length(xyz.xy.sub(center.xy));
  return select(h.lessThanEqual(height).and(r.lessThanEqual(radius)), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => {
  const zero = vec3(0);
  const c = vec3(0, 0, -1);
  let m = sphere(1.0, zero, xyz);
  m = m.sub(sphere(0.6, zero, xyz));
  m = m.sub(cylinder_z(0.6, 2.0, c, xyz));
  m = m.sub(cylinder_z(0.6, 2.0, c, xyz.zyx));
  m = m.sub(cylinder_z(0.6, 2.0, c, xyz.xzy));
  return [m];
};
```

* Try loading [libfive-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-1.irmf) now in the IRMF editor!
* Try loading [libfive-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-1-wgsl.irmf) now in the IRMF editor!
* Try loading [libfive-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## libfive-2.irmf - Scheme (low-level)

The second [libfive example](https://libfive.com/examples/#scheme):

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [3,3,3],
  min: [-3,-3,-3],
  units: "mm",
}*/

float cube(in vec3 xyz) {
  if (any(greaterThan(abs(xyz), vec3(1)))) { return 0.0; }
  return 1.0;
}

void mainModel4(out vec4 m, in vec3 xyz) {
  float cz = cos(xyz.z);
  float sz = sin(xyz.z);
  vec3 twist = vec3(cz*xyz.x + sz*xyz.y, cz*xyz.y-sz*xyz.x, xyz.z);
  m[0] = cube(twist);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [3,3,3],
  min: [-3,-3,-3],
  units: "mm",
}*/

fn cube(xyz: vec3f) -> f32 {
  if (any(abs(xyz) > vec3f(1.0))) { return 0.0; }
  return 1.0;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  let cz = cos(xyz.z);
  let sz = sin(xyz.z);
  let twist = vec3f(cz*xyz.x + sz*xyz.y, cz*xyz.y-sz*xyz.x, xyz.z);
  var m = vec4f(0.0);
  m[0] = cube(twist);
  return m;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [3,3,3],
  min: [-3,-3,-3],
  units: "mm",
}*/

const cube = (xyz) => {
  return select(any(xyz.abs().greaterThan(1.0)), float(0.0), float(1.0));
};

const mainModel4 = (xyz) => {
  const cz = cos(xyz.z);
  const sz = sin(xyz.z);
  const twist = vec3(
    cz.mul(xyz.x).add(sz.mul(xyz.y)),
    cz.mul(xyz.y).sub(sz.mul(xyz.x)),
    xyz.z
  );
  return [cube(twist)];
};
```

* Try loading [libfive-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-2.irmf) now in the IRMF editor!
* Try loading [libfive-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-2-wgsl.irmf) now in the IRMF editor!
* Try loading [libfive-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-2-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## libfive-3.irmf - CSG blend

The third [libfive example](https://libfive.com/examples/#scheme):

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [2,2,2],
  min: [-2,-2,-2],
  units: "mm",
}*/

float sphere(in float radius, in vec3 center, in vec3 xyz) {
  return length(center-xyz) - radius;
}

float blend(in float a, in float b, in float m) {
  float v = sqrt(abs(a)) + sqrt(abs(b)) - m;
  return min(a, min(b, v));
}

void mainModel4(out vec4 m, in vec3 xyz) {
  float v =
    blend(
      blend(
        sphere(1.0, vec3(0,-1,-1), xyz),
        sphere(1.0, vec3(0, 1,-1), xyz), 0.75),
      blend(
        sphere(1.0, vec3(0,-1, 1), xyz),
        sphere(1.0, vec3(0, 1, 1), xyz), 0.75),
      0.75);
  m[0] = v > 0.01 ? 0.0 : 1.0;
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [2,2,2],
  min: [-2,-2,-2],
  units: "mm",
}*/

fn sphere(radius: f32, center: vec3f, xyz: vec3f) -> f32 {
  return length(center-xyz) - radius;
}

fn blend(a: f32, b: f32, m: f32) -> f32 {
  let v = sqrt(abs(a)) + sqrt(abs(b)) - m;
  return min(a, min(b, v));
}

fn mainModel4(xyz: vec3f) -> vec4f {
  let v =
    blend(
      blend(
        sphere(1.0, vec3f(0.0, -1.0, -1.0), xyz),
        sphere(1.0, vec3f(0.0, 1.0, -1.0), xyz), 0.75),
      blend(
        sphere(1.0, vec3f(0.0, -1.0, 1.0), xyz),
        sphere(1.0, vec3f(0.0, 1.0, 1.0), xyz), 0.75),
      0.75);
  var m = vec4f(0.0);
  m[0] = select(1.0, 0.0, v > 0.01);
  return m;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [2,2,2],
  min: [-2,-2,-2],
  units: "mm",
}*/

const sphere = (radius, center, xyz) => {
  return length(center.sub(xyz)).sub(radius);
};

const blend = (a, b, m) => {
  const v = sqrt(a.abs()).add(sqrt(b.abs())).sub(m);
  return min(a, min(b, v));
};

const mainModel4 = (xyz) => {
  const v = blend(
    blend(
      sphere(1.0, vec3(0, -1, -1), xyz),
      sphere(1.0, vec3(0, 1, -1), xyz),
      0.75
    ),
    blend(
      sphere(1.0, vec3(0, -1, 1), xyz),
      sphere(1.0, vec3(0, 1, 1), xyz),
      0.75
    ),
    0.75
  );
  return [select(v.greaterThan(0.01), float(0.0), float(1.0))];
};
```

* Try loading [libfive-3.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-3.irmf) now in the IRMF editor!
* Try loading [libfive-3-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-3-wgsl.irmf) now in the IRMF editor!
* Try loading [libfive-3-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/027-libfive/libfive-3-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

----------------------------------------------------------------------

# License

Copyright 2022 Glenn M. Lewis. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
