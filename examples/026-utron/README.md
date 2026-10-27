# 026-utron

Otis T. Carr invented the utron. Here is a simple model of half of a utron.

## half-utron-1.irmf

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper"],
  max: [36,36,46],
  min: [-36,-36,0],
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

float sphere(in float radius, in vec3 xyz) {
  float r = length(xyz);
  return r <= radius ? 1.0 : 0.0;
}

float cylinder(float radius, float height, in vec3 xyz) {
  // First, trivial reject on the two ends of the cylinder.
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  
  // Then, constrain radius of the cylinder:
  float rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  
  return 1.0;
}

float halfUtron(in float sphereDiam, in float edgeLen, in float shaftDiam, in float shaftHeight, in vec3 xyz) {
    float h = sqrt(edgeLen*edgeLen/2.0);
    float r = sphereDiam/2.0;
    float shaftR = shaftDiam/2.0;
    vec3 shaftPos = vec3(0,0,h-shaftR);
    return cone(xyz/h) + cylinder(shaftR, shaftHeight, xyz-shaftPos) - sphere(1.0, xyz/r);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials[0] = halfUtron(44.0, 50.0, 4.98, 10.0, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper"],
  max: [36,36,46],
  min: [-36,-36,0],
  units: "mm",
}*/

fn cone(xyz: vec3f) -> f32 {
  if (xyz.z < 0.0 || xyz.z > 1.0) { return 0.0; }
  let zsize = mix(1.0, 0.0, xyz.z);
  let r = length(xyz.xy);
  if (r > zsize) { return 0.0; }
  return 1.0;
}

fn sphere(radius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  return select(0.0, 1.0, r <= radius);
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  return 1.0;
}

fn halfUtron(sphereDiam: f32, edgeLen: f32, shaftDiam: f32, shaftHeight: f32, xyz: vec3f) -> f32 {
    let h = sqrt(edgeLen*edgeLen/2.0);
    let r = sphereDiam/2.0;
    let shaftR = shaftDiam/2.0;
    let shaftPos = vec3f(0.0, 0.0, h-shaftR);
    return cone(xyz/h) + cylinder(shaftR, shaftHeight, xyz-shaftPos) - sphere(1.0, xyz/r);
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = halfUtron(44.0, 50.0, 4.98, 10.0, xyz);
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper"],
  max: [36,36,46],
  min: [-36,-36,0],
  units: "mm",
}*/

const cone = (xyz) => {
  const zOK = xyz.z.greaterThanEqual(0.0).and(xyz.z.lessThanEqual(1.0));
  const zsize = mix(float(1.0), float(0.0), xyz.z);
  return select(zOK.and(length(xyz.xy).lessThanEqual(zsize)), float(1.0), float(0.0));
};

const sphere = (radius, xyz) => {
  return select(length(xyz).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinder = (radius, height, xyz) => {
  const zOK = xyz.z.greaterThanEqual(0.0).and(xyz.z.lessThanEqual(height));
  return select(zOK.and(length(xyz.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const halfUtron = (sphereDiam, edgeLen, shaftDiam, shaftHeight, xyz) => {
  const h = Math.sqrt((edgeLen * edgeLen) / 2.0);
  const r = sphereDiam / 2.0;
  const shaftR = shaftDiam / 2.0;
  const shaftPos = vec3(0, 0, h - shaftR);
  return cone(xyz.div(h))
    .add(cylinder(shaftR, shaftHeight, xyz.sub(shaftPos)))
    .sub(sphere(1.0, xyz.div(r)));
};

const mainModel4 = (xyz) => [halfUtron(44.0, 50.0, 4.98, 10.0, xyz)];
```

* Try loading [half-utron-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/026-utron/half-utron-1.irmf) now in the IRMF editor!
* Try loading [half-utron-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/026-utron/half-utron-1-wgsl.irmf) now in the IRMF editor!
* Try loading [half-utron-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/026-utron/half-utron-1-tsl.irmf) now in the IRMF editor!

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
