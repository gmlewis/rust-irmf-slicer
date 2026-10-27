# 022-superquadrics

One of my favorite computer graphics classes I took at Caltech
in 1988 was taught by the teaching assistant,
[John Snyder](https://www.microsoft.com/en-us/research/people/johnsny/)
for the professor, [Al Barr](http://www.gg.caltech.edu/~barr/index.html).

In that class, we wrote our own renderer and one of the primitives
was [superquadrics](https://authors.library.caltech.edu/9756/).

Below are example superquadrics that are beautifully suited for
use in IRMF.

## superquad-ellipsoids-1.irmf

This is a replication of Figure 7 in Al Barr's article linked above.

![superquad-ellipsoids-1.png](superquad-ellipsoids-1.png)

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

float superquad(in float e1, in float e2, in vec3 xyz) {
  xyz = abs(xyz); // Due to GLSL 'pow' definition.
  float f = pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / e1) + pow(xyz.z, 2.0 / e1);
  return f <= 1.0 ? 1.0 : 0.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  float u = 1.2;
  float v = 3.4;
  materials[0] =
  superquad(0.3, 0.3, xyz - vec3(-v, 0, v))
  + superquad(0.3, 0.1, xyz - vec3(-u, 0, v))
  + superquad(0.3, 1.0, xyz - vec3(u, 0, v))
  + superquad(0.3, 3.0, xyz - vec3(v, 0, v))
  + superquad(0.1, 0.3, xyz - vec3(-v, 0, u))
  + superquad(0.1, 0.1, xyz - vec3(-u, 0, u))
  + superquad(0.1, 1.0, xyz - vec3(u, 0, u))
  + superquad(0.1, 3.0, xyz - vec3(v, 0, u))
  + superquad(1.0, 0.3, xyz - vec3(-v, 0, - u))
  + superquad(1.0, 0.1, xyz - vec3(-u, 0, - u))
  + superquad(1.0, 1.0, xyz - vec3(u, 0, - u))
  + superquad(1.0, 3.0, xyz - vec3(v, 0, - u))
  + superquad(3.0, 0.3, xyz - vec3(-v, 0, - v))
  + superquad(3.0, 0.1, xyz - vec3(-u, 0, - v))
  + superquad(3.0, 1.0, xyz - vec3(u, 0, - v))
  + superquad(3.0, 3.0, xyz - vec3(v, 0, - v));
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

fn superquad(e1: f32, e2: f32, xyz_in: vec3f) -> f32 {
  let xyz = abs(xyz_in);
  let f = pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / e1) + pow(xyz.z, 2.0 / e1);
  return select(0.0, 1.0, f <= 1.0);
}

fn mainModel4(xyz: vec3f) -> vec4f {
  let u = 1.2;
  let v = 3.4;
  var materials = vec4f(0.0);
  materials[0] =
  superquad(0.3, 0.3, xyz - vec3f(-v, 0.0, v))
  + superquad(0.3, 0.1, xyz - vec3f(-u, 0.0, v))
  + superquad(0.3, 1.0, xyz - vec3f(u, 0.0, v))
  + superquad(0.3, 3.0, xyz - vec3f(v, 0.0, v))
  + superquad(0.1, 0.3, xyz - vec3f(-v, 0.0, u))
  + superquad(0.1, 0.1, xyz - vec3f(-u, 0.0, u))
  + superquad(0.1, 1.0, xyz - vec3f(u, 0.0, u))
  + superquad(0.1, 3.0, xyz - vec3f(v, 0.0, u))
  + superquad(1.0, 0.3, xyz - vec3f(-v, 0.0, - u))
  + superquad(1.0, 0.1, xyz - vec3f(-u, 0.0, - u))
  + superquad(1.0, 1.0, xyz - vec3f(u, 0.0, - u))
  + superquad(1.0, 3.0, xyz - vec3f(v, 0.0, - u))
  + superquad(3.0, 0.3, xyz - vec3f(-v, 0.0, - v))
  + superquad(3.0, 0.1, xyz - vec3f(-u, 0.0, - v))
  + superquad(3.0, 1.0, xyz - vec3f(u, 0.0, - v))
  + superquad(3.0, 3.0, xyz - vec3f(v, 0.0, - v));
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

const superquad = (e1, e2, xyz) => {
  const p = xyz.abs(); // Due to GLSL 'pow' definition.
  const f = pow(
    pow(p.x, 2.0 / e2).add(pow(p.y, 2.0 / e2)),
    e2 / e1
  ).add(pow(p.z, 2.0 / e1));
  return select(f.lessThanEqual(1.0), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => {
  const u = 1.2;
  const v = 3.4;
  const entries = [
    [0.3, 0.3, -v, v], [0.3, 0.1, -u, v], [0.3, 1.0, u, v], [0.3, 3.0, v, v],
    [0.1, 0.3, -v, u], [0.1, 0.1, -u, u], [0.1, 1.0, u, u], [0.1, 3.0, v, u],
    [1.0, 0.3, -v, -u], [1.0, 0.1, -u, -u], [1.0, 1.0, u, -u], [1.0, 3.0, v, -u],
    [3.0, 0.3, -v, -v], [3.0, 0.1, -u, -v], [3.0, 1.0, u, -v], [3.0, 3.0, v, -v],
  ];
  let sum = float(0.0);
  for (const [e1, e2, cx, cz] of entries) {
    sum = sum.add(superquad(e1, e2, xyz.sub(vec3(cx, 0, cz))));
  }
  return [sum];
};
```

* Try loading [superquad-ellipsoids-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-ellipsoids-1.irmf) now in the IRMF editor!
* Try loading [superquad-ellipsoids-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-ellipsoids-1-wgsl.irmf) now in the IRMF editor!
* Try loading [superquad-ellipsoids-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-ellipsoids-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## superquad-ellipsoids-2.irmf

This version is sliced into two materials to make it easier to visualize.

![superquad-ellipsoids-2.png](superquad-ellipsoids-2.png)

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

vec2 superquad2(in float slices, in float e1, in float e2, in vec3 xyz) {
  xyz = abs(xyz); // Due to GLSL 'pow' definition.
  float f = pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / e1) + pow(xyz.z, 2.0 / e1);
  if (f > 1.0) { return vec2(0); }
  return (mod(abs(xyz.z) * slices, 1.0) <= 0.5) ? vec2(1, 0) : vec2(0, 1);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials.xy = vec2(0);
  
  vec4 e = vec4(0.1, 1.0, 2.0, 3.0);
  vec4 o = vec4(-3.4, - 1.2, 1.2, 3.4);
  for(int i = 0; i < 4; i ++ ) {
    for(int j = 0; j < 4; j ++ ) {
      materials.xy += superquad2(6.0, e[i], e[j], xyz - vec3(o[j], 0, o[3 - i]));
    }
  }
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

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn superquad2(slices: f32, e1: f32, e2: f32, xyz_in: vec3f) -> vec2f {
  let xyz = abs(xyz_in);
  let f = pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / e1) + pow(xyz.z, 2.0 / e1);
  if (f > 1.0) { return vec2f(0.0); }
  return select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), wgsl_mod(abs(xyz.z) * slices, 1.0) <= 0.5);
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var mat_xy = vec2f(0.0);
  let e = vec4f(0.1, 1.0, 2.0, 3.0);
  let o = vec4f(-3.4, - 1.2, 1.2, 3.4);
  for(var i = 0; i < 4; i ++ ) {
    for(var j = 0; j < 4; j ++ ) {
      mat_xy += superquad2(6.0, e[i], e[j], xyz - vec3f(o[j], 0.0, o[3 - i]));
    }
  }
  return vec4f(mat_xy, 0.0, 0.0);
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

const superquad2 = (slices, e1, e2, p) => {
  const q = p.abs(); // Due to GLSL 'pow' definition.
  const f = pow(pow(q.x, 2.0 / e2).add(pow(q.y, 2.0 / e2)), e2 / e1).add(pow(q.z, 2.0 / e1));
  const inside = f.lessThanEqual(1.0);
  const pick = mod(q.z.mul(slices), 1.0).lessThanEqual(0.5);
  return [
    select(inside, select(pick, float(1.0), float(0.0)), float(0.0)),
    select(inside, select(pick, float(0.0), float(1.0)), float(0.0)),
  ];
};

const mainModel4 = (xyz) => {
  const e = [0.1, 1.0, 2.0, 3.0];
  const o = [-3.4, -1.2, 1.2, 3.4];
  let m0 = float(0.0);
  let m1 = float(0.0);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const [a, b] = superquad2(6.0, e[i], e[j], xyz.sub(vec3(o[j], 0, o[3 - i])));
      m0 = m0.add(a);
      m1 = m1.add(b);
    }
  }
  return [m0, m1];
};
```

* Try loading [superquad-ellipsoids-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-ellipsoids-2.irmf) now in the IRMF editor!
* Try loading [superquad-ellipsoids-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-ellipsoids-2-wgsl.irmf) now in the IRMF editor!
* Try loading [superquad-ellipsoids-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-ellipsoids-2-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## superquad-toroids-1.irmf

This is a replication of Figure 10 in Al Barr's article linked above.

![superquad-toroids-1.png](superquad-toroids-1.png)

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

float superquad(in float e1, in float e2, in float a4, in vec3 xyz) {
  xyz.xyz *= 2.5;
  float angle = -1.3;
  float c = cos(angle);
  float s = sin(angle);
  xyz.yz = mat2(c, - s, s, c) * xyz.yz;
  xyz = abs(xyz); // Due to GLSL 'pow' definition.
  float f = pow(pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / 2.0) - a4, 2.0 / e1) + pow(xyz.z, 2.0 / e1);
  return f <= 1.0 ? 1.0 : 0.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  float u = 1.2;
  float v = 3.4;
  float a4 = 1.25;
  materials[0] =
  superquad(0.3, 0.3, a4, xyz - vec3(-v, 0, v))
  + superquad(0.3, 0.1, a4, xyz - vec3(-u, 0, v))
  + superquad(0.3, 1.0, a4, xyz - vec3(u, 0, v))
  + superquad(0.3, 3.0, a4, xyz - vec3(v, 0, v))
  + superquad(0.1, 0.3, a4, xyz - vec3(-v, 0, u))
  + superquad(0.1, 0.1, a4, xyz - vec3(-u, 0, u))
  + superquad(0.1, 1.0, a4, xyz - vec3(u, 0, u))
  + superquad(0.1, 3.0, a4, xyz - vec3(v, 0, u))
  + superquad(1.0, 0.3, a4, xyz - vec3(-v, 0, - u))
  + superquad(1.0, 0.1, a4, xyz - vec3(-u, 0, - u))
  + superquad(1.0, 1.0, a4, xyz - vec3(u, 0, - u))
  + superquad(1.0, 3.0, a4, xyz - vec3(v, 0, - u))
  + superquad(3.0, 0.3, a4, xyz - vec3(-v, 0, - v))
  + superquad(3.0, 0.1, a4, xyz - vec3(-u, 0, - v))
  + superquad(3.0, 1.0, a4, xyz - vec3(u, 0, - v))
  + superquad(3.0, 3.0, a4, xyz - vec3(v, 0, - v));
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

fn superquad(e1: f32, e2: f32, a4: f32, xyz_orig: vec3f) -> f32 {
  var xyz = xyz_orig;
  xyz *= 2.5;
  let angle = -1.3;
  let c = cos(angle);
  let s = sin(angle);
  let y = xyz.y * c - xyz.z * s;
  let z = xyz.y * s + xyz.z * c;
  xyz.y = y;
  xyz.z = z;
  xyz = abs(xyz);
  let f = pow(pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / 2.0) - a4, 2.0 / e1) + pow(xyz.z, 2.0 / e1);
  return select(0.0, 1.0, f <= 1.0);
}

fn mainModel4(xyz: vec3f) -> vec4f {
  let u = 1.2;
  let v = 3.4;
  let a4 = 1.25;
  var materials = vec4f(0.0);
  materials[0] =
  superquad(0.3, 0.3, a4, xyz - vec3f(-v, 0.0, v))
  + superquad(0.3, 0.1, a4, xyz - vec3f(-u, 0.0, v))
  + superquad(0.3, 1.0, a4, xyz - vec3f(u, 0.0, v))
  + superquad(0.3, 3.0, a4, xyz - vec3f(v, 0.0, v))
  + superquad(0.1, 0.3, a4, xyz - vec3f(-v, 0.0, u))
  + superquad(0.1, 0.1, a4, xyz - vec3f(-u, 0.0, u))
  + superquad(0.1, 1.0, a4, xyz - vec3f(u, 0.0, u))
  + superquad(0.1, 3.0, a4, xyz - vec3f(v, 0.0, u))
  + superquad(1.0, 0.3, a4, xyz - vec3f(-v, 0.0, - u))
  + superquad(1.0, 0.1, a4, xyz - vec3f(-u, 0.0, - u))
  + superquad(1.0, 1.0, a4, xyz - vec3f(u, 0.0, - u))
  + superquad(1.0, 3.0, a4, xyz - vec3f(v, 0.0, - u))
  + superquad(3.0, 0.3, a4, xyz - vec3f(-v, 0.0, - v))
  + superquad(3.0, 0.1, a4, xyz - vec3f(-u, 0.0, - v))
  + superquad(3.0, 1.0, a4, xyz - vec3f(u, 0.0, - v))
  + superquad(3.0, 3.0, a4, xyz - vec3f(v, 0.0, - v));
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

const superquad = (e1, e2, a4, p) => {
  let q = p.mul(2.5);
  const angle = -1.3;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  // GLSL: xyz.yz = mat2(c,-s,s,c) * xyz.yz
  const y2 = q.y.mul(c).add(q.z.mul(s));
  const z2 = q.y.mul(s).negate().add(q.z.mul(c));
  q = vec3(q.x, y2, z2).abs(); // Due to GLSL 'pow' definition.
  const f = pow(
    pow(pow(q.x, 2.0 / e2).add(pow(q.y, 2.0 / e2)), e2 / 2.0).sub(a4),
    2.0 / e1
  ).add(pow(q.z, 2.0 / e1));
  return select(f.lessThanEqual(1.0), float(1.0), float(0.0));
};

const mainModel4 = (xyz) => {
  const u = 1.2;
  const v = 3.4;
  const a4 = 1.25;
  const entries = [
    [0.3, 0.3, -v, v], [0.3, 0.1, -u, v], [0.3, 1.0, u, v], [0.3, 3.0, v, v],
    [0.1, 0.3, -v, u], [0.1, 0.1, -u, u], [0.1, 1.0, u, u], [0.1, 3.0, v, u],
    [1.0, 0.3, -v, -u], [1.0, 0.1, -u, -u], [1.0, 1.0, u, -u], [1.0, 3.0, v, -u],
    [3.0, 0.3, -v, -v], [3.0, 0.1, -u, -v], [3.0, 1.0, u, -v], [3.0, 3.0, v, -v],
  ];
  let sum = float(0.0);
  for (const [e1, e2, cx, cz] of entries) {
    sum = sum.add(superquad(e1, e2, a4, xyz.sub(vec3(cx, 0, cz))));
  }
  return [sum];
};
```

* Try loading [superquad-toroids-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-toroids-1.irmf) now in the IRMF editor!
* Try loading [superquad-toroids-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-toroids-1-wgsl.irmf) now in the IRMF editor!
* Try loading [superquad-toroids-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-toroids-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## superquad-toroids-2.irmf

This version is sliced into two materials to make it easier to visualize.

![superquad-toroids-2.png](superquad-toroids-2.png)

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

vec2 superquad2(in float slices, in float e1, in float e2, in float a4, in vec3 xyz) {
  xyz.xyz *= 2.5;
  vec2 result = (mod(abs(xyz.z) * 6.0, 1.0) <= 0.5) ? vec2(1, 0) : vec2(0, 1);
  float angle = -1.3;
  float c = cos(angle);
  float s = sin(angle);
  xyz.yz = mat2(c, - s, s, c) * xyz.yz;
  xyz = abs(xyz); // Due to GLSL 'pow' definition.
  float f = pow(pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / 2.0) - a4, 2.0 / e1) + pow(xyz.z, 2.0 / e1);
  return f <= 1.0 ? result : vec2(0);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials.xy = vec2(0);
  
  vec4 e = vec4(0.1, 0.3, 1.0, 3.0);
  vec4 o = vec4(-3.4, - 1.2, 1.2, 3.4);
  for(int i = 0; i < 4; i ++ ) {
    for(int j = 0; j < 4; j ++ ) {
      materials.xy += superquad2(6.0, e[i], e[j], 1.5, xyz - vec3(o[j], 0, o[3 - i]));
    }
  }
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

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn superquad2(slices: f32, e1: f32, e2: f32, a4: f32, xyz_orig: vec3f) -> vec2f {
  var xyz = xyz_orig;
  xyz *= 2.5;
  let result = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), wgsl_mod(abs(xyz.z) * 6.0, 1.0) <= 0.5);
  let angle = -1.3;
  let c = cos(angle);
  let s = sin(angle);
  let y_new = xyz.y * c - xyz.z * s;
  let z_new = xyz.y * s + xyz.z * c;
  xyz.y = y_new;
  xyz.z = z_new;
  let xyz_abs = abs(xyz);
  let f = pow(pow(pow(xyz_abs.x, 2.0 / e2) + pow(xyz_abs.y, 2.0 / e2), e2 / 2.0) - a4, 2.0 / e1) + pow(xyz_abs.z, 2.0 / e1);
  return select(vec2f(0.0), result, f <= 1.0);
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var mat_xy = vec2f(0.0);
  let e = vec4f(0.1, 0.3, 1.0, 3.0);
  let o = vec4f(-3.4, - 1.2, 1.2, 3.4);
  for(var i = 0; i < 4; i ++ ) {
    for(var j = 0; j < 4; j ++ ) {
      mat_xy += superquad2(6.0, e[i], e[j], 1.5, xyz - vec3f(o[j], 0.0, o[3 - i]));
    }
  }
  return vec4f(mat_xy, 0.0, 0.0);
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

const superquad2 = (slices, e1, e2, a4, p) => {
  let q = p.mul(2.5);
  const pick = mod(q.z.abs().mul(6.0), 1.0).lessThanEqual(0.5);
  const angle = -1.3;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const y2 = q.y.mul(c).add(q.z.mul(s));
  const z2 = q.y.mul(s).negate().add(q.z.mul(c));
  q = vec3(q.x, y2, z2).abs(); // Due to GLSL 'pow' definition.
  const f = pow(
    pow(pow(q.x, 2.0 / e2).add(pow(q.y, 2.0 / e2)), e2 / 2.0).sub(a4),
    2.0 / e1
  ).add(pow(q.z, 2.0 / e1));
  const inside = f.lessThanEqual(1.0);
  return [
    select(inside, select(pick, float(1.0), float(0.0)), float(0.0)),
    select(inside, select(pick, float(0.0), float(1.0)), float(0.0)),
  ];
};

const mainModel4 = (xyz) => {
  const e = [0.1, 0.3, 1.0, 3.0];
  const o = [-3.4, -1.2, 1.2, 3.4];
  let m0 = float(0.0);
  let m1 = float(0.0);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const [a, b] = superquad2(6.0, e[i], e[j], 1.5, xyz.sub(vec3(o[j], 0, o[3 - i])));
      m0 = m0.add(a);
      m1 = m1.add(b);
    }
  }
  return [m0, m1];
};
```

* Try loading [superquad-toroids-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-toroids-2.irmf) now in the IRMF editor!
* Try loading [superquad-toroids-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-toroids-2-wgsl.irmf) now in the IRMF editor!
* Try loading [superquad-toroids-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/superquad-toroids-2-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## sphericon-1.irmf

A [sphericon](https://en.wikipedia.org/wiki/Sphericon) is easy to make
out of two half-superquadrics.

![sphericon-1.png](sphericon-1.png)

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

float superquad(in float e1, in float e2, in vec3 xyz) {
  xyz = abs(xyz); // Due to GLSL 'pow' definition.
  float f = pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / e1) + pow(xyz.z, 2.0 / e1);
  return f <= 1.0 ? 1.0 : 0.0;
}

float sphericon(in vec3 xyz) {
  if (xyz.x <= 0.0) {
    return superquad(2.0, 1.0, xyz);
  }
  return superquad(2.0, 1.0, xyz.xzy);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz /= 5.0;
  materials[0] = sphericon(xyz);
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

fn superquad(e1: f32, e2: f32, xyz_in: vec3f) -> f32 {
  let xyz = abs(xyz_in);
  let f = pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / e1) + pow(xyz.z, 2.0 / e1);
  return select(0.0, 1.0, f <= 1.0);
}

fn sphericon(xyz: vec3f) -> f32 {
  if (xyz.x <= 0.0) {
    return superquad(2.0, 1.0, xyz);
  }
  return superquad(2.0, 1.0, xyz.xzy);
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  let xyz = xyz_in / 5.0;
  var materials = vec4f(0.0);
  materials[0] = sphericon(xyz);
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

const superquad = (e1, e2, xyz) => {
  const p = xyz.abs(); // Due to GLSL 'pow' definition.
  const f = pow(
    pow(p.x, 2.0 / e2).add(pow(p.y, 2.0 / e2)),
    e2 / e1
  ).add(pow(p.z, 2.0 / e1));
  return select(f.lessThanEqual(1.0), float(1.0), float(0.0));
};

const sphericon = (xyz) => {
  const a = superquad(2.0, 1.0, xyz);
  const b = superquad(2.0, 1.0, xyz.xzy);
  return select(xyz.x.lessThanEqual(0.0), a, b);
};

const mainModel4 = (xyz) => [sphericon(xyz.div(5.0))];
```

* Try loading [sphericon-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/sphericon-1.irmf) now in the IRMF editor!
* Try loading [sphericon-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/sphericon-1-wgsl.irmf) now in the IRMF editor!
* Try loading [sphericon-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/sphericon-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## sphericon-2.irmf

It turns out that it is much easier to visualize a sphericon in two colors
(at least in the IRMF editor), which translates to using two materials.

![sphericon-2.png](sphericon-2.png)

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

float superquad(in float e1, in float e2, in vec3 xyz) {
  xyz = abs(xyz); // Due to GLSL 'pow' definition.
  float f = pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / e1) + pow(xyz.z, 2.0 / e1);
  return f <= 1.0 ? 1.0 : 0.0;
}

vec2 sphericon2(in float slices, in vec3 xyz) {
  if (xyz.x <= 0.0) {
    float v = superquad(2.0, 1.0, xyz);
    if (mod(abs(xyz.z) * slices, 1.0) <= 0.5) { return vec2(v, 0); }
    return vec2(0, v);
  }
  float v = superquad(2.0, 1.0, xyz.xzy);
  if (mod(-abs(xyz.y) * slices, 1.0) <= 0.5) { return vec2(v, 0); }
  return vec2(0, v);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz /= 5.0;
  materials.xy = sphericon2(6.0, xyz);
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

fn superquad(e1: f32, e2: f32, xyz_in: vec3f) -> f32 {
  let xyz = abs(xyz_in);
  let f = pow(pow(xyz.x, 2.0 / e2) + pow(xyz.y, 2.0 / e2), e2 / e1) + pow(xyz.z, 2.0 / e1);
  return select(0.0, 1.0, f <= 1.0);
}

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn sphericon2(slices: f32, xyz: vec3f) -> vec2f {
  if (xyz.x <= 0.0) {
    let v = superquad(2.0, 1.0, xyz);
    if (wgsl_mod(abs(xyz.z) * slices, 1.0) <= 0.5) { return vec2f(v, 0.0); }
    return vec2f(0.0, v);
  }
  let v = superquad(2.0, 1.0, xyz.xzy);
  if (wgsl_mod(-abs(xyz.y) * slices, 1.0) <= 0.5) { return vec2f(v, 0.0); }
  return vec2f(0.0, v);
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  let xyz = xyz_in / 5.0;
  var materials = vec4f(0.0);
  let res = sphericon2(6.0, xyz);
  materials[0] = res.x;
  materials[1] = res.y;
  return materials;
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

const superquad = (e1, e2, p) => {
  const q = p.abs(); // Due to GLSL 'pow' definition.
  const f = pow(pow(q.x, 2.0 / e2).add(pow(q.y, 2.0 / e2)), e2 / e1).add(pow(q.z, 2.0 / e1));
  return select(f.lessThanEqual(1.0), float(1.0), float(0.0));
};

const sphericon2 = (slices, p) => {
  const leftSide = p.x.lessThanEqual(0.0);
  const lv = superquad(2.0, 1.0, p);
  const rv = superquad(2.0, 1.0, p.xzy);
  const lp = mod(p.z.abs().mul(slices), 1.0).lessThanEqual(0.5);
  const rp = mod(p.y.abs().mul(slices).negate(), 1.0).lessThanEqual(0.5);
  return [
    select(leftSide, select(lp, lv, float(0)), select(rp, rv, float(0))),
    select(leftSide, select(lp, float(0), lv), select(rp, float(0), rv)),
  ];
};

const mainModel4 = (xyz) => sphericon2(6.0, xyz.div(5.0));
```

* Try loading [sphericon-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/sphericon-2.irmf) now in the IRMF editor!
* Try loading [sphericon-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/sphericon-2-wgsl.irmf) now in the IRMF editor!
* Try loading [sphericon-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/022-superquadrics/sphericon-2-tsl.irmf) now in the IRMF editor!

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
