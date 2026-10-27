# 021-line2d

## line2d-1.irmf

2D primitives can be useful for extruding. Let's start with a simple 2D line.
This one's start point is at the origin which is very common when writing
GLSL (and therefore IRMF) shaders. Transformations can be applied before
calling the function and this keeps each primitive much simpler as can be
seen in the `line2d-2` example below.

![line2d-1.png](line2d-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA"],
  max: [1.5,1,0.1],
  min: [-1.5,-1,-0.1],
  units: "mm",
}*/

float line2d(in float lineLength, in float width, in bool startRoundCap, in bool endRoundCap, in vec3 xyz) {
  if (xyz.x < 0.0) {
    if (startRoundCap) {
      float r = length(xyz.xy);
      if (r > 0.5 * width) { return 0.0; }
    } else { return 0.0; }
  }

  if (xyz.x > lineLength) {
    if (endRoundCap) {
      float r = length(xyz.xy - vec2(lineLength,0));
      if (r > 0.5 * width) { return 0.0; }
    } else { return 0.0; }
  }

  float r = length(xyz.y);
  if (r > 0.5 * width) { return 0.0; }

  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.x += 1.0;
  materials[0] = line2d(2.0, 0.5, true, true, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA"],
  max: [1.5,1,0.1],
  min: [-1.5,-1,-0.1],
  units: "mm",
}*/

fn line2d_len(lineLength: f32, width: f32, startRoundCap: bool, endRoundCap: bool, xyz: vec3f) -> f32 {
  if (xyz.x < 0.0) {
    if (startRoundCap) {
      let r = length(xyz.xy);
      if (r > 0.5 * width) { return 0.0; }
    } else { return 0.0; }
  }

  if (xyz.x > lineLength) {
    if (endRoundCap) {
      let r = length(xyz.xy - vec2f(lineLength, 0.0));
      if (r > 0.5 * width) { return 0.0; }
    } else { return 0.0; }
  }

  let r = abs(xyz.y);
  if (r > 0.5 * width) { return 0.0; }

  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.x += 1.0;
  var materials = vec4f(0.0);
  materials[0] = line2d_len(2.0, 0.5, true, true, xyz);
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
  max: [1.5,1,0.1],
  min: [-1.5,-1,-0.1],
  units: "mm",
}*/

const line2d = (lineLength, width, startRoundCap, endRoundCap, xyz) => {
  const bodyOK = xyz.y.abs().lessThanEqual(0.5 * width);
  const beforeOK = startRoundCap
    ? length(xyz.xy).lessThanEqual(0.5 * width)
    : bool(false);
  const afterOK = endRoundCap
    ? length(xyz.xy.sub(vec2(lineLength, 0))).lessThanEqual(0.5 * width)
    : bool(false);
  const regionOK = select(
    xyz.x.lessThan(0.0),
    beforeOK,
    select(xyz.x.greaterThan(lineLength), afterOK, bodyOK)
  );
  return select(regionOK, float(1.0), float(0.0));
};

const mainModel4 = (xyz) => [line2d(2.0, 0.5, true, true, xyz.add(vec3(1, 0, 0)))];
```

* Try loading [line2d-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/021-line2d/line2d-1.irmf) now in the IRMF editor!
* Try loading [line2d-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/021-line2d/line2d-1-wgsl.irmf) now in the IRMF editor!
* Try loading [line2d-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/021-line2d/line2d-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## line2d-2.irmf

Sometimes you just want to specify the start and end points.
GLSL allows you to use the same function name with different
parameter signatures and it figures out which version to call.
As a result, we can make a new `line2d` function that takes
a start and end point which then calls the old `line2d` from
`line2d-1` above that is located at the origin.

![line2d-2.png](line2d-2.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA"],
  max: [3,3,0.1],
  min: [-3,-3,-0.1],
  units: "mm",
}*/

float line2d(in float lineLength, in float width, in bool startRoundCap, in bool endRoundCap, in vec3 xyz) {
  if (xyz.x < 0.0) {
    if (startRoundCap) {
      float r = length(xyz.xy);
      if (r > 0.5 * width) { return 0.0; }
    } else { return 0.0; }
  }

  if (xyz.x > lineLength) {
    if (endRoundCap) {
      float r = length(xyz.xy - vec2(lineLength,0));
      if (r > 0.5 * width) { return 0.0; }
    } else { return 0.0; }
  }

  float r = length(xyz.y);
  if (r > 0.5 * width) { return 0.0; }

  return 1.0;
}

float line2d(in vec2 start, in vec2 end, in float width, in bool startRoundCap, in bool endRoundCap, in vec3 xyz) {
  xyz.xy -= start;
  vec2 diff = end - start;
  float lineLength = length(diff);
  float angle = atan(diff.y, diff.x);
  float c = cos(angle);
  float s = sin(angle);
  xyz.xy = mat2(c,-s,s,c) * xyz.xy;
  return line2d(lineLength, width, startRoundCap, endRoundCap, xyz);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials[0] = line2d(vec2(-1,-1), vec2(1,1), 0.5, true, true, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA"],
  max: [3,3,0.1],
  min: [-3,-3,-0.1],
  units: "mm",
}*/

fn line2d_len(lineLength: f32, width: f32, startRoundCap: bool, endRoundCap: bool, xyz: vec3f) -> f32 {
  if (xyz.x < 0.0) {
    if (startRoundCap) {
      let r = length(xyz.xy);
      if (r > 0.5 * width) { return 0.0; }
    } else { return 0.0; }
  }

  if (xyz.x > lineLength) {
    if (endRoundCap) {
      let r = length(xyz.xy - vec2f(lineLength, 0.0));
      if (r > 0.5 * width) { return 0.0; }
    } else { return 0.0; }
  }

  let r = abs(xyz.y);
  if (r > 0.5 * width) { return 0.0; }

  return 1.0;
}

fn line2d_pts(start: vec2f, end: vec2f, width: f32, startRoundCap: bool, endRoundCap: bool, xyz_orig: vec3f) -> f32 {
  var xyz = xyz_orig;
  let xy = xyz.xy - start;
  let diff = end - start;
  let lineLength = length(diff);
  let angle = atan2(diff.y, diff.x);
  let c = cos(angle);
  let s = sin(angle);
  let x = xy.x * c + xy.y * s;
  let y = xy.x * -s + xy.y * c;
  xyz.x = x;
  xyz.y = y;
  return line2d_len(lineLength, width, startRoundCap, endRoundCap, xyz);
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = line2d_pts(vec2f(-1.0, -1.0), vec2f(1.0, 1.0), 0.5, true, true, xyz);
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
  max: [3,3,0.1],
  min: [-3,-3,-0.1],
  units: "mm",
}*/

const line2dLen = (lineLength, width, startRoundCap, endRoundCap, p) => {
  const bodyOK = p.y.abs().lessThanEqual(0.5 * width);
  const beforeOK = startRoundCap
    ? length(p.xy).lessThanEqual(0.5 * width)
    : bool(false);
  const afterOK = endRoundCap
    ? length(p.xy.sub(vec2(lineLength, 0))).lessThanEqual(0.5 * width)
    : bool(false);
  const regionOK = select(
    p.x.lessThan(0.0),
    beforeOK,
    select(p.x.greaterThan(lineLength), afterOK, bodyOK)
  );
  return select(regionOK, float(1.0), float(0.0));
};

const line2dSeg = (sx, sy, ex, ey, width, startRoundCap, endRoundCap, p) => {
  const q = p.xy.sub(vec2(sx, sy));
  const dx = ex - sx;
  const dy = ey - sy;
  const lineLength = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  // GLSL: xyz.xy = mat2(c,-s,s,c) * xyz.xy
  const rx = q.x.mul(c).add(q.y.mul(s));
  const ry = q.y.mul(c).sub(q.x.mul(s));
  return line2dLen(lineLength, width, startRoundCap, endRoundCap, vec3(rx, ry, p.z));
};

const mainModel4 = (xyz) => [
  line2dSeg(-1, -1, 1, 1, 0.5, true, true, xyz),
];
```

* Try loading [line2d-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/021-line2d/line2d-2.irmf) now in the IRMF editor!
* Try loading [line2d-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/021-line2d/line2d-2-wgsl.irmf) now in the IRMF editor!
* Try loading [line2d-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/021-line2d/line2d-2-tsl.irmf) now in the IRMF editor!

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
