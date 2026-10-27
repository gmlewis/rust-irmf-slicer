# 009-spiral-circle-face

## spiral-circle.irmf

Once the square-face spiral was done, it is a rather simple matter
to round the edges.

![spiral-circle.png](spiral-circle.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA"],
  max: [5.5,5.5,0.5],
  min: [-5.5,-5.5,-0.5],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float spiralCircleFace(float startRadius, float size, float gap, float nTurns, in vec3 xyz) {
  // First, trivial reject above and below the spiral.
  if (xyz.z < -0.5 * size || xyz.z > 0.5 * size) { return 0.0; }
  
  float r = length(xyz.xy);
  if (r < startRadius - 0.5 * size ||
      r > startRadius + 0.5 * size + (size + gap) * nTurns) { return 0.0; }
  
  // If the current point is between the spirals, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between spirals from center to center.
  // 0 <= dr <= (size+gap) between spirals from center to center.
  float dr = mod(r - startRadius, size + gap);
  
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

    // At this point, we are within the circle cross-section face, so let's round the edge.
  lastSpiralR = floor((r - startRadius) / (size + gap)) * (size + gap) +
    startRadius + angle * (size + gap);
  if (lastSpiralR > r) { lastSpiralR -= (size + gap); }
  nextSpiralR = lastSpiralR + size + gap;
  angle *= 2.0 * M_PI;
  vec3 lastSpiralCenter = vec3(lastSpiralR * cos(angle), lastSpiralR * sin(angle), 0.0);
  vec3 nextSpiralCenter = vec3(nextSpiralR * cos(angle), nextSpiralR * sin(angle), 0.0);
  float d1 = length(xyz - lastSpiralCenter);
  float d2 = length(xyz - nextSpiralCenter);
  if (d1 > 0.5 * size && d2 > 0.5 * size) { return 0.0; }
  
  return 1.0;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials[0] = spiralCircleFace(3.0, 0.85, 0.15, 2.0, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA"],
  max: [5.5,5.5,0.5],
  min: [-5.5,-5.5,-0.5],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn spiralCircleFace(startRadius: f32, size: f32, gap: f32, nTurns: f32, xyz: vec3f) -> f32 {
  // First, trivial reject above and below the spiral.
  if (xyz.z < -0.5 * size || xyz.z > 0.5 * size) { return 0.0; }
  
  let r = length(xyz.xy);
  if (r < startRadius - 0.5 * size ||
      r > startRadius + 0.5 * size + (size + gap) * nTurns) { return 0.0; }
  
  // If the current point is between the spirals, return no material:
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between spirals from center to center.
  // 0 <= dr <= (size+gap) between spirals from center to center.
  let dr = (r - startRadius) - (size + gap) * floor((r - startRadius) / (size + gap));
  
  var coilNum = 0.0;
  var lastSpiralR_tmp = angle * (size + gap);
  if (lastSpiralR_tmp > dr) {
    lastSpiralR_tmp -= (size + gap);  // center of current coil.
    coilNum = -1.0;
  }
  var nextSpiralR_tmp = lastSpiralR_tmp + (size + gap);  // center of next outer coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dr > lastSpiralR_tmp + 0.5 * size && dr < nextSpiralR_tmp - 0.5 * size) { return 0.0; }
  
  coilNum += floor((r - startRadius + (0.5 * size) - lastSpiralR_tmp) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }

    // At this point, we are within the circle cross-section face, so let's round the edge.
  var lastSpiralR = floor((r - startRadius) / (size + gap)) * (size + gap) +
    startRadius + angle * (size + gap);
  if (lastSpiralR > r) { lastSpiralR -= (size + gap); }
  let nextSpiralR = lastSpiralR + size + gap;
  let rad_angle = angle * 2.0 * M_PI;
  let lastSpiralCenter = vec3f(lastSpiralR * cos(rad_angle), lastSpiralR * sin(rad_angle), 0.0);
  let nextSpiralCenter = vec3f(nextSpiralR * cos(rad_angle), nextSpiralR * sin(rad_angle), 0.0);
  let d1 = length(xyz - lastSpiralCenter);
  let d2 = length(xyz - nextSpiralCenter);
  if (d1 > 0.5 * size && d2 > 0.5 * size) { return 0.0; }
  
  return 1.0;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = spiralCircleFace(3.0, 0.85, 0.15, 2.0, xyz);
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
  max: [5.5,5.5,0.5],
  min: [-5.5,-5.5,-0.5],
  units: "mm",
}*/

const spiralCircleFace = (startRadius, size, gap, nTurns, xyz) => {
  const zOK = xyz.z.greaterThanEqual(-0.5 * size).and(xyz.z.lessThanEqual(0.5 * size));
  const r = length(xyz.xy);
  const rOK = r.greaterThanEqual(startRadius - 0.5 * size).and(
    r.lessThanEqual(startRadius + 0.5 * size + (size + gap) * nTurns)
  );

  let angle = atan(xyz.y, xyz.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dr = mod(r.sub(startRadius), size + gap);

  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dr);
  const lastSpiralR0 = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextSpiralR0 = lastSpiralR0.add(size + gap);

  const inGap = dr.greaterThan(lastSpiralR0.add(0.5 * size)).and(
    dr.lessThan(nextSpiralR0.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(
    floor(r.sub(startRadius).add(0.5 * size).sub(lastSpiralR0).div(size + gap))
  );
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));

  // Round the edge of the circle cross-section face.
  const base2 = floor(r.sub(startRadius).div(size + gap))
    .mul(size + gap)
    .add(startRadius)
    .add(angle.mul(size + gap));
  const lastSpiralR = select(base2.greaterThan(r), base2.sub(size + gap), base2);
  const nextSpiralR = lastSpiralR.add(size + gap);
  const ang = angle.mul(2.0 * Math.PI);
  const lastSpiralCenter = vec3(lastSpiralR.mul(cos(ang)), lastSpiralR.mul(sin(ang)), float(0));
  const nextSpiralCenter = vec3(nextSpiralR.mul(cos(ang)), nextSpiralR.mul(sin(ang)), float(0));
  const d1 = length(xyz.sub(lastSpiralCenter));
  const d2 = length(xyz.sub(nextSpiralCenter));
  const roundOK = d1.lessThanEqual(0.5 * size).or(d2.lessThanEqual(0.5 * size));

  return select(
    zOK.and(rOK).and(inGap.not()).and(coilOK).and(roundOK),
    float(1.0),
    float(0.0)
  );
};

const mainModel4 = (xyz) => [spiralCircleFace(3.0, 0.85, 0.15, 2.0, xyz)];
```

* Try loading [spiral-circle.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/009-spiral-circle-face/spiral-circle.irmf) now in the IRMF editor!
* Try loading [spiral-circle-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/009-spiral-circle-face/spiral-circle-wgsl.irmf) now in the IRMF editor!
* Try loading [spiral-circle-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/009-spiral-circle-face/spiral-circle-tsl.irmf) now in the IRMF editor!

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
