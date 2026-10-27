# 008-spiral-square-face

## spiral-1.irmf

To make a Nikola Tesla bifilar coil, we need to be able to model a
spiral, and one with a square face cross-section is easier, so
we'll start with that.

![spiral-1.png](spiral-1.png)

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

void mainModel4(out vec4 materials, in vec3 xyz) {
  materials[0] = spiralSquareFace(3.0, 0.85, 0.15, 2.0, xyz);
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

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = spiralSquareFace(3.0, 0.85, 0.15, 2.0, xyz);
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

const spiralSquareFace = (startRadius, size, gap, nTurns, xyz) => {
  // First, trivial reject above and below the spiral.
  const zOK = xyz.z.greaterThanEqual(-0.5 * size).and(xyz.z.lessThanEqual(0.5 * size));
  const r = length(xyz.xy);
  const rOK = r.greaterThanEqual(startRadius - 0.5 * size).and(
    r.lessThanEqual(startRadius + 0.5 * size + (size + gap) * nTurns)
  );

  let angle = atan(xyz.y, xyz.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle); // 0 <= angle <= 1
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

const mainModel4 = (xyz) => [spiralSquareFace(3.0, 0.85, 0.15, 2.0, xyz)];
```

* Try loading [spiral-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/008-spiral-square-face/spiral-1.irmf) now in the IRMF editor!
* Try loading [spiral-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/008-spiral-square-face/spiral-1-wgsl.irmf) now in the IRMF editor!
* Try loading [spiral-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/008-spiral-square-face/spiral-1-tsl.irmf) now in the IRMF editor!

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
