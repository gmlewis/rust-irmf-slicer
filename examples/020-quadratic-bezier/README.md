# 020-quadratic-bezier

## quadratic-bezier-1.irmf

It turns out that a quadratic bezier is also relatively easy to model and
demonstrates a smooth curve that STL can not represent well:

![quadratic-bezier-1.png](quadratic-bezier-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA"],
  max: [5,5,5],
  min: [-5,-5,0],
  units: "mm",
}*/

float interpQuadratic(vec2 p0,vec2 p1,vec2 p2,float y){
  float a=p2.y+p0.y-2.*p1.y;
  float b=2.*(p1.y-p0.y);
  float c=p0.y-y;
  if(b*b<4.*a*c){return 0.;}// bad (imaginary) quadratic
  float det=sqrt(b*b-4.*a*c);
  float t=(-b+det)/(2.*a);
  float x=(1.-t)*(1.-t)*p0.x+2.*(1.-t)*t*p1.x+t*t*p2.x;
  return x;
}

float qBezier(vec2 p0,vec2 p1,vec2 p2,in vec3 xyz){
  float r=length(xyz.xy);
  if(r>interpQuadratic(p0,p1,p2,xyz.z)){return 0.;}
  return 1.;
}

void mainModel4(out vec4 materials,in vec3 xyz){
  xyz.z+=1.;
  materials[0]=qBezier(vec2(-2,2),vec2(0,0),vec2(2,2),xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA"],
  max: [5,5,5],
  min: [-5,-5,0],
  units: "mm",
}*/

fn interpQuadratic(p0: vec2f, p1: vec2f, p2: vec2f, y: f32) -> f32 {
  let a = p2.y + p0.y - 2.0*p1.y;
  let b = 2.0 * (p1.y - p0.y);
  let c = p0.y - y;
  if (b*b < 4.0*a*c) { return 0.0; }
  let det = sqrt(b*b - 4.0*a*c);
  let t = (-b + det) / (2.0 * a);
  let x = (1.0-t)*(1.0-t)*p0.x + 2.0*(1.0-t)*t*p1.x + t*t*p2.x;
  return x;
}

fn qBezier(p0: vec2f, p1: vec2f, p2: vec2f, xyz: vec3f) -> f32 {
  let r = length(xyz.xy);
  if (r > interpQuadratic(p0, p1, p2, xyz.z)) { return 0.0; }
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 1.0;
  var materials = vec4f(0.0);
  materials[0] = qBezier(vec2f(-2.0, 2.0), vec2f(0.0, 0.0), vec2f(2.0, 2.0), xyz);
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
  min: [-5,-5,0],
  units: "mm",
}*/

const interpQuadratic = (p0, p1, p2, y) => {
  const a = float(p2.y + p0.y - 2 * p1.y);
  const b = float(2 * (p1.y - p0.y));
  const c = float(p0.y).sub(y);
  const disc = b.mul(b).sub(a.mul(c).mul(4.0));
  const bad = b.mul(b).lessThan(a.mul(c).mul(4.0)); // imaginary
  const det = sqrt(disc.max(0.0));
  const t = det.sub(b).div(a.mul(2.0));
  const omt = float(1.0).sub(t);
  const x = omt.mul(omt).mul(p0.x)
    .add(omt.mul(t).mul(2.0).mul(p1.x))
    .add(t.mul(t).mul(p2.x));
  return select(bad, float(0.0), x);
};

const qBezier = (p0, p1, p2, xyz) => {
  const r = length(xyz.xy);
  const x = interpQuadratic(p0, p1, p2, xyz.z);
  return select(r.greaterThan(x), float(0.0), float(1.0));
};

const mainModel4 = (xyz) => [
  qBezier({ x: -2, y: 2 }, { x: 0, y: 0 }, { x: 2, y: 2 }, xyz.add(vec3(0, 0, 1.0))),
];
```

* Try loading [quadratic-bezier-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/020-quadratic-bezier/quadratic-bezier-1.irmf) now in the IRMF editor!
* Try loading [quadratic-bezier-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/020-quadratic-bezier/quadratic-bezier-1-wgsl.irmf) now in the IRMF editor!
* Try loading [quadratic-bezier-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/020-quadratic-bezier/quadratic-bezier-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## quadratic-bezier-2.irmf

We can hollow out the interior with a second quadratic bezier:

![quadratic-bezier-2.png](quadratic-bezier-2.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA"],
  max: [5,5,5],
  min: [-5,-5,0],
  units: "mm",
}*/

float interpQuadratic(vec2 p0,vec2 p1,vec2 p2,float y){
  float a=p2.y+p0.y-2.*p1.y;
  float b=2.*(p1.y-p0.y);
  float c=p0.y-y;
  if(b*b<4.*a*c){return 0.;}// bad (imaginary) quadratic
  float det=sqrt(b*b-4.*a*c);
  float t=(-b+det)/(2.*a);
  float x=(1.-t)*(1.-t)*p0.x+2.*(1.-t)*t*p1.x+t*t*p2.x;
  return x;
}

float qBezier(vec2 p0,vec2 p1,vec2 p2,in vec3 xyz){
  float r=length(xyz.xy);
  if(r>interpQuadratic(p0,p1,p2,xyz.z)){return 0.;}
  return 1.;
}

void mainModel4(out vec4 materials,in vec3 xyz){
  xyz.z+=1.;
  float thickness=.5;
  materials[0]=qBezier(vec2(-2,2),vec2(0,0),vec2(2,2),xyz)-
  qBezier(vec2(-2,2.+thickness),vec2(0,thickness),vec2(2,2.+thickness),xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA"],
  max: [5,5,5],
  min: [-5,-5,0],
  units: "mm",
}*/

fn interpQuadratic(p0: vec2f, p1: vec2f, p2: vec2f, y: f32) -> f32 {
  let a = p2.y + p0.y - 2.0*p1.y;
  let b = 2.0 * (p1.y - p0.y);
  let c = p0.y - y;
  if (b*b < 4.0*a*c) { return 0.0; }
  let det = sqrt(b*b - 4.0*a*c);
  let t = (-b + det) / (2.0 * a);
  let x = (1.0-t)*(1.0-t)*p0.x + 2.0*(1.0-t)*t*p1.x + t*t*p2.x;
  return x;
}

fn qBezier(p0: vec2f, p1: vec2f, p2: vec2f, xyz: vec3f) -> f32 {
  let r = length(xyz.xy);
  if (r > interpQuadratic(p0, p1, p2, xyz.z)) { return 0.0; }
  return 1.0;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 1.0;
  let thickness = 0.5;
  var materials = vec4f(0.0);
  materials[0] = qBezier(vec2f(-2.0, 2.0), vec2f(0.0, 0.0), vec2f(2.0, 2.0), xyz) -
                 qBezier(vec2f(-2.0, 2.0 + thickness), vec2f(0.0, thickness), vec2f(2.0, 2.0 + thickness), xyz);
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
  min: [-5,-5,0],
  units: "mm",
}*/

const interpQuadratic = (p0, p1, p2, y) => {
  const a = float(p2.y + p0.y - 2 * p1.y);
  const b = float(2 * (p1.y - p0.y));
  const c = float(p0.y).sub(y);
  const disc = b.mul(b).sub(a.mul(c).mul(4.0));
  const bad = b.mul(b).lessThan(a.mul(c).mul(4.0)); // imaginary
  const det = sqrt(disc.max(0.0));
  const t = det.sub(b).div(a.mul(2.0));
  const omt = float(1.0).sub(t);
  const x = omt.mul(omt).mul(p0.x)
    .add(omt.mul(t).mul(2.0).mul(p1.x))
    .add(t.mul(t).mul(p2.x));
  return select(bad, float(0.0), x);
};

const qBezier = (p0, p1, p2, p) => {
  const r = length(p.xy);
  return select(r.greaterThan(interpQuadratic(p0, p1, p2, p.z)), float(0.0), float(1.0));
};

const mainModel4 = (xyz) => {
  const p = xyz.add(vec3(0, 0, 1.0));
  const thickness = 0.5;
  return [
    qBezier({ x: -2, y: 2 }, { x: 0, y: 0 }, { x: 2, y: 2 }, p).sub(
      qBezier(
        { x: -2, y: 2.0 + thickness },
        { x: 0, y: thickness },
        { x: 2, y: 2.0 + thickness },
        p
      )
    ),
  ];
};
```

* Try loading [quadratic-bezier-2.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/020-quadratic-bezier/quadratic-bezier-2.irmf) now in the IRMF editor!
* Try loading [quadratic-bezier-2-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/020-quadratic-bezier/quadratic-bezier-2-wgsl.irmf) now in the IRMF editor!
* Try loading [quadratic-bezier-2-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/020-quadratic-bezier/quadratic-bezier-2-tsl.irmf) now in the IRMF editor!

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
