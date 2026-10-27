# 018-rodin-coil

## rodin-coil-1.irmf

The Rodin coil is an interesting structure with some fairly unusual
electromagnetic properties. Search YouTube for more information.

In order to work our way up to a full coil with copper wires, let's
first model the wire bundles as a single massive wire, then in the
next version, we will model the individual wires within each bundle.

One really cool property about these models is that the resulting
copper wire is a single very long wire that can be tapped at one
point to make "start" and "end" wires.

![rodin-coil-1.png](rodin-coil-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper"],
  max: [170,150,50],
  min: [-150,-150,-50],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float torus(float majorRadius,float minorRadius,in vec3 xyz){
  float r=length(xyz);
  if(r>majorRadius+minorRadius||r<majorRadius-minorRadius){return 0.;}
  
  float angle=atan(xyz.y,xyz.x);
  vec3 center=vec3(majorRadius*cos(angle),majorRadius*sin(angle),0);
  vec3 v=xyz-center;
  float r2=length(v);
  if(r2>minorRadius){return 0.;}
  
  return 1.;
}

mat3 rotAxis(vec3 axis,float a){
  // This is from: http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
  float s=sin(a);
  float c=cos(a);
  float oc=1.-c;
  vec3 as=axis*s;
  mat3 p=mat3(axis.x*axis,axis.y*axis,axis.z*axis);
  mat3 q=mat3(c,-as.z,as.y,as.z,c,-as.x,-as.y,as.x,c);
  return p*oc+q;
}

mat4 rotX(float angle){
  return mat4(rotAxis(vec3(1,0,0),angle));
}

mat4 rotY(float angle){
  return mat4(rotAxis(vec3(0,1,0),angle));
}

mat4 rotZ(float angle){
  return mat4(rotAxis(vec3(0,0,1),angle));
}

float cube(in mat4 xfm,in float size, in vec3 xyz){
  xyz=(vec4(xyz,1)*xfm).xyz;
  if(any(greaterThan(abs(xyz),vec3(size)))){return 0.;}
  return 1.;
}

float rodinCoil(float majorRadius,float minorRadius,float bundleRadius,float numBundles,int twistsPerRevolution,in vec3 xyz){
  float r=length(xyz);
  if(r>majorRadius+minorRadius||r<majorRadius-minorRadius){return 0.;}
  
  float angle=atan(xyz.y,xyz.x);
  if(angle<0.){angle+=(2.*M_PI);}
  vec3 torusSliceXYZ=(vec4(xyz,1)*rotZ(-angle)).xyz-vec3(majorRadius,0,0);
  vec3 twistedSliceXYZ=(vec4(torusSliceXYZ,1)*rotY((float(twistsPerRevolution)+(1./numBundles))*angle)).xyz;
  
  float subAngle=atan(twistedSliceXYZ.z,twistedSliceXYZ.x);
  if(subAngle<0.){subAngle+=(2.*M_PI);}
  float bundleNum=floor(subAngle*numBundles/(2.*M_PI)+.5*numBundles/(2.*M_PI));
  float bundleAngle=2.*M_PI*bundleNum/numBundles;
  float offsetRadius=minorRadius-bundleRadius;
  vec3 bundleCenter=vec3(offsetRadius*cos(bundleAngle),0.,offsetRadius*sin(bundleAngle));
  vec3 subBundleXYZ=twistedSliceXYZ-bundleCenter;
  float r2=length(subBundleXYZ.xz);
  if(r2>bundleRadius){return 0.;}
  
  return 1.;
}

float wire(in mat4 xfm,vec3 start,vec3 end,float size,in vec3 xyz){
  xyz=(vec4(xyz,1)*xfm).xyz;
  if(xyz.x<start.x-size||xyz.x>end.x+size){return 0.;}
  xyz-=start;
  float r=length(xyz.yz);
  if(r>.75*size){return 0.;}
  return 1.;
}

float wiredRodinCoil(float majorRadius,float minorRadius,float bundleRadius,int numBundles,int twistsPerRevolution,in vec3 xyz){
  float coil=rodinCoil(majorRadius,minorRadius,bundleRadius,float(numBundles),twistsPerRevolution,xyz);
  float x=majorRadius+minorRadius-bundleRadius;
  mat4 rot=rotX(.25*M_PI);
  coil-=cube(rot,bundleRadius,xyz-vec3(x,0,0));
  coil+=wire(rot,vec3(x,0,bundleRadius),vec3(x+4.*bundleRadius,0,bundleRadius),bundleRadius,xyz);
  coil+=wire(rot,vec3(x,0,-bundleRadius),vec3(x+4.*bundleRadius,0,-bundleRadius),bundleRadius,xyz);
  
  return coil;
}

void mainModel4(out vec4 materials,in vec3 xyz){
  materials[0]=wiredRodinCoil(100.,50.,5.,9,3,xyz);
  // materials[1]=torus(100.,50.,xyz)-materials[0];
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper"],
  max: [170,150,50],
  min: [-150,-150,-50],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn rotAxis(axis: vec3f, a: f32) -> mat3x3f {
  let s = sin(a);
  let c = cos(a);
  let oc = 1.0 - c;
  let as_ = axis * s;
  let p = mat3x3f(axis.x * axis, axis.y * axis, axis.z * axis);
  let q = mat3x3f(c, as_.z, -as_.y, -as_.z, c, as_.x, as_.y, -as_.x, c);
  return p * oc + q;
}

fn mat3ToMat4(m: mat3x3f) -> mat4x4f {
  return mat4x4f(
    vec4f(m[0], 0.0),
    vec4f(m[1], 0.0),
    vec4f(m[2], 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0)
  );
}

fn rotX(angle: f32) -> mat4x4f {
  return mat3ToMat4(rotAxis(vec3f(1.0,0.0,0.0), angle));
}

fn rotY(angle: f32) -> mat4x4f {
  return mat3ToMat4(rotAxis(vec3f(0.0,1.0,0.0), angle));
}

fn rotZ(angle: f32) -> mat4x4f {
  return mat3ToMat4(rotAxis(vec3f(0.0,0.0,1.0), angle));
}

fn cube(xfm: mat4x4f, size: f32, xyz: vec3f) -> f32 {
  let xyz_xfm = (xfm * vec4f(xyz, 1.0)).xyz;
  if (any(abs(xyz_xfm) > vec3f(size))) { return 0.0; }
  return 1.0;
}

fn rodinCoil(majorRadius: f32, minorRadius: f32, bundleRadius: f32, numBundles: f32, twistsPerRevolution: i32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  var angle = atan2(xyz.y, xyz.x);
  if (angle < 0.0) { angle += 2.0 * M_PI; }
  let torusSliceXYZ = (rotZ(-angle) * vec4f(xyz, 1.0)).xyz - vec3f(majorRadius, 0.0, 0.0);
  let twistedSliceXYZ = (rotY((f32(twistsPerRevolution) + (1.0/numBundles)) * angle) * vec4f(torusSliceXYZ, 1.0)).xyz;
  
  var subAngle = atan2(twistedSliceXYZ.z, twistedSliceXYZ.x);
  if (subAngle < 0.0) { subAngle += 2.0 * M_PI; }
  let bundleNum = floor(subAngle * numBundles / (2.0 * M_PI) + 0.5 * numBundles / (2.0 * M_PI));
  let bundleAngle = 2.0 * M_PI * bundleNum / numBundles;
  let offsetRadius = minorRadius - bundleRadius;
  let bundleCenter = vec3f(offsetRadius * cos(bundleAngle), 0.0, offsetRadius * sin(bundleAngle));
  let subBundleXYZ = twistedSliceXYZ - bundleCenter;
  let r2 = length(subBundleXYZ.xz);
  if (r2 > bundleRadius) { return 0.0; }
  
  return 1.0;
}

fn wire(xfm: mat4x4f, start: vec3f, end: vec3f, size: f32, xyz: vec3f) -> f32 {
  let xyz_xfm = (xfm * vec4f(xyz, 1.0)).xyz;
  if (xyz_xfm.x < start.x - size || xyz_xfm.x > end.x + size) { return 0.0; }
  let v = xyz_xfm - start;
  let r = length(v.yz);
  if (r > 0.75 * size) { return 0.0; }
  return 1.0;
}

fn wiredRodinCoil(majorRadius: f32, minorRadius: f32, bundleRadius: f32, numBundles: i32, twistsPerRevolution: i32, xyz: vec3f) -> f32 {
  var coil = rodinCoil(majorRadius, minorRadius, bundleRadius, f32(numBundles), twistsPerRevolution, xyz);
  let x = majorRadius + minorRadius - bundleRadius;
  let rot = rotX(0.25 * M_PI);
  coil -= cube(rot, bundleRadius, xyz - vec3f(x, 0.0, 0.0));
  coil += wire(rot, vec3f(x, 0.0, bundleRadius), vec3f(x + 4.0 * bundleRadius, 0.0, bundleRadius), bundleRadius, xyz);
  coil += wire(rot, vec3f(x, 0.0, -bundleRadius), vec3f(x + 4.0 * bundleRadius, 0.0, -bundleRadius), bundleRadius, xyz);
  
  return coil;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = wiredRodinCoil(100.0, 50.0, 5.0, 9, 3, xyz);
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
  max: [170,150,50],
  min: [-150,-150,-50],
  units: "mm",
}*/

const cosN = (a) => (typeof a === 'number' ? Math.cos(a) : cos(a));
const sinN = (a) => (typeof a === 'number' ? Math.sin(a) : sin(a));

// Standard rotations. GLSL's `v * mat` equals `transpose(mat) * v`, and the
// matrices in this shader are built so that this reduces to the standard
// rotation below (verified against the GLSL render).
const rotX = (p, a) => {
  const c = cosN(a);
  const s = sinN(a);
  return vec3(p.x, p.y.mul(c).sub(p.z.mul(s)), p.y.mul(s).add(p.z.mul(c)));
};
const rotY = (p, a) => {
  const c = cosN(a);
  const s = sinN(a);
  return vec3(p.x.mul(c).add(p.z.mul(s)), p.y, p.z.mul(c).sub(p.x.mul(s)));
};
const rotZ = (p, a) => {
  const c = cosN(a);
  const s = sinN(a);
  return vec3(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)), p.z);
};

const torus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  return select(ringOK.and(length(p.sub(center)).lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const cubeRot = (angle, size, p) => {
  const q = rotX(p, angle);
  return select(any(q.abs().greaterThan(size)), float(0.0), float(1.0));
};

const wireRot = (angle, start, end, size, p) => {
  const q = rotX(p, angle);
  const xOK = q.x.greaterThanEqual(start.x.sub(size)).and(q.x.lessThanEqual(end.x.add(size)));
  const qq = q.sub(start);
  const rOK = length(qq.yz).lessThanEqual(size * 0.75);
  return select(xOK.and(rOK), float(1.0), float(0.0));
};

const rodinCoil = (majorRadius, minorRadius, bundleRadius, numBundles, twistsPerRevolution, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );

  const angleRaw = atan(p.y, p.x);
  const angle = select(angleRaw.lessThan(0.0), angleRaw.add(2.0 * Math.PI), angleRaw);
  const torusSliceXYZ = rotZ(p, angle.negate()).sub(vec3(majorRadius, 0, 0));
  const twistedSliceXYZ = rotY(
    torusSliceXYZ,
    float(twistsPerRevolution + 1.0 / numBundles).mul(angle)
  );

  const subAngleRaw = atan(twistedSliceXYZ.z, twistedSliceXYZ.x);
  const subAngle = select(subAngleRaw.lessThan(0.0), subAngleRaw.add(2.0 * Math.PI), subAngleRaw);
  const bundleNum = floor(
    subAngle.mul(numBundles).div(2.0 * Math.PI).add(0.5 * numBundles / (2.0 * Math.PI))
  );
  const bundleAngle = bundleNum.mul(2.0 * Math.PI).div(numBundles);
  const offsetRadius = minorRadius - bundleRadius;
  const bundleCenter = vec3(
    cos(bundleAngle).mul(offsetRadius),
    float(0),
    sin(bundleAngle).mul(offsetRadius)
  );
  const subBundleXYZ = twistedSliceXYZ.sub(bundleCenter);
  const r2 = length(subBundleXYZ.xz);

  return select(ringOK.and(r2.lessThanEqual(bundleRadius)), float(1.0), float(0.0));
};

const wiredRodinCoil = (majorRadius, minorRadius, bundleRadius, numBundles, twistsPerRevolution, p) => {
  const coil = rodinCoil(majorRadius, minorRadius, bundleRadius, numBundles, twistsPerRevolution, p);
  const x = majorRadius + minorRadius - bundleRadius;
  const rot = 0.25 * Math.PI;
  let result = coil.sub(cubeRot(rot, bundleRadius, p.sub(vec3(x, 0, 0))));
  result = result.add(
    wireRot(
      rot,
      vec3(x, 0, bundleRadius),
      vec3(x + 4.0 * bundleRadius, 0, bundleRadius),
      bundleRadius,
      p
    )
  );
  result = result.add(
    wireRot(
      rot,
      vec3(x, 0, -bundleRadius),
      vec3(x + 4.0 * bundleRadius, 0, -bundleRadius),
      bundleRadius,
      p
    )
  );
  return result;
};

const mainModel4 = (xyz) => [wiredRodinCoil(100.0, 50.0, 5.0, 9, 3, xyz)];
```

* Try loading [rodin-coil-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/018-rodin-coil/rodin-coil-1.irmf) now in the IRMF editor!
* Try loading [rodin-coil-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/018-rodin-coil/rodin-coil-1-wgsl.irmf) now in the IRMF editor!
* Try loading [rodin-coil-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/018-rodin-coil/rodin-coil-1-tsl.irmf) now in the IRMF editor!

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
