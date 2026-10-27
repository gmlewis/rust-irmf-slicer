# 012-bifilar-electromagnet

**NOTE**: For a more modern design of the bifilar electromagnet with
a stronger wire cage and support material to hold the coils, please see:
https://github.com/gmlewis/moonbit-step/tree/master/examples/12-bifilar-electromagnet#012-bifilar-electromagnet

## bifilar-electromagnet-1.irmf

This is the model I wanted to build using convential CAD tools that simply
were not capable of handling the complexity without jumping through hoops
to accomodate the inner workings of the tools (typically by breaking up
the model into much smaller parts that it could more easily handle).

So I determined that there must be a better way, and I believe I have
finally found it... IRMF shaders. The next step will be to get IRMF shader
support built into the 3D printers themselves so that these shaders can
be sent directly to the printer as input, and out comes the part as fast
as the printer can make it. No STL. No slicing. No G-Code. Just the
IRMF shader.

It's also interesting to note that this model uses 6845 bytes as an IRMF
shader. If you slice this model at 42 micron resolution (with my experimental
[IRMF slicer](https://github.com/gmlewis/irmf-slicer)), it generates
a ZIP file around 135MB in size, and so far there are no traditional CAD tools
(free or commercial) that can even generate an STL file for it (due to the
complex boolean operation of subtracting the helical windings from the
dielectric material). Even if they could, the resulting STL file would be
absolutely enormous... much larger than online 3D printing sites support.

![bifilar-electromagnet-1.png](bifilar-electromagnet-1.png)

Here's a cut-away view of the same model showing the inner winding structure:

![cut-away](cutaway.png)

Here's a video describing the process of how I came up with this model:

[![Axial plus Radial Bifilar Electromagnet](https://img.youtube.com/vi/CV303dJUbq0/0.jpg)](https://www.youtube.com/watch?v=CV303dJUbq0)

and here's an animation of the slices of the electromagnet:

[![MRI of Axial plus Radial Bifilar Electromagnet](https://img.youtube.com/vi/pV56Pq2dErc/0.jpg)](https://www.youtube.com/watch?v=pV56Pq2dErc)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-63],
  units: "mm",
}*/

#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/rotation.glsl"
#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/primitives.glsl"

#define M_PI 3.1415926535897932384626433832795

float coilSquareFace(float radius,float size,float gap,float nTurns,float trimEndAngle,in vec3 xyz){
  // First, trivial reject on the two vertical ends of the coil.
  if(xyz.z<-.5*size||xyz.z>nTurns*(size+gap)+.5*size){return 0.;}
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  float rxy=length(xyz.xy);
  if(rxy<radius-.5*size||rxy>radius+.5*size){return 0.;}
  
  // If the current point is between the coils, return no material:
  float angle=atan(xyz.y,xyz.x)/(2.*M_PI);
  if(angle<0.){angle+=1.;}// 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  float dz=mod(xyz.z,size+gap);
  
  float lastHelixZ=angle*(size+gap);
  float coilNum=0.;
  if(lastHelixZ>dz){
    lastHelixZ-=(size+gap);// center of current coil.
    coilNum=-1.;
  }
  float nextHelixZ=lastHelixZ+(size+gap);// center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if(dz>lastHelixZ+.5*size&&dz<nextHelixZ-.5*size){return 0.;}
  
  coilNum+=floor((xyz.z+(.5*size)-lastHelixZ)/(size+gap));
  
  // If the current point is in a coil numbered outside the current range, reject it.
  if(coilNum<0.||coilNum>=nTurns){return 0.;}
  
  // TODO: Incorporate the trimEndAngle.
  
  return 1.;
}

float coilPlusConnectorWires(int coilNum,int numCoils,float inc,float innerRadius,float connectorRadius,float size,float gap,float nTurns,in vec3 xyz){
  float radiusOffset=float(coilNum-1);
  mat4 xfm=mat4(1)*rotZ(radiusOffset*inc);
  float coilRadius=radiusOffset+innerRadius;
  float trimEndAngle=2.*inc;
  if(coilNum==numCoils){
    trimEndAngle=.5*inc;// Special case to access the exit wire.
  }else if(coilNum==numCoils-1){
    trimEndAngle=3.*inc;
  }
  xyz=(vec4(xyz,1.)*xfm).xyz;
  float coil=coilSquareFace(coilRadius,size,gap,nTurns,trimEndAngle,xyz);
  
  float bz=-(size+gap);
  float tz=nTurns*(size+gap);
  float tzp1=(nTurns+1.)*(size+gap);
  
  coil+=box(vec3(coilRadius,0.,0.),vec3(coilRadius,0.,bz),size,xyz);
  coil+=box(vec3(coilRadius,0.,bz),vec3(connectorRadius,0.,bz),size,xyz);
  coil+=box(vec3(connectorRadius,0.,bz),vec3(connectorRadius,0.,tzp1),size,xyz);
  
  float zexit=(nTurns+10.)*(size+gap);
  if(coilNum>=3){// Connect the start of this coil to the end of two coils prior.
    float lastCoilRadius=radiusOffset-2.+innerRadius;
    coil+=box(vec3(lastCoilRadius,0.,tzp1),vec3(connectorRadius,0.,tzp1),size,xyz);
    coil+=box(vec3(lastCoilRadius,0.,tz),vec3(lastCoilRadius,0.,tzp1),size,xyz);
  }else if(coilNum==2){// Connect the start of 2 to the end of the last odd coil.
    float endOddRadius=float(numCoils-2)+innerRadius;
    coil+=box(vec3(endOddRadius,0.,tzp1),vec3(connectorRadius,0.,tzp1),size,xyz);
    coil+=box(vec3(endOddRadius,0.,tz),vec3(endOddRadius,0.,tzp1),size,xyz);
  }else if(coilNum==1){// Bring out the exit wires.
    // Start of coil1:
    coil+=box(vec3(connectorRadius,0.,tz),vec3(connectorRadius,0.,zexit),size,xyz);
  }
  
  if(coilNum==numCoils){// Special case to access the exit wire.
    // End of coil 'numCoils':
    xfm=mat4(1)*rotZ(.5*inc);
    xyz=(vec4(xyz,1.)*xfm).xyz;
    coil+=box(vec3(connectorRadius-(size+gap),0.,tz),vec3(connectorRadius-(size+gap),0.,zexit),size,xyz);
  }
  
  return coil;
}

vec3 bifilarElectromagnet(int numPairs,float innerRadius,float size,float gap,int numTurns,in vec3 xyz){
  float nTurns=float(numTurns);
  int numCoils=2*numPairs;
  float inc=2.*M_PI/float(numCoils);
  float connectorRadius=innerRadius+float(numCoils)*(size+gap);
  
  float metal1=0.;
  float metal2=0.;
  
  for(int i=1;i<=numCoils;i+=2){
    metal1+=coilPlusConnectorWires(i,numCoils,inc,innerRadius,connectorRadius,size,gap,nTurns,xyz);
    metal2+=coilPlusConnectorWires(i+1,numCoils,inc,innerRadius,connectorRadius,size,gap,nTurns,xyz);
  }
  metal1=clamp(metal1,0.,1.);
  metal2=clamp(metal2,0.,1.);
  
  float dielectric=0.;
  float dielectricRadius=2.*float(numPairs)*(size+gap)+innerRadius+gap;
  float dielectricHeight=(nTurns+4.)*(size+gap);
  dielectric+=cylinder(dielectricRadius,dielectricHeight,xyz+vec3(0,0,2));
  
  float spindleRadius=2.*float(numPairs+1)*(size+gap)+innerRadius;
  dielectric+=cylinder(spindleRadius,2.*(size+gap),xyz+vec3(0,0,2));
  dielectric+=cylinder(spindleRadius,2.*(size+gap),xyz-vec3(0,0,dielectricHeight-4.));
  dielectric=clamp(dielectric,0.,1.);
  // This next step is important... we don't want any dielectric material wherever
  // the metal is located, so we just subtract the metal out of the dielectric.
  dielectric-=clamp(metal1+metal2,0.,1.);
  
  return vec3(metal1,metal2,dielectric);
}

const float innerRadius=3.;
const float wireSize=.85;// mm
const float wireGap=.15;// mm
const int turnsPerCoil=122;
const int numCoilPairs=10;

void mainModel4(out vec4 materials,in vec3 xyz){
  const float wireSpacing=wireSize+wireGap;
  xyz.z+=float(turnsPerCoil)*wireSpacing/2.;
  materials.xyz=bifilarElectromagnet(numCoilPairs,innerRadius,wireSize,wireGap,turnsPerCoil,xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-63],
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

fn rotZ(angle: f32) -> mat4x4f {
  let m = rotAxis(vec3f(0, 0, 1), angle);
  return mat4x4f(
    vec4f(m[0], 0.0),
    vec4f(m[1], 0.0),
    vec4f(m[2], 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0)
  );
}

fn box_prim(start: vec3f, end: vec3f, size: f32, xyz: vec3f) -> f32 {
  let ll = min(start, end) - vec3f(0.5 * size);
  let ur = max(start, end) + vec3f(0.5 * size);
  if (any(xyz < ll) || any(xyz > ur)) { return 0.0; }
  return 1.0;
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  return 1.0;
}

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: f32, trimEndAngle: f32, xyz: vec3f) -> f32 {
  if (xyz.z < -0.5 * size || xyz.z > nTurns * (size + gap) + 0.5 * size) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; }
  let dz = xyz.z - (size + gap) * floor(xyz.z / (size + gap));
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap);
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  if (coilNum < 0.0 || coilNum >= nTurns) { return 0.0; }
  return 1.0;
}

fn coilPlusConnectorWires(coilNum: i32, numCoils: i32, inc: f32, innerRadius: f32, connectorRadius: f32, size: f32, gap: f32, nTurns: f32, xyz_in: vec3f) -> f32 {
  let radiusOffset = f32(coilNum - 1);
  let xfm = rotZ(radiusOffset * inc);
  let coilRadius = radiusOffset + innerRadius;
  var trimEndAngle = 2.0 * inc;
  if (coilNum == numCoils) {
    trimEndAngle = 0.5 * inc;
  } else if (coilNum == numCoils - 1) {
    trimEndAngle = 3.0 * inc;
  }
  let xyz = (xfm * vec4f(xyz_in, 1.0)).xyz;
  var coil = coilSquareFace(coilRadius, size, gap, nTurns, trimEndAngle, xyz);
  
  let bz = -(size + gap);
  let tz = nTurns * (size + gap);
  let tzp1 = (nTurns + 1.0) * (size + gap);
  
  coil += box_prim(vec3f(coilRadius, 0.0, 0.0), vec3f(coilRadius, 0.0, bz), size, xyz);
  coil += box_prim(vec3f(coilRadius, 0.0, bz), vec3f(connectorRadius, 0.0, bz), size, xyz);
  coil += box_prim(vec3f(connectorRadius, 0.0, bz), vec3f(connectorRadius, 0.0, tzp1), size, xyz);
  
  let zexit = (nTurns + 10.0) * (size + gap);
  if (coilNum >= 3) {
    let lastCoilRadius = radiusOffset - 2.0 + innerRadius;
    coil += box_prim(vec3f(lastCoilRadius, 0.0, tzp1), vec3f(connectorRadius, 0.0, tzp1), size, xyz);
    coil += box_prim(vec3f(lastCoilRadius, 0.0, tz), vec3f(lastCoilRadius, 0.0, tzp1), size, xyz);
  } else if (coilNum == 2) {
    let endOddRadius = f32(numCoils - 2) + innerRadius;
    coil += box_prim(vec3f(endOddRadius, 0.0, tzp1), vec3f(connectorRadius, 0.0, tzp1), size, xyz);
    coil += box_prim(vec3f(endOddRadius, 0.0, tz), vec3f(endOddRadius, 0.0, tzp1), size, xyz);
  } else if (coilNum == 1) {
    coil += box_prim(vec3f(connectorRadius, 0.0, tz), vec3f(connectorRadius, 0.0, zexit), size, xyz);
  }
  
  if (coilNum == numCoils) {
    let xfm2 = rotZ(0.5 * inc);
    let xyz2 = (xfm2 * vec4f(xyz, 1.0)).xyz;
    coil += box_prim(vec3f(connectorRadius - (size + gap), 0.0, tz), vec3f(connectorRadius - (size + gap), 0.0, zexit), size, xyz2);
  }
  
  return coil;
}

fn bifilarElectromagnet(numPairs: i32, innerRadius: f32, size: f32, gap: f32, numTurns: i32, xyz: vec3f) -> vec3f {
  let nTurns = f32(numTurns);
  let numCoils = 2*numPairs;
  let inc = 2.0 * M_PI / f32(numCoils);
  let connectorRadius = innerRadius + f32(numCoils) * (size + gap);
  
  var metal1 = 0.0;
  var metal2 = 0.0;
  
  for(var i = 1; i <= numCoils; i += 2) {
    metal1 += coilPlusConnectorWires(i, numCoils, inc, innerRadius, connectorRadius, size, gap, nTurns, xyz);
    metal2 += coilPlusConnectorWires(i + 1, numCoils, inc, innerRadius, connectorRadius, size, gap, nTurns, xyz);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2, 0.0, 1.0);
  
  var dielectric = 0.0;
  let dielectricRadius = 2.0 * f32(numPairs) * (size + gap) + innerRadius + gap;
  let dielectricHeight = (nTurns + 4.0) * (size + gap);
  dielectric += cylinder(dielectricRadius, dielectricHeight, xyz + vec3f(0, 0, 2));
  
  let spindleRadius = 2.0 * f32(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3f(0, 0, 2));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3f(0, 0, dielectricHeight - 4.0));
  dielectric = clamp(dielectric, 0.0, 1.0);
  dielectric -= clamp(metal1 + metal2, 0.0, 1.0);
  
  return vec3f(metal1, metal2, clamp(dielectric, 0.0, 1.0));
}

const innerRadius_val = 3.0;
const wireSize_val = 0.85; // mm
const wireGap_val = 0.15; // mm
const turnsPerCoil_val = 122;
const numCoilPairs_val = 10;

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  let wireSpacing = wireSize_val + wireGap_val;
  xyz.z += f32(turnsPerCoil_val) * wireSpacing / 2.0;
  let res = bifilarElectromagnet(numCoilPairs_val, innerRadius_val, wireSize_val, wireGap_val, turnsPerCoil_val, xyz);
  return vec4f(res, 0.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-63],
  units: "mm",
}*/


const rotZ = (p, a) => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return vec3(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)), p.z);
};

const box = (sx, sy, sz, ex, ey, ez, size, p) => {
  const h = 0.5 * size;
  const llx = Math.min(sx, ex) - h, lly = Math.min(sy, ey) - h, llz = Math.min(sz, ez) - h;
  const urx = Math.max(sx, ex) + h, ury = Math.max(sy, ey) + h, urz = Math.max(sz, ez) + h;
  const outside = p.x.lessThan(llx).or(p.x.greaterThan(urx))
    .or(p.y.lessThan(lly)).or(p.y.greaterThan(ury))
    .or(p.z.lessThan(llz)).or(p.z.greaterThan(urz));
  return select(outside, float(0.0), float(1.0));
};

const cylinder = (radius, height, p) => {
  const zOK = p.z.greaterThanEqual(0.0).and(p.z.lessThanEqual(height));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const coilSquareFace = (radius, size, gap, nTurns, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(
    p.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );
  const rxy = length(p.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );
  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dz = mod(p.z, size + gap);
  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dz);
  const lastHelixZ = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextHelixZ = lastHelixZ.add(size + gap);
  const inGap = dz.greaterThan(lastHelixZ.add(0.5 * size)).and(
    dz.lessThan(nextHelixZ.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(floor(p.z.add(0.5 * size).sub(lastHelixZ).div(size + gap)));
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  // TODO: the GLSL source ignores trimEndAngle here.
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), float(1.0), float(0.0));
};

const coilPlusConnectorWires = (coilNum, numCoils, inc, innerRadius, connectorRadius, size, gap, nTurns, p0) => {
  const radiusOffset = coilNum - 1;
  const p = rotZ(p0, radiusOffset * inc);
  const coilRadius = radiusOffset + innerRadius;
  let coil = coilSquareFace(coilRadius, size, gap, nTurns, p);

  const bz = -(size + gap);
  const tz = nTurns * (size + gap);
  const tzp1 = (nTurns + 1.0) * (size + gap);
  const add = (v) => { coil = coil.add(v); };

  add(box(coilRadius, 0, 0, coilRadius, 0, bz, size, p));
  add(box(coilRadius, 0, bz, connectorRadius, 0, bz, size, p));
  add(box(connectorRadius, 0, bz, connectorRadius, 0, tzp1, size, p));

  const zexit = (nTurns + 10.0) * (size + gap);
  if (coilNum >= 3) {
    // Connect the start of this coil to the end of two coils prior.
    const lastCoilRadius = radiusOffset - 2 + innerRadius;
    add(box(lastCoilRadius, 0, tzp1, connectorRadius, 0, tzp1, size, p));
    add(box(lastCoilRadius, 0, tz, lastCoilRadius, 0, tzp1, size, p));
  } else if (coilNum === 2) {
    // Connect the start of 2 to the end of the last odd coil.
    const endOddRadius = numCoils - 2 + innerRadius;
    add(box(endOddRadius, 0, tzp1, connectorRadius, 0, tzp1, size, p));
    add(box(endOddRadius, 0, tz, endOddRadius, 0, tzp1, size, p));
  } else if (coilNum === 1) {
    // Bring out the exit wires.
    add(box(connectorRadius, 0, tz, connectorRadius, 0, zexit, size, p));
  }

  if (coilNum === numCoils) {
    const ps = rotZ(p, 0.5 * inc);
    add(box(connectorRadius - (size + gap), 0, tz, connectorRadius - (size + gap), 0, zexit, size, ps));
  }
  return coil;
};

const bifilarElectromagnet = (numPairs, innerRadius, size, gap, numTurns, p) => {
  const nTurns = numTurns;
  const numCoils = 2 * numPairs;
  const inc = (2.0 * Math.PI) / numCoils;
  const connectorRadius = innerRadius + numCoils * (size + gap);

  let metal1 = float(0.0);
  let metal2 = float(0.0);
  for (let i = 1; i <= numCoils; i += 2) {
    metal1 = metal1.add(
      coilPlusConnectorWires(i, numCoils, inc, innerRadius, connectorRadius, size, gap, nTurns, p)
    );
    metal2 = metal2.add(
      coilPlusConnectorWires(i + 1, numCoils, inc, innerRadius, connectorRadius, size, gap, nTurns, p)
    );
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2, 0.0, 1.0);

  const dielectricRadius = 2.0 * numPairs * (size + gap) + innerRadius + gap;
  const dielectricHeight = (nTurns + 4.0) * (size + gap);
  let dielectric = cylinder(dielectricRadius, dielectricHeight, p.add(vec3(0, 0, 2)));
  const spindleRadius = 2.0 * (numPairs + 1) * (size + gap) + innerRadius;
  dielectric = dielectric.add(cylinder(spindleRadius, 2.0 * (size + gap), p.add(vec3(0, 0, 2))));
  dielectric = dielectric.add(
    cylinder(spindleRadius, 2.0 * (size + gap), p.sub(vec3(0, 0, dielectricHeight - 4.0)))
  );
  dielectric = clamp(dielectric, 0.0, 1.0);
  // Don't put dielectric where the metal is.
  dielectric = dielectric.sub(clamp(metal1.add(metal2), 0.0, 1.0));
  return [metal1, metal2, dielectric];
};

const mainModel4 = (xyz) => {
  const wireSize = 0.85;
  const wireGap = 0.15;
  const turnsPerCoil = 122;
  const wireSpacing = wireSize + wireGap;
  const p = xyz.add(vec3(0, 0, 0.5 * turnsPerCoil * wireSpacing));
  return bifilarElectromagnet(10, 3.0, wireSize, wireGap, turnsPerCoil, p);
};
```

* Try loading [bifilar-electromagnet-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/bifilar-electromagnet-1.irmf) now in the IRMF editor!
* Try loading [bifilar-electromagnet-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/bifilar-electromagnet-1-wgsl.irmf) now in the IRMF editor!
* Try loading [bifilar-electromagnet-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/bifilar-electromagnet-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## axial-radial-bifilar-electromagnet-1.irmf

It turns out that the `bifilar-electromagnet-1.irmf` design above is strictly
bifilar in the radial direction, meaning that the capacitance of the coils
is only radial.

If the coils are wound such that the first and second coils share the same
radius but are wound next to each other (180-degrees out of phase), and then
the next pair of windings is completely rotated 180 degrees, then the
capacitance between the coils will be in both the radial direction _and_ the
axial direction!

![axial-radial-bifilar-electromagnet-1.png](axial-radial-bifilar-electromagnet-1.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper","copper"],
  max: [13,13,15],
  min: [-13,-13,-3],
  units: "mm",
}*/

#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/rotation.glsl"
#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/primitives.glsl"

#define M_PI 3.1415926535897932384626433832795

float coilSquareFace(in float radius, in float size, in float gap, in int nTurns, in float trimStartAngle, in float trimEndAngle, in vec3 xyz) {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > float(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  float rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  
  // If the current point is between the coils, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  float dz = mod(xyz.z, size + gap);
  
  float lastHelixZ = angle * (size + gap);
  float coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap); // center of current coil.
    coilNum = -1.0;
  }
  float nextHelixZ = lastHelixZ + (size + gap); // center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  
  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= float(nTurns)) { return 0.0; }
  
  // For the last two checks, convert angle back to radians.
  angle *= (2.0 * M_PI);
  
  // If the current point is in the first coil, use the trimStartAngle.
  if (coilNum < 1.0) { return angle >= trimStartAngle ? 1.0 : 0.0; }
  
  // If the current point is in the last coil, use the trimEndAngle.
  if (coilNum >= float(nTurns) - 1.0 && trimEndAngle > 0.0) { return angle <= trimEndAngle ? 1.0 : 0.0; }
  
  return 1.0;
}

float coilPlusConnectorWires(in int wireNum, in int coilNum, in int numCoils, in float inc, in float innerRadius, in float connectorRadius, in float size, in float singleGap, in int nTurns, in vec3 xyz) {
  float gap = size + 2.0 * singleGap;
  float radiusOffset = float(coilNum - 1);
  mat4 xfm = mat4(1) * rotZ(radiusOffset * inc);
  float coilRadius = radiusOffset + innerRadius;
  float trimStartAngle = inc - 0.1; // This is a hack that works for numCoils=10 and may need adjusting for other values.
  
  xyz = (vec4(xyz, 1.0) * xfm).xyz;
  float coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);
  
  float tz = float(nTurns) * (size + gap);
  float zexit = (float(nTurns) + 10.0) * (size + gap);
  
  if (coilNum == numCoils) {
    // Special case to access the exit wires.
    if (wireNum == 1) {
      coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, zexit), size, xyz);
    } else {
      coil += box(vec3(coilRadius, 0, tz), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(innerRadius, 0, - size), size, xyz);
      coil += box(vec3(innerRadius, 0, - size), vec3(innerRadius, 0, 0), size, xyz);
      // Now the other exit wire
      coil += box(vec3(-innerRadius, 0, - size), vec3(-innerRadius, 0, 0), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-innerRadius, 0, - size), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-connectorRadius, 0, zexit), size, xyz);
    }
  } else {
    coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, tz + size), size, xyz);
    coil += box(vec3(coilRadius, 0, tz + size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box(vec3(coilRadius, 0, - size), vec3(connectorRadius, 0, - size), size, xyz);
    coil += box(vec3(coilRadius, 0, - size), vec3(coilRadius, 0, 0), size, xyz);
  }
  
  return clamp(coil, 0.0, 1.0);
}

vec3 arBifilarElectromagnet(int numPairs, float innerRadius, float size, float gap, int numTurns, in vec3 xyz) {
  float nTurns = float(numTurns);
  float inc = M_PI / float(numPairs);
  float connectorRadius = innerRadius + float(numPairs) * (size + gap);
  
  float metal1 = 0.0;
  float metal2 = 0.0;
  
  mat4 xfm = mat4(1) * rotZ(M_PI);
  vec3 xyz180Z = (vec4(xyz, 1.0) * xfm).xyz;
  
  for(int i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0); // Don't double the metals on overlap.
  
  float dielectric = 0.0;
  float dielectricRadius = 2.0 * float(numPairs) * (size + gap) + innerRadius + gap;
  float dielectricHeight = (nTurns + 4.0) * (size + gap);
  dielectric += cylinder(dielectricRadius, dielectricHeight, xyz + vec3(0, 0, 2));
  
  float spindleRadius = 2.0 * float(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3(0, 0, 2));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3(0, 0, dielectricHeight - 4.0));
  dielectric = clamp(dielectric, 0.0, 1.0);
  // This next step is important... we don't want any dielectric material wherever
  // the metal is located, so we just subtract the metal out of the dielectric.
  dielectric -= clamp(metal1 + metal2, 0.0, 1.0);
  
  return vec3(metal1, metal2, dielectric);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // xyz.z+=60.;
  materials.xyz = arBifilarElectromagnet(10, 3.0, 0.85, 0.15, 2, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper","copper"],
  max: [13,13,15],
  min: [-13,-13,-3],
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

fn rotZ(angle: f32) -> mat4x4f {
  return mat3ToMat4(rotAxis(vec3f(0.0, 0.0, 1.0), angle));
}

fn box_prim(start: vec3f, end: vec3f, size: f32, xyz: vec3f) -> f32 {
  let ll = min(start, end) - vec3f(0.5 * size);
  let ur = max(start, end) + vec3f(0.5 * size);
  if (any(xyz < ll) || any(xyz > ur)) { return 0.0; }
  return 1.0;
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  return 1.0;
}

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: i32, trimStartAngle: f32, trimEndAngle: f32, xyz: vec3f) -> f32 {
  if (xyz.z < -0.5 * size || xyz.z > f32(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; }
  let dz = wgsl_mod(xyz.z, size + gap);
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap);
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  if (coilNum < 0.0 || coilNum >= f32(nTurns)) { return 0.0; }
  let rad_angle = angle * (2.0 * M_PI);
  if (coilNum < 1.0) { return select(0.0, 1.0, rad_angle >= trimStartAngle); }
  if (coilNum >= f32(nTurns) - 1.0 && trimEndAngle > 0.0) { return select(0.0, 1.0, rad_angle <= trimEndAngle); }
  return 1.0;
}

fn coilPlusConnectorWires(wireNum: i32, coilNum: i32, numCoils: i32, inc: f32, innerRadius: f32, connectorRadius: f32, size: f32, singleGap: f32, nTurns: i32, xyz_orig: vec3f) -> f32 {
  let gap = size + 2.0 * singleGap;
  let radiusOffset = f32(coilNum - 1);
  let xfm = rotZ(radiusOffset * inc);
  var coilRadius = radiusOffset + innerRadius;
  let trimStartAngle = inc - 0.1;
  let xyz = (xfm * vec4f(xyz_orig, 1.0)).xyz;
  var coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);
  let tz = f32(nTurns) * (size + gap);
  let zexit = (f32(nTurns) + 10.0) * (size + gap);
  if (coilNum == numCoils) {
    if (wireNum == 1) {
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, zexit), size, xyz);
    } else {
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(innerRadius, 0.0, - size), vec3f(innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-innerRadius, 0.0, - size), vec3f(-innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-connectorRadius, 0.0, zexit), size, xyz);
    }
  } else {
    coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, tz + size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(connectorRadius, 0.0, - size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(coilRadius, 0.0, 0.0), size, xyz);
  }
  return clamp(coil, 0.0, 1.0);
}

fn arBifilarElectromagnet(numPairs: i32, innerRadius: f32, size: f32, gap: f32, numTurns: i32, xyz: vec3f) -> vec3f {
  let nTurns = f32(numTurns);
  let inc = M_PI / f32(numPairs);
  let connectorRadius = innerRadius + f32(numPairs) * (size + gap);
  var metal1 = 0.0;
  var metal2 = 0.0;
  let xfm = rotZ(M_PI);
  let xyz180Z = (xfm * vec4f(xyz, 1.0)).xyz;
  for(var i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0);
  var dielectric = 0.0;
  let dielectricRadius = 2.0 * f32(numPairs) * (size + gap) + innerRadius + gap;
  let dielectricHeight = (nTurns + 4.0) * (size + gap);
  dielectric += cylinder(dielectricRadius, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  let spindleRadius = 2.0 * f32(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3f(0.0, 0.0, 2.0));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3f(0.0, 0.0, dielectricHeight - 4.0));
  dielectric = clamp(dielectric, 0.0, 1.0);
  dielectric -= clamp(metal1 + metal2, 0.0, 1.0);
  return vec3f(metal1, metal2, clamp(dielectric, 0.0, 1.0));
}

fn mainModel4(xyz: vec3f) -> vec4f {
  let res = arBifilarElectromagnet(10, 3.0, 0.85, 0.15, 2, xyz);
  return vec4f(res, 0.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper","copper"],
  max: [13,13,15],
  min: [-13,-13,-3],
  units: "mm",
}*/


const rotZ = (p, a) => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return vec3(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)), p.z);
};

const box = (sx, sy, sz, ex, ey, ez, size, p) => {
  const h = 0.5 * size;
  const llx = Math.min(sx, ex) - h, lly = Math.min(sy, ey) - h, llz = Math.min(sz, ez) - h;
  const urx = Math.max(sx, ex) + h, ury = Math.max(sy, ey) + h, urz = Math.max(sz, ez) + h;
  const outside = p.x.lessThan(llx).or(p.x.greaterThan(urx))
    .or(p.y.lessThan(lly)).or(p.y.greaterThan(ury))
    .or(p.z.lessThan(llz)).or(p.z.greaterThan(urz));
  return select(outside, float(0.0), float(1.0));
};

const cylinder = (radius, height, p) => {
  const zOK = p.z.greaterThanEqual(0.0).and(p.z.lessThanEqual(height));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const coilSquareFace = (radius, size, gap, nTurns, trimStartAngle, trimEndAngle, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(
    p.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );
  const rxy = length(p.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );
  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dz = mod(p.z, size + gap);
  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dz);
  const lastHelixZ = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextHelixZ = lastHelixZ.add(size + gap);
  const inGap = dz.greaterThan(lastHelixZ.add(0.5 * size)).and(
    dz.lessThan(nextHelixZ.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(floor(p.z.add(0.5 * size).sub(lastHelixZ).div(size + gap)));
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  const angleRad = angle.mul(2.0 * Math.PI);
  const startTrim = select(angleRad.greaterThanEqual(trimStartAngle), float(1.0), float(0.0));
  let trim = select(coilNum.lessThan(1.0), startTrim, float(1.0));
  if (trimEndAngle > 0.0) {
    const endTrim = select(angleRad.lessThanEqual(trimEndAngle), float(1.0), float(0.0));
    trim = select(coilNum.lessThan(1.0), startTrim,
      select(coilNum.greaterThanEqual(nTurns - 1.0), endTrim, float(1.0)));
  }
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), trim, float(0.0));
};

const coilPlusConnectorWires = (wireNum, coilNum, numCoils, inc, innerRadius, connectorRadius, size, singleGap, nTurns, p0) => {
  const gap = size + 2.0 * singleGap;
  const radiusOffset = coilNum - 1;
  const spacingAngle = (numCoils - 4) * inc * Math.atan((2.0 * radiusOffset) / (numCoils - 1));
  const p = rotZ(p0, spacingAngle);
  const coilRadius = radiusOffset + innerRadius;
  const trimStartAngle = 0.05;
  let coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, p);
  const tz = nTurns * (size + gap);
  const zexit = (nTurns + 10.0) * (size + gap);
  const add = (v) => { coil = coil.add(v); };

  if (coilNum === numCoils) {
    if (wireNum === 1) {
      const ps = rotZ(p, 0.45);
      add(box(coilRadius, 0, tz, coilRadius, 0, zexit, size, ps));
    } else {
      add(box(coilRadius, 0, tz, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, innerRadius, 0, -size, size, p));
      add(box(innerRadius, 0, -size, innerRadius, 0, 0, size, p));
      add(box(-innerRadius, 0, -size, -innerRadius, 0, 0, size, p));
      add(box(-connectorRadius, 0, -size, -innerRadius, 0, -size, size, p));
      add(box(-connectorRadius, 0, -size, -connectorRadius, 0, zexit, size, p));
    }
  } else {
    add(box(coilRadius, 0, tz, coilRadius, 0, tz + size, size, p));
    add(box(coilRadius, 0, tz + size, connectorRadius, 0, tz + size, size, p));
    add(box(connectorRadius, 0, -size, connectorRadius, 0, tz + size, size, p));
    const coilRadius2 = coilRadius + size + singleGap;
    add(box(coilRadius2, 0, -size, connectorRadius, 0, -size, size, p));
    add(box(coilRadius2, 0, -size, coilRadius2, 0, 0, size, p));
  }
  return clamp(coil, 0.0, 1.0);
};

const arBifilarElectromagnet = (numPairs, innerRadius, size, gap, numTurns, p) => {
  const nTurns = numTurns;
  const inc = Math.PI / numPairs;
  const connectorRadius = innerRadius + numPairs * (size + gap);
  let metal1 = float(0.0);
  let metal2 = float(0.0);
  const p180 = rotZ(p, Math.PI);
  for (let i = 1; i <= numPairs; i++) {
    metal1 = metal1.add(
      coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p)
    );
    metal2 = metal2.add(
      coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p180)
    );
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2.sub(metal1), 0.0, 1.0);
  const dielectricRadius = numPairs * (size + gap) + innerRadius + gap;
  const dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  let dielectric = cylinder(dielectricRadius, dielectricHeight, p.add(vec3(0, 0, 2)));
  const spindleRadius = (numPairs + 1) * (size + gap) + innerRadius;
  dielectric = dielectric.add(cylinder(spindleRadius, 2.0 * (size + gap), p.add(vec3(0, 0, 2))));
  dielectric = dielectric.add(
    cylinder(spindleRadius, 2.0 * (size + gap), p.sub(vec3(0, 0, dielectricHeight - 4.0)))
  );
  dielectric = dielectric.sub(
    cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, p.add(vec3(0, 0, 2))).mul(2.0)
  );
  dielectric = clamp(dielectric.sub(metal1).sub(metal2), 0.0, 1.0);
  return [metal1, metal2, dielectric];
};

const mainModel4 = (xyz) => {
  const shifted = xyz;
  const r = arBifilarElectromagnet(10, 3.0, 0.85, 0.15, 2, shifted);
  return [r[0], r[1]];
};
```

* Try loading [axial-radial-bifilar-electromagnet-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial-radial-bifilar-electromagnet-1.irmf) now in the IRMF editor!
* Try loading [axial-radial-bifilar-electromagnet-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial-radial-bifilar-electromagnet-1-wgsl.irmf) now in the IRMF editor!
* Try loading [axial-radial-bifilar-electromagnet-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial-radial-bifilar-electromagnet-1-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## full-coil-metal-only.irmf

Here is the full model only showing the metal:

![full-coil-metal-only.png](full-coil-metal-only.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper","copper"],
  max: [25,25,71],
  min: [-25,-25,-61],
  units: "mm",
}*/

#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/rotation.glsl"
#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/primitives.glsl"

#define M_PI 3.1415926535897932384626433832795

float coilSquareFace(in float radius, in float size, in float gap, in int nTurns, in float trimStartAngle, in float trimEndAngle, in vec3 xyz) {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > float(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  float rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  
  // If the current point is between the coils, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  float dz = mod(xyz.z, size + gap);
  
  float lastHelixZ = angle * (size + gap);
  float coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap); // center of current coil.
    coilNum = -1.0;
  }
  float nextHelixZ = lastHelixZ + (size + gap); // center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  
  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= float(nTurns)) { return 0.0; }
  
  // For the last two checks, convert angle back to radians.
  angle *= (2.0 * M_PI);
  
  // If the current point is in the first coil, use the trimStartAngle.
  if (coilNum < 1.0) { return angle >= trimStartAngle ? 1.0 : 0.0; }
  
  // If the current point is in the last coil, use the trimEndAngle.
  if (coilNum >= float(nTurns) - 1.0 && trimEndAngle > 0.0) { return angle <= trimEndAngle ? 1.0 : 0.0; }
  
  return 1.0;
}

float coilPlusConnectorWires(in int wireNum, in int coilNum, in int numCoils, in float inc, in float innerRadius, in float connectorRadius, in float size, in float singleGap, in int nTurns, in vec3 xyz) {
  float gap = size + 2.0 * singleGap;
  float radiusOffset = float(coilNum - 1);
  // next line is experimental - works for numPairs=20:
  float spacingAngle = float(numCoils - 4) * inc * atan(2.0 * radiusOffset / float(numCoils - 1));
  mat4 xfm = mat4(1) * rotZ(spacingAngle);
  float coilRadius = radiusOffset + innerRadius;
  float trimStartAngle = 0.05; // experimental - works somewhat for numPairs=20, but not ideal.
  
  xyz = (vec4(xyz, 1.0) * xfm).xyz;
  float coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);
  
  float tz = float(nTurns) * (size + gap);
  float zexit = (float(nTurns) + 10.0) * (size + gap);
  
  if (coilNum == numCoils) {
    // Special case to access the exit wires.
    if (wireNum == 1) {
      coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, zexit), size, xyz);
    } else {
      coil += box(vec3(coilRadius, 0, tz), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(innerRadius, 0, - size), size, xyz);
      coil += box(vec3(innerRadius, 0, - size), vec3(innerRadius, 0, 0), size, xyz);
      // Now the other exit wire
      coil += box(vec3(-innerRadius, 0, - size), vec3(-innerRadius, 0, 0), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-innerRadius, 0, - size), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-connectorRadius, 0, zexit), size, xyz);
    }
  } else {
    coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, tz + size), size, xyz);
    coil += box(vec3(coilRadius, 0, tz + size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box(vec3(coilRadius, 0, - size), vec3(connectorRadius, 0, - size), size, xyz);
    coil += box(vec3(coilRadius, 0, - size), vec3(coilRadius, 0, 0), size, xyz);
  }
  
  return clamp(coil, 0.0, 1.0);
}

vec3 arBifilarElectromagnet(int numPairs, float innerRadius, float size, float gap, int numTurns, in vec3 xyz) {
  float nTurns = float(numTurns);
  float inc = M_PI / float(numPairs);
  float connectorRadius = innerRadius + float(numPairs) * (size + gap);
  
  float metal1 = 0.0;
  float metal2 = 0.0;
  
  mat4 xfm = mat4(1) * rotZ(M_PI);
  vec3 xyz180Z = (vec4(xyz, 1.0) * xfm).xyz;
  
  for(int i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0); // Don't double the metals on overlap.
  
  float dielectricRadius = float(numPairs) * (size + gap) + innerRadius + gap;
  float dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  float dielectric = cylinder(dielectricRadius, dielectricHeight, xyz + vec3(0, 0, 2));
  
  float spindleRadius = float(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3(0, 0, 2));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3(0, 0, dielectricHeight - 4.0));
  dielectric -= 2.0 * cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, xyz + vec3(0, 0, 2));
  // This next step is important... we don't want any dielectric material wherever
  // the metal is located, so we just subtract the metal out of the dielectric.
  dielectric = clamp(dielectric - metal1 - metal2, 0.0, 1.0);
  
  return vec3(metal1, metal2, dielectric);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 60.0;
  materials.xyz = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper","copper"],
  max: [25,25,71],
  min: [-25,-25,-61],
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

fn rotZ(angle: f32) -> mat4x4f {
  return mat3ToMat4(rotAxis(vec3f(0.0, 0.0, 1.0), angle));
}

fn box_prim(start: vec3f, end: vec3f, size: f32, xyz: vec3f) -> f32 {
  let ll = min(start, end) - vec3f(0.5 * size);
  let ur = max(start, end) + vec3f(0.5 * size);
  if (any(xyz < ll) || any(xyz > ur)) { return 0.0; }
  return 1.0;
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  return 1.0;
}

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: i32, trimStartAngle: f32, trimEndAngle: f32, xyz: vec3f) -> f32 {
  if (xyz.z < -0.5 * size || xyz.z > f32(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; }
  let dz = wgsl_mod(xyz.z, size + gap);
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap);
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  if (coilNum < 0.0 || coilNum >= f32(nTurns)) { return 0.0; }
  let rad_angle = angle * (2.0 * M_PI);
  if (coilNum < 1.0) { return select(0.0, 1.0, rad_angle >= trimStartAngle); }
  if (coilNum >= f32(nTurns) - 1.0 && trimEndAngle > 0.0) { return select(0.0, 1.0, rad_angle <= trimEndAngle); }
  return 1.0;
}

fn coilPlusConnectorWires(wireNum: i32, coilNum: i32, numCoils: i32, inc: f32, innerRadius: f32, connectorRadius: f32, size: f32, singleGap: f32, nTurns: i32, xyz_orig: vec3f) -> f32 {
  let gap = size + 2.0 * singleGap;
  let radiusOffset = f32(coilNum - 1);
  let spacingAngle = f32(numCoils - 4) * inc * atan(2.0 * radiusOffset / f32(numCoils - 1));
  let xfm = rotZ(spacingAngle);
  var coilRadius = radiusOffset + innerRadius;
  let trimStartAngle = 0.05;
  var xyz = (xfm * vec4f(xyz_orig, 1.0)).xyz;
  var coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);
  let tz = f32(nTurns) * (size + gap);
  let zexit = (f32(nTurns) + 10.0) * (size + gap);
  if (coilNum == numCoils) {
    if (wireNum == 1) {
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, zexit), size, xyz);
    } else {
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(innerRadius, 0.0, - size), vec3f(innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-innerRadius, 0.0, - size), vec3f(-innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-connectorRadius, 0.0, zexit), size, xyz);
    }
  } else {
    coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, tz + size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(connectorRadius, 0.0, - size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(coilRadius, 0.0, 0.0), size, xyz);
  }
  return clamp(coil, 0.0, 1.0);
}

fn arBifilarElectromagnet(numPairs: i32, innerRadius: f32, size: f32, gap: f32, numTurns: i32, xyz: vec3f) -> vec3f {
  let nTurns = f32(numTurns);
  let inc = M_PI / f32(numPairs);
  let connectorRadius = innerRadius + f32(numPairs) * (size + gap);
  var metal1 = 0.0;
  var metal2 = 0.0;
  let xfm = rotZ(M_PI);
  let xyz180Z = (xfm * vec4f(xyz, 1.0)).xyz;
  for(var i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0);
  let dielectricRadius = f32(numPairs) * (size + gap) + innerRadius + gap;
  let dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  var dielectric = cylinder(dielectricRadius, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  let spindleRadius = f32(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3f(0.0, 0.0, 2.0));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3f(0.0, 0.0, dielectricHeight - 4.0));
  dielectric -= 2.0 * cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  dielectric = clamp(dielectric - metal1 - metal2, 0.0, 1.0);
  return vec3f(metal1, metal2, dielectric);
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 60.0;
  let res = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, xyz);
  return vec4f(res, 0.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper","copper"],
  max: [25,25,71],
  min: [-25,-25,-61],
  units: "mm",
}*/


const rotZ = (p, a) => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return vec3(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)), p.z);
};

const box = (sx, sy, sz, ex, ey, ez, size, p) => {
  const h = 0.5 * size;
  const llx = Math.min(sx, ex) - h, lly = Math.min(sy, ey) - h, llz = Math.min(sz, ez) - h;
  const urx = Math.max(sx, ex) + h, ury = Math.max(sy, ey) + h, urz = Math.max(sz, ez) + h;
  const outside = p.x.lessThan(llx).or(p.x.greaterThan(urx))
    .or(p.y.lessThan(lly)).or(p.y.greaterThan(ury))
    .or(p.z.lessThan(llz)).or(p.z.greaterThan(urz));
  return select(outside, float(0.0), float(1.0));
};

const cylinder = (radius, height, p) => {
  const zOK = p.z.greaterThanEqual(0.0).and(p.z.lessThanEqual(height));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const coilSquareFace = (radius, size, gap, nTurns, trimStartAngle, trimEndAngle, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(
    p.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );
  const rxy = length(p.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );
  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dz = mod(p.z, size + gap);
  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dz);
  const lastHelixZ = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextHelixZ = lastHelixZ.add(size + gap);
  const inGap = dz.greaterThan(lastHelixZ.add(0.5 * size)).and(
    dz.lessThan(nextHelixZ.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(floor(p.z.add(0.5 * size).sub(lastHelixZ).div(size + gap)));
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  const angleRad = angle.mul(2.0 * Math.PI);
  const startTrim = select(angleRad.greaterThanEqual(trimStartAngle), float(1.0), float(0.0));
  let trim = select(coilNum.lessThan(1.0), startTrim, float(1.0));
  if (trimEndAngle > 0.0) {
    const endTrim = select(angleRad.lessThanEqual(trimEndAngle), float(1.0), float(0.0));
    trim = select(coilNum.lessThan(1.0), startTrim,
      select(coilNum.greaterThanEqual(nTurns - 1.0), endTrim, float(1.0)));
  }
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), trim, float(0.0));
};

const coilPlusConnectorWires = (wireNum, coilNum, numCoils, inc, innerRadius, connectorRadius, size, singleGap, nTurns, p0) => {
  const gap = size + 2.0 * singleGap;
  const radiusOffset = coilNum - 1;
  const spacingAngle = (numCoils - 4) * inc * Math.atan((2.0 * radiusOffset) / (numCoils - 1));
  const p = rotZ(p0, spacingAngle);
  const coilRadius = radiusOffset + innerRadius;
  const trimStartAngle = 0.05;
  let coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, p);
  const tz = nTurns * (size + gap);
  const zexit = (nTurns + 10.0) * (size + gap);
  const add = (v) => { coil = coil.add(v); };

  if (coilNum === numCoils) {
    if (wireNum === 1) {
      const ps = rotZ(p, 0.45);
      add(box(coilRadius, 0, tz, coilRadius, 0, zexit, size, ps));
    } else {
      add(box(coilRadius, 0, tz, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, innerRadius, 0, -size, size, p));
      add(box(innerRadius, 0, -size, innerRadius, 0, 0, size, p));
      add(box(-innerRadius, 0, -size, -innerRadius, 0, 0, size, p));
      add(box(-connectorRadius, 0, -size, -innerRadius, 0, -size, size, p));
      add(box(-connectorRadius, 0, -size, -connectorRadius, 0, zexit, size, p));
    }
  } else {
    add(box(coilRadius, 0, tz, coilRadius, 0, tz + size, size, p));
    add(box(coilRadius, 0, tz + size, connectorRadius, 0, tz + size, size, p));
    add(box(connectorRadius, 0, -size, connectorRadius, 0, tz + size, size, p));
    const coilRadius2 = coilRadius + size + singleGap;
    add(box(coilRadius2, 0, -size, connectorRadius, 0, -size, size, p));
    add(box(coilRadius2, 0, -size, coilRadius2, 0, 0, size, p));
  }
  return clamp(coil, 0.0, 1.0);
};

const arBifilarElectromagnet = (numPairs, innerRadius, size, gap, numTurns, p) => {
  const nTurns = numTurns;
  const inc = Math.PI / numPairs;
  const connectorRadius = innerRadius + numPairs * (size + gap);
  let metal1 = float(0.0);
  let metal2 = float(0.0);
  const p180 = rotZ(p, Math.PI);
  for (let i = 1; i <= numPairs; i++) {
    metal1 = metal1.add(
      coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p)
    );
    metal2 = metal2.add(
      coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p180)
    );
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2.sub(metal1), 0.0, 1.0);
  const dielectricRadius = numPairs * (size + gap) + innerRadius + gap;
  const dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  let dielectric = cylinder(dielectricRadius, dielectricHeight, p.add(vec3(0, 0, 2)));
  const spindleRadius = (numPairs + 1) * (size + gap) + innerRadius;
  dielectric = dielectric.add(cylinder(spindleRadius, 2.0 * (size + gap), p.add(vec3(0, 0, 2))));
  dielectric = dielectric.add(
    cylinder(spindleRadius, 2.0 * (size + gap), p.sub(vec3(0, 0, dielectricHeight - 4.0)))
  );
  dielectric = dielectric.sub(
    cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, p.add(vec3(0, 0, 2))).mul(2.0)
  );
  dielectric = clamp(dielectric.sub(metal1).sub(metal2), 0.0, 1.0);
  return [metal1, metal2, dielectric];
};

const mainModel4 = (xyz) => {
  const shifted = xyz.add(vec3(0, 0, 60));
  const r = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, shifted);
  return [r[0], r[1]];
};
```

* Try loading [full-coil-metal-only.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/full-coil-metal-only.irmf) now in the IRMF editor!
* Try loading [full-coil-metal-only-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/full-coil-metal-only-wgsl.irmf) now in the IRMF editor!
* Try loading [full-coil-metal-only-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/full-coil-metal-only-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## full-coil-metal-with-dielectric.irmf

Here is the full model with metal and dielectric:

![full-coil-metal-with-dielectric.png](full-coil-metal-with-dielectric.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-61],
  units: "mm",
}*/

#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/rotation.glsl"
#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/primitives.glsl"

#define M_PI 3.1415926535897932384626433832795

float coilSquareFace(in float radius, in float size, in float gap, in int nTurns, in float trimStartAngle, in float trimEndAngle, in vec3 xyz) {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > float(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  
  // Then, constrain the coil to the cylinder with wall thickness "size":
  float rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  
  // If the current point is between the coils, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  float dz = mod(xyz.z, size + gap);
  
  float lastHelixZ = angle * (size + gap);
  float coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap); // center of current coil.
    coilNum = -1.0;
  }
  float nextHelixZ = lastHelixZ + (size + gap); // center of next higher vertical coil.
  
  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  
  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= float(nTurns)) { return 0.0; }
  
  // For the last two checks, convert angle back to radians.
  angle *= (2.0 * M_PI);
  
  // If the current point is in the first coil, use the trimStartAngle.
  if (coilNum < 1.0) { return angle >= trimStartAngle ? 1.0 : 0.0; }
  
  // If the current point is in the last coil, use the trimEndAngle.
  if (coilNum >= float(nTurns) - 1.0 && trimEndAngle > 0.0) { return angle <= trimEndAngle ? 1.0 : 0.0; }
  
  return 1.0;
}

float coilPlusConnectorWires(in int wireNum, in int coilNum, in int numCoils, in float inc, in float innerRadius, in float connectorRadius, in float size, in float singleGap, in int nTurns, in vec3 xyz) {
  float gap = size + 2.0 * singleGap;
  float radiusOffset = float(coilNum - 1);
  // next line is experimental - works for numPairs=20:
  float spacingAngle = float(numCoils - 4) * inc * atan(2.0 * radiusOffset / float(numCoils - 1));
  mat4 xfm = mat4(1) * rotZ(spacingAngle);
  float coilRadius = radiusOffset + innerRadius;
  float trimStartAngle = 0.05; // experimental - works somewhat for numPairs=20, but not ideal.
  
  xyz = (vec4(xyz, 1.0) * xfm).xyz;
  float coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);
  
  float tz = float(nTurns) * (size + gap);
  float zexit = (float(nTurns) + 10.0) * (size + gap);
  
  if (coilNum == numCoils) {
    // Special case to access the exit wires.
    if (wireNum == 1) {
      coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, zexit), size, xyz);
    } else {
      coil += box(vec3(coilRadius, 0, tz), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(innerRadius, 0, - size), size, xyz);
      coil += box(vec3(innerRadius, 0, - size), vec3(innerRadius, 0, 0), size, xyz);
      // Now the other exit wire
      coil += box(vec3(-innerRadius, 0, - size), vec3(-innerRadius, 0, 0), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-innerRadius, 0, - size), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-connectorRadius, 0, zexit), size, xyz);
    }
  } else {
    coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, tz + size), size, xyz);
    coil += box(vec3(coilRadius, 0, tz + size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box(vec3(coilRadius, 0, - size), vec3(connectorRadius, 0, - size), size, xyz);
    coil += box(vec3(coilRadius, 0, - size), vec3(coilRadius, 0, 0), size, xyz);
  }
  
  return clamp(coil, 0.0, 1.0);
}

vec3 arBifilarElectromagnet(int numPairs, float innerRadius, float size, float gap, int numTurns, in vec3 xyz) {
  float nTurns = float(numTurns);
  float inc = M_PI / float(numPairs);
  float connectorRadius = innerRadius + float(numPairs) * (size + gap);
  
  float metal1 = 0.0;
  float metal2 = 0.0;
  
  mat4 xfm = mat4(1) * rotZ(M_PI);
  vec3 xyz180Z = (vec4(xyz, 1.0) * xfm).xyz;
  
  for(int i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0); // Don't double the metals on overlap.
  
  float dielectricRadius = float(numPairs) * (size + gap) + innerRadius + gap;
  float dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  float dielectric = cylinder(dielectricRadius, dielectricHeight, xyz + vec3(0, 0, 2));
  
  float spindleRadius = float(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3(0, 0, 2));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3(0, 0, dielectricHeight - 4.0));
  dielectric -= 2.0 * cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, xyz + vec3(0, 0, 2));
  // This next step is important... we don't want any dielectric material wherever
  // the metal is located, so we just subtract the metal out of the dielectric.
  dielectric = clamp(dielectric - metal1 - metal2, 0.0, 1.0);
  
  return vec3(metal1, metal2, dielectric);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 60.0;
  materials.xyz = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-61],
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

fn rotZ(angle: f32) -> mat4x4f {
  return mat3ToMat4(rotAxis(vec3f(0.0, 0.0, 1.0), angle));
}

fn box_prim(start: vec3f, end: vec3f, size: f32, xyz: vec3f) -> f32 {
  let ll = min(start, end) - vec3f(0.5 * size);
  let ur = max(start, end) + vec3f(0.5 * size);
  if (any(xyz < ll) || any(xyz > ur)) { return 0.0; }
  return 1.0;
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  return 1.0;
}

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: i32, trimStartAngle: f32, trimEndAngle: f32, xyz: vec3f) -> f32 {
  if (xyz.z < -0.5 * size || xyz.z > f32(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; }
  let dz = wgsl_mod(xyz.z, size + gap);
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap);
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  if (coilNum < 0.0 || coilNum >= f32(nTurns)) { return 0.0; }
  let rad_angle = angle * (2.0 * M_PI);
  if (coilNum < 1.0) { return select(0.0, 1.0, rad_angle >= trimStartAngle); }
  if (coilNum >= f32(nTurns) - 1.0 && trimEndAngle > 0.0) { return select(0.0, 1.0, rad_angle <= trimEndAngle); }
  return 1.0;
}

fn coilPlusConnectorWires(wireNum: i32, coilNum: i32, numCoils: i32, inc: f32, innerRadius: f32, connectorRadius: f32, size: f32, singleGap: f32, nTurns: i32, xyz_orig: vec3f) -> f32 {
  let gap = size + 2.0 * singleGap;
  let radiusOffset = f32(coilNum - 1);
  let spacingAngle = f32(numCoils - 4) * inc * atan(2.0 * radiusOffset / f32(numCoils - 1));
  let xfm = rotZ(spacingAngle);
  var coilRadius = radiusOffset + innerRadius;
  let trimStartAngle = 0.05;
  var xyz = (xfm * vec4f(xyz_orig, 1.0)).xyz;
  var coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);
  let tz = f32(nTurns) * (size + gap);
  let zexit = (f32(nTurns) + 10.0) * (size + gap);
  if (coilNum == numCoils) {
    if (wireNum == 1) {
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, zexit), size, xyz);
    } else {
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(innerRadius, 0.0, - size), vec3f(innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-innerRadius, 0.0, - size), vec3f(-innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-connectorRadius, 0.0, zexit), size, xyz);
    }
  } else {
    coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, tz + size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(connectorRadius, 0.0, - size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(coilRadius, 0.0, 0.0), size, xyz);
  }
  return clamp(coil, 0.0, 1.0);
}

fn arBifilarElectromagnet(numPairs: i32, innerRadius: f32, size: f32, gap: f32, numTurns: i32, xyz: vec3f) -> vec3f {
  let nTurns = f32(numTurns);
  let inc = M_PI / f32(numPairs);
  let connectorRadius = innerRadius + f32(numPairs) * (size + gap);
  var metal1 = 0.0;
  var metal2 = 0.0;
  let xfm = rotZ(M_PI);
  let xyz180Z = (xfm * vec4f(xyz, 1.0)).xyz;
  for(var i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0);
  let dielectricRadius = f32(numPairs) * (size + gap) + innerRadius + gap;
  let dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  var dielectric = cylinder(dielectricRadius, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  let spindleRadius = f32(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3f(0.0, 0.0, 2.0));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3f(0.0, 0.0, dielectricHeight - 4.0));
  dielectric -= 2.0 * cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  dielectric = clamp(dielectric - metal1 - metal2, 0.0, 1.0);
  return vec3f(metal1, metal2, dielectric);
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 60.0;
  let res = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, xyz);
  return vec4f(res, 0.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-61],
  units: "mm",
}*/


const rotZ = (p, a) => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return vec3(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)), p.z);
};

const box = (sx, sy, sz, ex, ey, ez, size, p) => {
  const h = 0.5 * size;
  const llx = Math.min(sx, ex) - h, lly = Math.min(sy, ey) - h, llz = Math.min(sz, ez) - h;
  const urx = Math.max(sx, ex) + h, ury = Math.max(sy, ey) + h, urz = Math.max(sz, ez) + h;
  const outside = p.x.lessThan(llx).or(p.x.greaterThan(urx))
    .or(p.y.lessThan(lly)).or(p.y.greaterThan(ury))
    .or(p.z.lessThan(llz)).or(p.z.greaterThan(urz));
  return select(outside, float(0.0), float(1.0));
};

const cylinder = (radius, height, p) => {
  const zOK = p.z.greaterThanEqual(0.0).and(p.z.lessThanEqual(height));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const coilSquareFace = (radius, size, gap, nTurns, trimStartAngle, trimEndAngle, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(
    p.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );
  const rxy = length(p.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );
  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dz = mod(p.z, size + gap);
  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dz);
  const lastHelixZ = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextHelixZ = lastHelixZ.add(size + gap);
  const inGap = dz.greaterThan(lastHelixZ.add(0.5 * size)).and(
    dz.lessThan(nextHelixZ.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(floor(p.z.add(0.5 * size).sub(lastHelixZ).div(size + gap)));
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  const angleRad = angle.mul(2.0 * Math.PI);
  const startTrim = select(angleRad.greaterThanEqual(trimStartAngle), float(1.0), float(0.0));
  let trim = select(coilNum.lessThan(1.0), startTrim, float(1.0));
  if (trimEndAngle > 0.0) {
    const endTrim = select(angleRad.lessThanEqual(trimEndAngle), float(1.0), float(0.0));
    trim = select(coilNum.lessThan(1.0), startTrim,
      select(coilNum.greaterThanEqual(nTurns - 1.0), endTrim, float(1.0)));
  }
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), trim, float(0.0));
};

const coilPlusConnectorWires = (wireNum, coilNum, numCoils, inc, innerRadius, connectorRadius, size, singleGap, nTurns, p0) => {
  const gap = size + 2.0 * singleGap;
  const radiusOffset = coilNum - 1;
  const spacingAngle = (numCoils - 4) * inc * Math.atan((2.0 * radiusOffset) / (numCoils - 1));
  const p = rotZ(p0, spacingAngle);
  const coilRadius = radiusOffset + innerRadius;
  const trimStartAngle = 0.05;
  let coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, p);
  const tz = nTurns * (size + gap);
  const zexit = (nTurns + 10.0) * (size + gap);
  const add = (v) => { coil = coil.add(v); };

  if (coilNum === numCoils) {
    if (wireNum === 1) {
      const ps = rotZ(p, 0.45);
      add(box(coilRadius, 0, tz, coilRadius, 0, zexit, size, ps));
    } else {
      add(box(coilRadius, 0, tz, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, innerRadius, 0, -size, size, p));
      add(box(innerRadius, 0, -size, innerRadius, 0, 0, size, p));
      add(box(-innerRadius, 0, -size, -innerRadius, 0, 0, size, p));
      add(box(-connectorRadius, 0, -size, -innerRadius, 0, -size, size, p));
      add(box(-connectorRadius, 0, -size, -connectorRadius, 0, zexit, size, p));
    }
  } else {
    add(box(coilRadius, 0, tz, coilRadius, 0, tz + size, size, p));
    add(box(coilRadius, 0, tz + size, connectorRadius, 0, tz + size, size, p));
    add(box(connectorRadius, 0, -size, connectorRadius, 0, tz + size, size, p));
    const coilRadius2 = coilRadius + size + singleGap;
    add(box(coilRadius2, 0, -size, connectorRadius, 0, -size, size, p));
    add(box(coilRadius2, 0, -size, coilRadius2, 0, 0, size, p));
  }
  return clamp(coil, 0.0, 1.0);
};

const arBifilarElectromagnet = (numPairs, innerRadius, size, gap, numTurns, p) => {
  const nTurns = numTurns;
  const inc = Math.PI / numPairs;
  const connectorRadius = innerRadius + numPairs * (size + gap);
  let metal1 = float(0.0);
  let metal2 = float(0.0);
  const p180 = rotZ(p, Math.PI);
  for (let i = 1; i <= numPairs; i++) {
    metal1 = metal1.add(
      coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p)
    );
    metal2 = metal2.add(
      coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p180)
    );
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2.sub(metal1), 0.0, 1.0);
  const dielectricRadius = numPairs * (size + gap) + innerRadius + gap;
  const dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  let dielectric = cylinder(dielectricRadius, dielectricHeight, p.add(vec3(0, 0, 2)));
  const spindleRadius = (numPairs + 1) * (size + gap) + innerRadius;
  dielectric = dielectric.add(cylinder(spindleRadius, 2.0 * (size + gap), p.add(vec3(0, 0, 2))));
  dielectric = dielectric.add(
    cylinder(spindleRadius, 2.0 * (size + gap), p.sub(vec3(0, 0, dielectricHeight - 4.0)))
  );
  dielectric = dielectric.sub(
    cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, p.add(vec3(0, 0, 2))).mul(2.0)
  );
  dielectric = clamp(dielectric.sub(metal1).sub(metal2), 0.0, 1.0);
  return [metal1, metal2, dielectric];
};

const mainModel4 = (xyz) => {
  const shifted = xyz.add(vec3(0, 0, 60));
  const r = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, shifted);
  return [r[0], r[1], r[2]];
};
```

* Try loading [full-coil-metal-with-dielectric.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/full-coil-metal-with-dielectric.irmf) now in the IRMF editor!
* Try loading [full-coil-metal-with-dielectric-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/full-coil-metal-with-dielectric-wgsl.irmf) now in the IRMF editor!
* Try loading [full-coil-metal-with-dielectric-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/full-coil-metal-with-dielectric-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## axial+radial-bifilar-electromagnet-with-solid-core.irmf

Here is the full model with metal, dielectric, and solid core:

![axial+radial-bifilar-electromagnet-with-solid-core.png](axial+radial-bifilar-electromagnet-with-solid-core.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-61],
  units: "mm",
}*/

#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/rotation.glsl"
#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/primitives.glsl"

#define M_PI 3.1415926535897932384626433832795

float coilSquareFace(in float radius, in float size, in float gap, in int nTurns, in float trimStartAngle, in float trimEndAngle, in vec3 xyz) {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > float(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }

  // Then, constrain the coil to the cylinder with wall thickness "size":
  float rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }

  // If the current point is between the coils, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  float dz = mod(xyz.z, size + gap);

  float lastHelixZ = angle * (size + gap);
  float coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap); // center of current coil.
    coilNum = -1.0;
  }
  float nextHelixZ = lastHelixZ + (size + gap); // center of next higher vertical coil.

  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }

  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= float(nTurns)) { return 0.0; }

  // For the last two checks, convert angle back to radians.
  angle *= (2.0 * M_PI);

  // If the current point is in the first coil, use the trimStartAngle.
  if (coilNum < 1.0) { return angle >= trimStartAngle ? 1.0 : 0.0; }

  // If the current point is in the last coil, use the trimEndAngle.
  if (coilNum >= float(nTurns) - 1.0 && trimEndAngle > 0.0) { return angle <= trimEndAngle ? 1.0 : 0.0; }

  return 1.0;
}

float coilPlusConnectorWires(in int wireNum, in int coilNum, in int numCoils, in float inc, in float innerRadius, in float connectorRadius, in float size, in float singleGap, in int nTurns, in vec3 xyz) {
  float gap = size + 2.0 * singleGap;
  float radiusOffset = float(coilNum - 1);
  // next line is experimental - works for numPairs=20:
  float spacingAngle = float(numCoils - 4) * inc * atan(2.0 * radiusOffset / float(numCoils - 1));
  mat4 xfm = mat4(1) * rotZ(spacingAngle);
  float coilRadius = radiusOffset + innerRadius;
  float trimStartAngle = 0.05; // experimental - works somewhat for numPairs=20, but not ideal.

  xyz = (vec4(xyz, 1.0) * xfm).xyz;
  float coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);

  float tz = float(nTurns) * (size + gap);
  float zexit = (float(nTurns) + 10.0) * (size + gap);

  if (coilNum == numCoils) {
    // Special case to access the exit wires.
    if (wireNum == 1) {
      coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, zexit), size, xyz);
    } else {
      coil += box(vec3(coilRadius, 0, tz), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(innerRadius, 0, - size), size, xyz);
      coil += box(vec3(innerRadius, 0, - size), vec3(innerRadius, 0, 0), size, xyz);
      // Now the other exit wire
      coil += box(vec3(-innerRadius, 0, - size), vec3(-innerRadius, 0, 0), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-innerRadius, 0, - size), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-connectorRadius, 0, zexit), size, xyz);
    }
  } else {
    coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, tz + size), size, xyz);
    coil += box(vec3(coilRadius, 0, tz + size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box(vec3(coilRadius, 0, - size), vec3(connectorRadius, 0, - size), size, xyz);
    coil += box(vec3(coilRadius, 0, - size), vec3(coilRadius, 0, 0), size, xyz);
  }

  return clamp(coil, 0.0, 1.0);
}

vec3 arBifilarElectromagnet(int numPairs, float innerRadius, float size, float gap, int numTurns, in vec3 xyz) {
  float nTurns = float(numTurns);
  float inc = M_PI / float(numPairs);
  float connectorRadius = innerRadius + float(numPairs) * (size + gap);

  float metal1 = 0.0;
  float metal2 = 0.0;

  mat4 xfm = mat4(1) * rotZ(M_PI);
  vec3 xyz180Z = (vec4(xyz, 1.0) * xfm).xyz;

  for(int i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0); // Don't double the metals on overlap.

  float dielectricRadius = float(numPairs) * (size + gap) + innerRadius + gap;
  float dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  float dielectric = cylinder(dielectricRadius, dielectricHeight, xyz + vec3(0, 0, 2));

  float solidCore = 2.0 * cylinder(innerRadius - 0.5 * size - gap - 0.05, dielectricHeight, xyz + vec3(0, 0, 2));

  float spindleRadius = float(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3(0, 0, 2));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3(0, 0, dielectricHeight - 4.0));
  dielectric -= 2.0 * cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, xyz + vec3(0, 0, 2));
  // This next step is important... we don't want any dielectric material wherever
  // the metal is located, so we just subtract the metal out of the dielectric.
  dielectric = clamp(dielectric - metal1 - metal2, 0.0, 1.0);

  return vec3(metal1 + metal2, solidCore, dielectric);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.z += 60.0;
  materials.xyz = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, xyz);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-61],
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

fn rotZ(angle: f32) -> mat4x4f {
  return mat3ToMat4(rotAxis(vec3f(0.0, 0.0, 1.0), angle));
}

fn box_prim(start: vec3f, end: vec3f, size: f32, xyz: vec3f) -> f32 {
  let ll = min(start, end) - vec3f(0.5 * size);
  let ur = max(start, end) + vec3f(0.5 * size);
  if (any(xyz < ll) || any(xyz > ur)) { return 0.0; }
  return 1.0;
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  return 1.0;
}

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: i32, trimStartAngle: f32, trimEndAngle: f32, xyz: vec3f) -> f32 {
  if (xyz.z < -0.5 * size || xyz.z > f32(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; }
  let dz = wgsl_mod(xyz.z, size + gap);
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap);
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  if (coilNum < 0.0 || coilNum >= f32(nTurns)) { return 0.0; }
  let rad_angle = angle * (2.0 * M_PI);
  if (coilNum < 1.0) { return select(0.0, 1.0, rad_angle >= trimStartAngle); }
  if (coilNum >= f32(nTurns) - 1.0 && trimEndAngle > 0.0) { return select(0.0, 1.0, rad_angle <= trimEndAngle); }
  return 1.0;
}

fn coilPlusConnectorWires(wireNum: i32, coilNum: i32, numCoils: i32, inc: f32, innerRadius: f32, connectorRadius: f32, size: f32, singleGap: f32, nTurns: i32, xyz_orig: vec3f) -> f32 {
  let gap = size + 2.0 * singleGap;
  let radiusOffset = f32(coilNum - 1);
  let spacingAngle = f32(numCoils - 4) * inc * atan(2.0 * radiusOffset / f32(numCoils - 1));
  let xfm = rotZ(spacingAngle);
  var coilRadius = radiusOffset + innerRadius;
  let trimStartAngle = 0.05;
  var xyz = (xfm * vec4f(xyz_orig, 1.0)).xyz;
  var coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);
  let tz = f32(nTurns) * (size + gap);
  let zexit = (f32(nTurns) + 10.0) * (size + gap);
  if (coilNum == numCoils) {
    if (wireNum == 1) {
      let slightRot = rotZ(0.45);
      xyz = (slightRot * vec4f(xyz, 1.0)).xyz;
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, zexit), size, xyz);
    } else {
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(innerRadius, 0.0, - size), vec3f(innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-innerRadius, 0.0, - size), vec3f(-innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-connectorRadius, 0.0, zexit), size, xyz);
    }
  } else {
    coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, tz + size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(connectorRadius, 0.0, - size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(coilRadius, 0.0, 0.0), size, xyz);
  }
  return clamp(coil, 0.0, 1.0);
}

fn arBifilarElectromagnet(numPairs: i32, innerRadius: f32, size: f32, gap: f32, numTurns: i32, xyz: vec3f) -> vec3f {
  let nTurns = f32(numTurns);
  let inc = M_PI / f32(numPairs);
  let connectorRadius = innerRadius + f32(numPairs) * (size + gap);
  var metal1 = 0.0;
  var metal2 = 0.0;
  let xfm = rotZ(M_PI);
  let xyz180Z = (xfm * vec4f(xyz, 1.0)).xyz;
  for(var i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0);
  let dielectricRadius = f32(numPairs) * (size + gap) + innerRadius + gap;
  let dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  var dielectric = cylinder(dielectricRadius, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  let solidCore = 2.0 * cylinder(innerRadius - 0.5 * size - gap - 0.05, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  let spindleRadius = f32(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3f(0.0, 0.0, 2.0));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3f(0.0, 0.0, dielectricHeight - 4.0));
  dielectric -= 2.0 * cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  dielectric = clamp(dielectric - metal1 - metal2, 0.0, 1.0);
  return vec3f(metal1 + metal2, solidCore, dielectric);
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.z += 60.0;
  let res = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, xyz);
  return vec4f(res, 0.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper","copper","dielectric"],
  max: [25,25,71],
  min: [-25,-25,-61],
  units: "mm",
}*/


const rotZ = (p, a) => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return vec3(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)), p.z);
};

const box = (sx, sy, sz, ex, ey, ez, size, p) => {
  const h = 0.5 * size;
  const llx = Math.min(sx, ex) - h, lly = Math.min(sy, ey) - h, llz = Math.min(sz, ez) - h;
  const urx = Math.max(sx, ex) + h, ury = Math.max(sy, ey) + h, urz = Math.max(sz, ez) + h;
  const outside = p.x.lessThan(llx).or(p.x.greaterThan(urx))
    .or(p.y.lessThan(lly)).or(p.y.greaterThan(ury))
    .or(p.z.lessThan(llz)).or(p.z.greaterThan(urz));
  return select(outside, float(0.0), float(1.0));
};

const cylinder = (radius, height, p) => {
  const zOK = p.z.greaterThanEqual(0.0).and(p.z.lessThanEqual(height));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const coilSquareFace = (radius, size, gap, nTurns, trimStartAngle, trimEndAngle, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(
    p.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );
  const rxy = length(p.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );
  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dz = mod(p.z, size + gap);
  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dz);
  const lastHelixZ = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextHelixZ = lastHelixZ.add(size + gap);
  const inGap = dz.greaterThan(lastHelixZ.add(0.5 * size)).and(
    dz.lessThan(nextHelixZ.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(floor(p.z.add(0.5 * size).sub(lastHelixZ).div(size + gap)));
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  const angleRad = angle.mul(2.0 * Math.PI);
  const startTrim = select(angleRad.greaterThanEqual(trimStartAngle), float(1.0), float(0.0));
  let trim = select(coilNum.lessThan(1.0), startTrim, float(1.0));
  if (trimEndAngle > 0.0) {
    const endTrim = select(angleRad.lessThanEqual(trimEndAngle), float(1.0), float(0.0));
    trim = select(coilNum.lessThan(1.0), startTrim,
      select(coilNum.greaterThanEqual(nTurns - 1.0), endTrim, float(1.0)));
  }
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), trim, float(0.0));
};

const coilPlusConnectorWires = (wireNum, coilNum, numCoils, inc, innerRadius, connectorRadius, size, singleGap, nTurns, p0) => {
  const gap = size + 2.0 * singleGap;
  const radiusOffset = coilNum - 1;
  const spacingAngle = (numCoils - 4) * inc * Math.atan((2.0 * radiusOffset) / (numCoils - 1));
  const p = rotZ(p0, spacingAngle);
  const coilRadius = radiusOffset + innerRadius;
  const trimStartAngle = 0.05;
  let coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, p);
  const tz = nTurns * (size + gap);
  const zexit = (nTurns + 10.0) * (size + gap);
  const add = (v) => { coil = coil.add(v); };

  if (coilNum === numCoils) {
    if (wireNum === 1) {
      const ps = rotZ(p, 0.45);
      add(box(coilRadius, 0, tz, coilRadius, 0, zexit, size, ps));
    } else {
      add(box(coilRadius, 0, tz, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, innerRadius, 0, -size, size, p));
      add(box(innerRadius, 0, -size, innerRadius, 0, 0, size, p));
      add(box(-innerRadius, 0, -size, -innerRadius, 0, 0, size, p));
      add(box(-connectorRadius, 0, -size, -innerRadius, 0, -size, size, p));
      add(box(-connectorRadius, 0, -size, -connectorRadius, 0, zexit, size, p));
    }
  } else {
    add(box(coilRadius, 0, tz, coilRadius, 0, tz + size, size, p));
    add(box(coilRadius, 0, tz + size, connectorRadius, 0, tz + size, size, p));
    add(box(connectorRadius, 0, -size, connectorRadius, 0, tz + size, size, p));
    const coilRadius2 = coilRadius + size + singleGap;
    add(box(coilRadius2, 0, -size, connectorRadius, 0, -size, size, p));
    add(box(coilRadius2, 0, -size, coilRadius2, 0, 0, size, p));
  }
  return clamp(coil, 0.0, 1.0);
};

const arBifilarElectromagnet = (numPairs, innerRadius, size, gap, numTurns, p) => {
  const nTurns = numTurns;
  const inc = Math.PI / numPairs;
  const connectorRadius = innerRadius + numPairs * (size + gap);
  let metal1 = float(0.0);
  let metal2 = float(0.0);
  const p180 = rotZ(p, Math.PI);
  for (let i = 1; i <= numPairs; i++) {
    metal1 = metal1.add(
      coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p)
    );
    metal2 = metal2.add(
      coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p180)
    );
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2.sub(metal1), 0.0, 1.0);
  const dielectricRadius = numPairs * (size + gap) + innerRadius + gap;
  const dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  let dielectric = cylinder(dielectricRadius, dielectricHeight, p.add(vec3(0, 0, 2)));
  const solidCore = cylinder(
    innerRadius - 0.5 * size - gap - 0.05,
    dielectricHeight,
    p.add(vec3(0, 0, 2))
  ).mul(2.0);
  const spindleRadius = (numPairs + 1) * (size + gap) + innerRadius;
  dielectric = dielectric.add(cylinder(spindleRadius, 2.0 * (size + gap), p.add(vec3(0, 0, 2))));
  dielectric = dielectric.add(
    cylinder(spindleRadius, 2.0 * (size + gap), p.sub(vec3(0, 0, dielectricHeight - 4.0)))
  );
  dielectric = dielectric.sub(
    cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, p.add(vec3(0, 0, 2))).mul(2.0)
  );
  dielectric = clamp(dielectric.sub(metal1).sub(metal2), 0.0, 1.0);
  return [metal1.add(metal2), solidCore, dielectric];
};

const mainModel4 = (xyz) => {
  const shifted = xyz.add(vec3(0, 0, 60));
  const r = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, shifted);
  return [r[0], r[1], r[2]];
};
```

* Try loading [axial+radial-bifilar-electromagnet-with-solid-core.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial+radial-bifilar-electromagnet-with-solid-core.irmf) now in the IRMF editor!
* Try loading [axial+radial-bifilar-electromagnet-with-solid-core-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial+radial-bifilar-electromagnet-with-solid-core-wgsl.irmf) now in the IRMF editor!
* Try loading [axial+radial-bifilar-electromagnet-with-solid-core-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial+radial-bifilar-electromagnet-with-solid-core-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## axial+radial-bifilar-electromagnet-with-solid-core-rot90.irmf

Here is the same model, rotated to the horizontal:

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["copper","copper","dielectric"],
  max: [71,25,25],
  min: [-61,-25,-25],
  units: "mm",
}*/

#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/rotation.glsl"
#include "github.com/gmlewis/irmf-examples/blob/6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021/examples/012-bifilar-electromagnet/primitives.glsl"

#define M_PI 3.1415926535897932384626433832795

float coilSquareFace(in float radius, in float size, in float gap, in int nTurns, in float trimStartAngle, in float trimEndAngle, in vec3 xyz) {
  // First, trivial reject on the two vertical ends of the coil.
  if (xyz.z < -0.5 * size || xyz.z > float(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }

  // Then, constrain the coil to the cylinder with wall thickness "size":
  float rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }

  // If the current point is between the coils, return no material:
  float angle = atan(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; } // 0 <= angle <= 1 between coils from center to center.
  // 0 <= dz <= (size+gap) between coils from center to center.
  float dz = mod(xyz.z, size + gap);

  float lastHelixZ = angle * (size + gap);
  float coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap); // center of current coil.
    coilNum = -1.0;
  }
  float nextHelixZ = lastHelixZ + (size + gap); // center of next higher vertical coil.

  // If the current point is within the gap between the two coils, reject it.
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }

  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));

  // If the current point is in a coil numbered outside the current range, reject it.
  if (coilNum < 0.0 || coilNum >= float(nTurns)) { return 0.0; }

  // For the last two checks, convert angle back to radians.
  angle *= (2.0 * M_PI);

  // If the current point is in the first coil, use the trimStartAngle.
  if (coilNum < 1.0) { return angle >= trimStartAngle ? 1.0 : 0.0; }

  // If the current point is in the last coil, use the trimEndAngle.
  if (coilNum >= float(nTurns) - 1.0 && trimEndAngle > 0.0) { return angle <= trimEndAngle ? 1.0 : 0.0; }

  return 1.0;
}

float coilPlusConnectorWires(in int wireNum, in int coilNum, in int numCoils, in float inc, in float innerRadius, in float connectorRadius, in float size, in float singleGap, in int nTurns, in vec3 xyz) {
  float gap = size + 2.0 * singleGap;
  float radiusOffset = float(coilNum - 1);
  // next line is experimental - works for numPairs=20:
  float spacingAngle = float(numCoils - 4) * inc * atan(2.0 * radiusOffset / float(numCoils - 1));
  mat4 xfm = mat4(1) * rotZ(spacingAngle);
  float coilRadius = radiusOffset + innerRadius;
  float trimStartAngle = 0.05; // experimental - works somewhat for numPairs=20, but not ideal.

  xyz = (vec4(xyz, 1.0) * xfm).xyz;
  float coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);

  float tz = float(nTurns) * (size + gap);
  float zexit = (float(nTurns) + 10.0) * (size + gap);

  if (coilNum == numCoils) {
    // Special case to access the exit wires.
    if (wireNum == 1) {
      coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, zexit), size, xyz);
    } else {
      coil += box(vec3(coilRadius, 0, tz), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz), size, xyz);
      coil += box(vec3(connectorRadius, 0, - size), vec3(innerRadius, 0, - size), size, xyz);
      coil += box(vec3(innerRadius, 0, - size), vec3(innerRadius, 0, 0), size, xyz);
      // Now the other exit wire
      coil += box(vec3(-innerRadius, 0, - size), vec3(-innerRadius, 0, 0), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-innerRadius, 0, - size), size, xyz);
      coil += box(vec3(-connectorRadius, 0, - size), vec3(-connectorRadius, 0, zexit), size, xyz);
    }
  } else {
    coil += box(vec3(coilRadius, 0, tz), vec3(coilRadius, 0, tz + size), size, xyz);
    coil += box(vec3(coilRadius, 0, tz + size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coil += box(vec3(connectorRadius, 0, - size), vec3(connectorRadius, 0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box(vec3(coilRadius, 0, - size), vec3(connectorRadius, 0, - size), size, xyz);
    coil += box(vec3(coilRadius, 0, - size), vec3(coilRadius, 0, 0), size, xyz);
  }

  return clamp(coil, 0.0, 1.0);
}

vec3 arBifilarElectromagnet(int numPairs, float innerRadius, float size, float gap, int numTurns, in vec3 xyz) {
  float nTurns = float(numTurns);
  float inc = M_PI / float(numPairs);
  float connectorRadius = innerRadius + float(numPairs) * (size + gap);

  float metal1 = 0.0;
  float metal2 = 0.0;

  mat4 xfm = mat4(1) * rotZ(M_PI);
  vec3 xyz180Z = (vec4(xyz, 1.0) * xfm).xyz;

  for(int i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0); // Don't double the metals on overlap.

  float dielectricRadius = float(numPairs) * (size + gap) + innerRadius + gap;
  float dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  float dielectric = cylinder(dielectricRadius, dielectricHeight, xyz + vec3(0, 0, 2));

  float solidCore = 2.0 * cylinder(innerRadius - 0.5 * size - gap - 0.05, dielectricHeight, xyz + vec3(0, 0, 2));

  float spindleRadius = float(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3(0, 0, 2));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3(0, 0, dielectricHeight - 4.0));
  dielectric -= 2.0 * cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, xyz + vec3(0, 0, 2));
  // This next step is important... we don't want any dielectric material wherever
  // the metal is located, so we just subtract the metal out of the dielectric.
  dielectric = clamp(dielectric - metal1 - metal2, 0.0, 1.0);

  return vec3(metal1 + metal2, solidCore, dielectric);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz.x += 60.0;
  materials.xyz = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, xyz.zyx);
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["copper","copper","dielectric"],
  max: [71,25,25],
  min: [-61,-25,-25],
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

fn rotZ(angle: f32) -> mat4x4f {
  return mat3ToMat4(rotAxis(vec3f(0.0, 0.0, 1.0), angle));
}

fn box_prim(start: vec3f, end: vec3f, size: f32, xyz: vec3f) -> f32 {
  let ll = min(start, end) - vec3f(0.5 * size);
  let ur = max(start, end) + vec3f(0.5 * size);
  if (any(xyz < ll) || any(xyz > ur)) { return 0.0; }
  return 1.0;
}

fn cylinder(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z < 0.0 || xyz.z > height) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy > radius) { return 0.0; }
  return 1.0;
}

fn wgsl_mod(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}

fn coilSquareFace(radius: f32, size: f32, gap: f32, nTurns: i32, trimStartAngle: f32, trimEndAngle: f32, xyz: vec3f) -> f32 {
  if (xyz.z < -0.5 * size || xyz.z > f32(nTurns) * (size + gap) + 0.5 * size) { return 0.0; }
  let rxy = length(xyz.xy);
  if (rxy < radius - 0.5 * size || rxy > radius + 0.5 * size) { return 0.0; }
  var angle = atan2(xyz.y, xyz.x) / (2.0 * M_PI);
  if (angle < 0.0) { angle += 1.0; }
  let dz = wgsl_mod(xyz.z, size + gap);
  var lastHelixZ = angle * (size + gap);
  var coilNum = 0.0;
  if (lastHelixZ > dz) {
    lastHelixZ -= (size + gap);
    coilNum = -1.0;
  }
  let nextHelixZ = lastHelixZ + (size + gap);
  if (dz > lastHelixZ + 0.5 * size && dz < nextHelixZ - 0.5 * size) { return 0.0; }
  coilNum += floor((xyz.z + (0.5 * size) - lastHelixZ) / (size + gap));
  if (coilNum < 0.0 || coilNum >= f32(nTurns)) { return 0.0; }
  let rad_angle = angle * (2.0 * M_PI);
  if (coilNum < 1.0) { return select(0.0, 1.0, rad_angle >= trimStartAngle); }
  if (coilNum >= f32(nTurns) - 1.0 && trimEndAngle > 0.0) { return select(0.0, 1.0, rad_angle <= trimEndAngle); }
  return 1.0;
}

fn coilPlusConnectorWires(wireNum: i32, coilNum: i32, numCoils: i32, inc: f32, innerRadius: f32, connectorRadius: f32, size: f32, singleGap: f32, nTurns: i32, xyz_orig: vec3f) -> f32 {
  let gap = size + 2.0 * singleGap;
  let radiusOffset = f32(coilNum - 1);
  let spacingAngle = f32(numCoils - 4) * inc * atan(2.0 * radiusOffset / f32(numCoils - 1));
  let xfm = rotZ(spacingAngle);
  var coilRadius = radiusOffset + innerRadius;
  let trimStartAngle = 0.05;
  var xyz = (xfm * vec4f(xyz_orig, 1.0)).xyz;
  var coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, xyz);
  let tz = f32(nTurns) * (size + gap);
  let zexit = (f32(nTurns) + 10.0) * (size + gap);
  if (coilNum == numCoils) {
    if (wireNum == 1) {
      let slightRot = rotZ(0.45);
      xyz = (slightRot * vec4f(xyz, 1.0)).xyz;
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, zexit), size, xyz);
    } else {
      coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz), size, xyz);
      coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(innerRadius, 0.0, - size), vec3f(innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-innerRadius, 0.0, - size), vec3f(-innerRadius, 0.0, 0.0), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-innerRadius, 0.0, - size), size, xyz);
      coil += box_prim(vec3f(-connectorRadius, 0.0, - size), vec3f(-connectorRadius, 0.0, zexit), size, xyz);
    }
  } else {
    coil += box_prim(vec3f(coilRadius, 0.0, tz), vec3f(coilRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, tz + size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coil += box_prim(vec3f(connectorRadius, 0.0, - size), vec3f(connectorRadius, 0.0, tz + size), size, xyz);
    coilRadius += size + singleGap;
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(connectorRadius, 0.0, - size), size, xyz);
    coil += box_prim(vec3f(coilRadius, 0.0, - size), vec3f(coilRadius, 0.0, 0.0), size, xyz);
  }
  return clamp(coil, 0.0, 1.0);
}

fn arBifilarElectromagnet(numPairs: i32, innerRadius: f32, size: f32, gap: f32, numTurns: i32, xyz: vec3f) -> vec3f {
  let nTurns = f32(numTurns);
  let inc = M_PI / f32(numPairs);
  let connectorRadius = innerRadius + f32(numPairs) * (size + gap);
  var metal1 = 0.0;
  var metal2 = 0.0;
  let xfm = rotZ(M_PI);
  let xyz180Z = (xfm * vec4f(xyz, 1.0)).xyz;
  for(var i = 1; i <= numPairs; i ++ ) {
    metal1 += coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz);
    metal2 += coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, xyz180Z);
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2 - metal1, 0.0, 1.0);
  let dielectricRadius = f32(numPairs) * (size + gap) + innerRadius + gap;
  let dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  var dielectric = cylinder(dielectricRadius, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  let solidCore = 2.0 * cylinder(innerRadius - 0.5 * size - gap - 0.05, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  let spindleRadius = f32(numPairs + 1) * (size + gap) + innerRadius;
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz + vec3f(0.0, 0.0, 2.0));
  dielectric += cylinder(spindleRadius, 2.0 * (size + gap), xyz - vec3f(0.0, 0.0, dielectricHeight - 4.0));
  dielectric -= 2.0 * cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, xyz + vec3f(0.0, 0.0, 2.0));
  dielectric = clamp(dielectric - metal1 - metal2, 0.0, 1.0);
  return vec3f(metal1 + metal2, solidCore, dielectric);
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz.x += 60.0;
  let res = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, xyz.zyx);
  return vec4f(res, 0.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["copper","copper","dielectric"],
  max: [71,25,25],
  min: [-61,-25,-25],
  units: "mm",
}*/


const rotZ = (p, a) => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return vec3(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)), p.z);
};

const box = (sx, sy, sz, ex, ey, ez, size, p) => {
  const h = 0.5 * size;
  const llx = Math.min(sx, ex) - h, lly = Math.min(sy, ey) - h, llz = Math.min(sz, ez) - h;
  const urx = Math.max(sx, ex) + h, ury = Math.max(sy, ey) + h, urz = Math.max(sz, ez) + h;
  const outside = p.x.lessThan(llx).or(p.x.greaterThan(urx))
    .or(p.y.lessThan(lly)).or(p.y.greaterThan(ury))
    .or(p.z.lessThan(llz)).or(p.z.greaterThan(urz));
  return select(outside, float(0.0), float(1.0));
};

const cylinder = (radius, height, p) => {
  const zOK = p.z.greaterThanEqual(0.0).and(p.z.lessThanEqual(height));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const coilSquareFace = (radius, size, gap, nTurns, trimStartAngle, trimEndAngle, p) => {
  const zOK = p.z.greaterThanEqual(-0.5 * size).and(
    p.z.lessThanEqual(nTurns * (size + gap) + 0.5 * size)
  );
  const rxy = length(p.xy);
  const rOK = rxy.greaterThanEqual(radius - 0.5 * size).and(
    rxy.lessThanEqual(radius + 0.5 * size)
  );
  let angle = atan(p.y, p.x).div(2.0 * Math.PI);
  angle = select(angle.lessThan(0.0), angle.add(1.0), angle);
  const dz = mod(p.z, size + gap);
  const base = angle.mul(size + gap);
  const wrapped = base.greaterThan(dz);
  const lastHelixZ = select(wrapped, base.sub(size + gap), base);
  const coilNum0 = select(wrapped, float(-1.0), float(0.0));
  const nextHelixZ = lastHelixZ.add(size + gap);
  const inGap = dz.greaterThan(lastHelixZ.add(0.5 * size)).and(
    dz.lessThan(nextHelixZ.sub(0.5 * size))
  );
  const coilNum = coilNum0.add(floor(p.z.add(0.5 * size).sub(lastHelixZ).div(size + gap)));
  const coilOK = coilNum.greaterThanEqual(0.0).and(coilNum.lessThan(nTurns));
  const angleRad = angle.mul(2.0 * Math.PI);
  const startTrim = select(angleRad.greaterThanEqual(trimStartAngle), float(1.0), float(0.0));
  let trim = select(coilNum.lessThan(1.0), startTrim, float(1.0));
  if (trimEndAngle > 0.0) {
    const endTrim = select(angleRad.lessThanEqual(trimEndAngle), float(1.0), float(0.0));
    trim = select(coilNum.lessThan(1.0), startTrim,
      select(coilNum.greaterThanEqual(nTurns - 1.0), endTrim, float(1.0)));
  }
  return select(zOK.and(rOK).and(inGap.not()).and(coilOK), trim, float(0.0));
};

const coilPlusConnectorWires = (wireNum, coilNum, numCoils, inc, innerRadius, connectorRadius, size, singleGap, nTurns, p0) => {
  const gap = size + 2.0 * singleGap;
  const radiusOffset = coilNum - 1;
  const spacingAngle = (numCoils - 4) * inc * Math.atan((2.0 * radiusOffset) / (numCoils - 1));
  const p = rotZ(p0, spacingAngle);
  const coilRadius = radiusOffset + innerRadius;
  const trimStartAngle = 0.05;
  let coil = coilSquareFace(coilRadius, size, gap, nTurns, trimStartAngle, 0.0, p);
  const tz = nTurns * (size + gap);
  const zexit = (nTurns + 10.0) * (size + gap);
  const add = (v) => { coil = coil.add(v); };

  if (coilNum === numCoils) {
    if (wireNum === 1) {
      const ps = rotZ(p, 0.45);
      add(box(coilRadius, 0, tz, coilRadius, 0, zexit, size, ps));
    } else {
      add(box(coilRadius, 0, tz, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, connectorRadius, 0, tz, size, p));
      add(box(connectorRadius, 0, -size, innerRadius, 0, -size, size, p));
      add(box(innerRadius, 0, -size, innerRadius, 0, 0, size, p));
      add(box(-innerRadius, 0, -size, -innerRadius, 0, 0, size, p));
      add(box(-connectorRadius, 0, -size, -innerRadius, 0, -size, size, p));
      add(box(-connectorRadius, 0, -size, -connectorRadius, 0, zexit, size, p));
    }
  } else {
    add(box(coilRadius, 0, tz, coilRadius, 0, tz + size, size, p));
    add(box(coilRadius, 0, tz + size, connectorRadius, 0, tz + size, size, p));
    add(box(connectorRadius, 0, -size, connectorRadius, 0, tz + size, size, p));
    const coilRadius2 = coilRadius + size + singleGap;
    add(box(coilRadius2, 0, -size, connectorRadius, 0, -size, size, p));
    add(box(coilRadius2, 0, -size, coilRadius2, 0, 0, size, p));
  }
  return clamp(coil, 0.0, 1.0);
};

const arBifilarElectromagnet = (numPairs, innerRadius, size, gap, numTurns, p) => {
  const nTurns = numTurns;
  const inc = Math.PI / numPairs;
  const connectorRadius = innerRadius + numPairs * (size + gap);
  let metal1 = float(0.0);
  let metal2 = float(0.0);
  const p180 = rotZ(p, Math.PI);
  for (let i = 1; i <= numPairs; i++) {
    metal1 = metal1.add(
      coilPlusConnectorWires(1, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p)
    );
    metal2 = metal2.add(
      coilPlusConnectorWires(2, i, numPairs, inc, innerRadius, connectorRadius, size, gap, numTurns, p180)
    );
  }
  metal1 = clamp(metal1, 0.0, 1.0);
  metal2 = clamp(metal2.sub(metal1), 0.0, 1.0);
  const dielectricRadius = numPairs * (size + gap) + innerRadius + gap;
  const dielectricHeight = 2.0 * (nTurns + 2.0) * (size + gap);
  let dielectric = cylinder(dielectricRadius, dielectricHeight, p.add(vec3(0, 0, 2)));
  const solidCore = cylinder(
    innerRadius - 0.5 * size - gap - 0.05,
    dielectricHeight,
    p.add(vec3(0, 0, 2))
  ).mul(2.0);
  const spindleRadius = (numPairs + 1) * (size + gap) + innerRadius;
  dielectric = dielectric.add(cylinder(spindleRadius, 2.0 * (size + gap), p.add(vec3(0, 0, 2))));
  dielectric = dielectric.add(
    cylinder(spindleRadius, 2.0 * (size + gap), p.sub(vec3(0, 0, dielectricHeight - 4.0)))
  );
  dielectric = dielectric.sub(
    cylinder(innerRadius - 0.5 * size - gap, dielectricHeight, p.add(vec3(0, 0, 2))).mul(2.0)
  );
  dielectric = clamp(dielectric.sub(metal1).sub(metal2), 0.0, 1.0);
  return [metal1.add(metal2), solidCore, dielectric];
};

const mainModel4 = (xyz) => {
  const shifted = xyz.add(vec3(60, 0, 0));
  const r = arBifilarElectromagnet(20, 3.0, 0.85, 0.15, 61, shifted.zyx);
  return [r[0], r[1], r[2]];
};
```

* Try loading [axial+radial-bifilar-electromagnet-with-solid-core-rot90.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial+radial-bifilar-electromagnet-with-solid-core-rot90.irmf) now in the IRMF editor!
* Try loading [axial+radial-bifilar-electromagnet-with-solid-core-rot90-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial+radial-bifilar-electromagnet-with-solid-core-rot90-wgsl.irmf) now in the IRMF editor!
* Try loading [axial+radial-bifilar-electromagnet-with-solid-core-rot90-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/012-bifilar-electromagnet/axial+radial-bifilar-electromagnet-with-solid-core-rot90-tsl.irmf) now in the IRMF editor!

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
