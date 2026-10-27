# 015-soapdish

Let's model a soapdish (like [this one](http://www.thingiverse.com/thing:135154) on Thingiverse.com)
in a step-by-step, tutorial fashion.

## soapdish-step-01.irmf

First, the general shape of the soapdish is a squished upside-down cone,
so let's start with a cone that is chopped off by its minimum bounding box.

![soapdish-step-01.png](soapdish-step-01.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float soapdish(in vec3 xyz) {
  float result = cone(42.5, 20.0, xyz);
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn soapdish(xyz: vec3f) -> f32 {
  let result = cone(42.5, 20.0, xyz);
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (p) => cone(42.5, 20.0, p);
const mainModel4 = (xyz) => [soapdish(xyz.add(vec3(0, 0, 12)))];
```

* Try loading [soapdish-step-01.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-01.irmf) now in the IRMF editor!
* Try loading [soapdish-step-01-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-01-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-01-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-01-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-02.irmf

Let's hollow out the dish with another identical cone slide up vertically by
a small amount. But this time, we need to stop the cone at z<0 so that the
base of the dish is solid.

![soapdish-step-02.png](soapdish-step-02.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float soapdish(in vec3 xyz) {
  float result = cone(42.5, 20.0, xyz);
  result -= cone(42.5, 20.0, xyz - vec3(0, 0, 3));
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } 
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn soapdish(xyz: vec3f) -> f32 {
  var result = cone(42.5, 20.0, xyz);
  result += halfTorus(42.5, 0.0, xyz - vec3f(0.0, 0.0, 20.0));
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (p) => {
  return cone(42.5, 20.0, p).sub(cone(42.5, 20.0, p.sub(vec3(0, 0, 3))));
};
const mainModel4 = (xyz) => [soapdish(xyz.add(vec3(0, 0, 12)))];
```

* Try loading [soapdish-step-02.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-02.irmf) now in the IRMF editor!
* Try loading [soapdish-step-02-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-02-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-02-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-02-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-03.irmf

Let's round the edge of the top of the dish so that it does not have sharp edges.
One way to do this is to add a half-torus (the upper half) to the original cone
before subtracting out the inner cone. That way, it trims the torus along with
the outer cone.

![soapdish-step-03.png](soapdish-step-03.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float halfTorus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } // Just to top half.
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float soapdish(in vec3 xyz) {
  float result = cone(42.5, 20.0, xyz);
  result += halfTorus(42.5 - 3.0, 3.0, xyz - vec3(0, 0, 20));
  result -= cone(42.5, 20.0, xyz - vec3(0, 0, 3));
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } 
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn soapdish(xyz: vec3f) -> f32 {
  var result = cone(42.5, 20.0, xyz);
  result += halfTorus(39.5, 3.0, xyz - vec3f(0.0, 0.0, 20.0));
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [42.5,42.5,12],
  min: [-42.5,-42.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (p) => {
  return cone(42.5, 20.0, p)
    .add(halfTorus(42.5 - 3.0, 3.0, p.sub(vec3(0, 0, 20))))
    .sub(cone(42.5, 20.0, p.sub(vec3(0, 0, 3))));
};
const mainModel4 = (xyz) => [soapdish(xyz.add(vec3(0, 0, 12)))];
```

* Try loading [soapdish-step-03.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-03.irmf) now in the IRMF editor!
* Try loading [soapdish-step-03-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-03-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-03-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-03-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-04.irmf

I just realized that I measured incorrectly. Instead of it being 85mm wide, it is
105mm side. I could go back and fix all the steps above, but this is actually a
great learning opportunity, so let's go in and fix the dimensions.

The MBB needs to change and let's add a couple parameters to the soapdish
function to make it easy to change.

![soapdish-step-04.png](soapdish-step-04.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float halfTorus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } // Just to top half.
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float soapdish(float width, float height, in vec3 xyz) {
  const float baseHeight = 4.0;
  const float separation = 3.0;
  float result = cone(0.5 * width, height - baseHeight, xyz);
  result += halfTorus(0.5 * width - separation, separation, xyz - vec3(0, 0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz - vec3(0, 0, separation));
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(105.0, 24.0, xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } 
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn soapdish(xyz: vec3f) -> f32 {
  let width = 105.0;
  let depth = 82.0;
  let height = 24.0;
  let squish = vec3f(1.0, width / depth, 1.0);
  var result = cone(0.5 * width, 20.0, xyz * squish);
  result += halfTorus(0.5 * width - 3.0, 3.0, xyz * squish - vec3f(0.0, 0.0, 20.0));
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (width, height, p) => {
  const baseHeight = 4;
  const separation = 3.0;
  const q = p;
  let result = cone(0.5 * width, height - baseHeight, q);
  result = result.add(
    halfTorus(0.5 * width - separation, separation, q.sub(vec3(0, 0, height - baseHeight)))
  );
  result = result.sub(cone(0.5 * width, height - baseHeight, q.sub(vec3(0, 0, separation))));

  return result;
};

const mainModel4 = (xyz) => {
  return [soapdish(105.0, 24.0, xyz.add(vec3(0, 0, 12)))];
};
```

* Try loading [soapdish-step-04.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-04.irmf) now in the IRMF editor!
* Try loading [soapdish-step-04-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-04-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-04-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-04-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-05.irmf

Now before adding all the details, let's squish the dish in the Y direction.

Note that it feels weird to _multiply_ by `width/height` (which is greater
than one) when we know we are _squishing_ the depth by `height/width`. But
the thing to remember here is that when writing shaders, it is tremendously
easier to alter the incoming coordinate space _before_ creating the objects
because it makes the math within the objects so much simpler by keeping
everything near the origin `(0,0,0)`.

![soapdish-step-05.png](soapdish-step-05.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float halfTorus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } // Just to top half.
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float soapdish(float width, float depth, float height, in vec3 xyz) {
  const float baseHeight = 4.0;
  const float separation = 3.0;
  vec3 squish = vec3(1, width / depth, 1);
  float result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3(0, 0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3(0, 0, separation));
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } 
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn base(radius: f32, baseHeight: f32, xyz: vec3f) -> f32 {
  if (xyz.z > baseHeight || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn soapdish(xyz: vec3f) -> f32 {
  let width = 105.0;
  let depth = 82.0;
  let height = 24.0;
  let baseHeight = 6.0;
  let squish = vec3f(1.0, width / depth, 1.0);
  var result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - 3.0, 3.0, xyz * squish - vec3f(0.0, 0.0, height - baseHeight));
  result += base(0.5 * width - height + baseHeight, baseHeight, xyz * squish + vec3f(0.0, 0.0, baseHeight));
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (width, depth, height, p) => {
  const baseHeight = 4;
  const separation = 3.0;
  const q = p.mul(vec3(1, width / depth, 1));
  let result = cone(0.5 * width, height - baseHeight, q);
  result = result.add(
    halfTorus(0.5 * width - separation, separation, q.sub(vec3(0, 0, height - baseHeight)))
  );
  result = result.sub(cone(0.5 * width, height - baseHeight, q.sub(vec3(0, 0, separation))));

  return result;
};

const mainModel4 = (xyz) => {
  return [soapdish(105.0, 82.0, 24.0, xyz.add(vec3(0, 0, 12)))];
};
```

* Try loading [soapdish-step-05.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-05.irmf) now in the IRMF editor!
* Try loading [soapdish-step-05-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-05-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-05-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-05-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-06.irmf

Now let's add a single little post at the center of the dish.

![soapdish-step-06.png](soapdish-step-06.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float halfTorus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } // Just to top half.
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float post(in vec3 xyz) {
  const float height = 5.0;
  const float radius = 5.0;
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius) { return 0.0; }
  return 1.0;
}

float soapdish(float width, float depth, float height, in vec3 xyz) {
  const float baseHeight = 4.0;
  const float separation = 3.0;
  vec3 squish = vec3(1, width / depth, 1);
  float result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3(0, 0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3(0, 0, separation));
  
  result += post(xyz - vec3(0, 0, separation));
  
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; }
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn post(xyz: vec3f) -> f32 {
  let height = 5.0;
  let radius = 5.0;
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn soapdish(width: f32, depth: f32, height: f32, xyz: vec3f) -> f32 {
  let baseHeight = 4.0;
  let separation = 3.0;
  let squish = vec3f(1.0, width / depth, 1.0);
  var result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3f(0.0, 0.0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3f(0.0, 0.0, separation));
  
  result += post(xyz - vec3f(0.0, 0.0, separation));
  
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (width, depth, height, p) => {
  const baseHeight = 4;
  const separation = 3.0;
  const q = p.mul(vec3(1, width / depth, 1));
  let result = cone(0.5 * width, height - baseHeight, q);
  result = result.add(
    halfTorus(0.5 * width - separation, separation, q.sub(vec3(0, 0, height - baseHeight)))
  );
  result = result.sub(cone(0.5 * width, height - baseHeight, q.sub(vec3(0, 0, separation))));

  const offset = vec3(12.5, 12.5, separation);
  result = result.add(post6(p.sub(vec3(-25, 0, separation))));
  result = result.add(post6(p.sub(vec3(25, 0, separation))));

  return result;
};

const mainModel4 = (xyz) => {
  return [soapdish(105.0, 82.0, 24.0, xyz.add(vec3(0, 0, 12)))];
};
```

* Try loading [soapdish-step-06.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-06.irmf) now in the IRMF editor!
* Try loading [soapdish-step-06-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-06-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-06-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-06-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-07.irmf

The post needs a hemispherical topper.

![soapdish-step-07.png](soapdish-step-07.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float halfTorus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } // Just to top half.
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float sphere(float radius, in vec3 xyz) {
  float r = length(xyz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

float post(in vec3 xyz) {
  const float height = 5.0;
  const float radius = 5.0;
  float result = sphere(radius, xyz - vec3(0, 0, height)); // Top the post with a sphere.
  if (xyz.z > height || xyz.z < 0.0) { return result; }
  float r = length(xyz.xy);
  if (r > radius) { return result; }
  return 1.0;
}

float soapdish(float width, float depth, float height, in vec3 xyz) {
  const float baseHeight = 4.0;
  const float separation = 3.0;
  vec3 squish = vec3(1, width / depth, 1);
  float result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3(0, 0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3(0, 0, separation));
  
  result += post(xyz - vec3(0, 0, separation));
  
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; }
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn sphere(radius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn post(xyz: vec3f) -> f32 {
  let height = 5.0;
  let radius = 5.0;
  let result = sphere(radius, xyz - vec3f(0.0, 0.0, height));
  if (xyz.z > height || xyz.z < 0.0) { return result; }
  let r = length(xyz.xy);
  if (r > radius) { return result; }
  return 1.0;
}

fn soapdish(width: f32, depth: f32, height: f32, xyz: vec3f) -> f32 {
  let baseHeight = 4.0;
  let separation = 3.0;
  let squish = vec3f(1.0, width / depth, 1.0);
  var result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3f(0.0, 0.0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3f(0.0, 0.0, separation));
  
  result += post(xyz - vec3f(0.0, 0.0, separation));
  
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (width, depth, height, p) => {
  const baseHeight = 4;
  const separation = 3.0;
  const q = p.mul(vec3(1, width / depth, 1));
  let result = cone(0.5 * width, height - baseHeight, q);
  result = result.add(
    halfTorus(0.5 * width - separation, separation, q.sub(vec3(0, 0, height - baseHeight)))
  );
  result = result.sub(cone(0.5 * width, height - baseHeight, q.sub(vec3(0, 0, separation))));

  const offset = vec3(12.5, 12.5, separation);
  result = result.add(post7(p.sub(vec3(-25, 0, separation))));
  result = result.add(post7(p.sub(vec3(25, 0, separation))));

  return result;
};

const mainModel4 = (xyz) => {
  return [soapdish(105.0, 82.0, 24.0, xyz.add(vec3(0, 0, 12)))];
};
```

* Try loading [soapdish-step-07.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-07.irmf) now in the IRMF editor!
* Try loading [soapdish-step-07-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-07-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-07-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-07-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-08.irmf

Now we need a total of 11 of these posts - a 3x3 grid and one on each end.

Normally in GLSL ES shaders, one typically does not write for-loops.
However, a small double-nested for-loop to instantiate multiple identical
objects seems like a good fit, so let's do that for this tutorial.

![soapdish-step-08.png](soapdish-step-08.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float halfTorus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } // Just to top half.
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float sphere(float radius, in vec3 xyz) {
  float r = length(xyz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

float post(in vec3 xyz) {
  const float height = 5.0;
  const float radius = 5.0;
  float result = sphere(radius, xyz - vec3(0, 0, height)); // Top the post with a sphere.
  if (xyz.z > height || xyz.z < 0.0) { return result; }
  float r = length(xyz.xy);
  if (r > radius) { return result; }
  return 1.0;
}

float soapdish(float width, float depth, float height, in vec3 xyz) {
  const float baseHeight = 4.0;
  const float separation = 3.0;
  vec3 squish = vec3(1, width / depth, 1);
  float result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3(0, 0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3(0, 0, separation));
  
  vec3 offset = vec3(12.5, 12.5, separation);
  for(int i = -1; i <= 1; i ++ ) {
    for(int j = -1; j <= 1; j ++ ) {
      result += post(xyz - offset * vec3(i, j, 1));
    }
  }
  result += post(xyz - vec3(-25, 0, separation));
  result += post(xyz - vec3(25, 0, separation));
  
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; }
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn sphere(radius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn post(xyz: vec3f) -> f32 {
  let height = 5.0;
  let radius = 5.0;
  let result = sphere(radius, xyz - vec3f(0.0, 0.0, height));
  if (xyz.z > height || xyz.z < 0.0) { return result; }
  let r = length(xyz.xy);
  if (r > radius) { return result; }
  return 1.0;
}

fn soapdish(width: f32, depth: f32, height: f32, xyz: vec3f) -> f32 {
  let baseHeight = 4.0;
  let separation = 3.0;
  let squish = vec3f(1.0, width / depth, 1.0);
  var result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3f(0.0, 0.0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3f(0.0, 0.0, separation));
  
  let offset = vec3f(12.5, 12.5, separation);
  for(var i = -1; i <= 1; i ++ ) {
    for(var j = -1; j <= 1; j ++ ) {
      result += post(xyz - offset * vec3f(f32(i), f32(j), 1.0));
    }
  }
  result += post(xyz - vec3f(-25.0, 0.0, separation));
  result += post(xyz - vec3f(25.0, 0.0, separation));
  
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (width, depth, height, p) => {
  const baseHeight = 4;
  const separation = 3.0;
  const q = p.mul(vec3(1, width / depth, 1));
  let result = cone(0.5 * width, height - baseHeight, q);
  result = result.add(
    halfTorus(0.5 * width - separation, separation, q.sub(vec3(0, 0, height - baseHeight)))
  );
  result = result.sub(cone(0.5 * width, height - baseHeight, q.sub(vec3(0, 0, separation))));

  const offset = vec3(12.5, 12.5, separation);
  for (let i = -1; i <= 1; i++) {
    for (let j = -1; j <= 1; j++) {
      result = result.add(post7(p.sub(offset.mul(vec3(i, j, 1)))));
    }
  }
  result = result.add(post7(p.sub(vec3(-25, 0, separation))));
  result = result.add(post7(p.sub(vec3(25, 0, separation))));

  return result;
};

const mainModel4 = (xyz) => {
  return [soapdish(105.0, 82.0, 24.0, xyz.add(vec3(0, 0, 12)))];
};
```

* Try loading [soapdish-step-08.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-08.irmf) now in the IRMF editor!
* Try loading [soapdish-step-08-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-08-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-08-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-08-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-09.irmf

Like the posts, we need drains. Let's model the drain and array
eight of them in one step.

![soapdish-step-09.png](soapdish-step-09.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float halfTorus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } // Just to top half.
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float sphere(float radius, in vec3 xyz) {
  float r = length(xyz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

float post(in vec3 xyz) {
  const float height = 5.0;
  const float radius = 5.0;
  float result = sphere(radius, xyz - vec3(0, 0, height)); // Top the post with a sphere.
  if (xyz.z > height || xyz.z < 0.0) { return result; }
  float r = length(xyz.xy);
  if (r > radius) { return result; }
  return 1.0;
}

float drain(in vec3 xyz) {
  float r3 = length(xyz - vec3(0, 0, 0.5));
  if (r3 <= 3.0) { return 1.0; } // Top the drain off with a sphere.
  float r = length(xyz.xy);
  if (r > 2.0) { return 0.0; }
  return 1.0;
}

float soapdish(float width, float depth, float height, in vec3 xyz) {
  const float baseHeight = 4.0;
  const float separation = 3.0;
  vec3 squish = vec3(1, width / depth, 1);
  float result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3(0, 0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3(0, 0, separation));
  
  vec3 offset = vec3(12.5, 12.5, separation);
  for(int i = -1; i <= 1; i ++ ) {
    for(int j = -1; j <= 1; j ++ ) {
      result += post(xyz - offset * vec3(i, j, 1));
    }
  }
  result += post(xyz - vec3(-25, 0, separation));
  result += post(xyz - vec3(25, 0, separation));
  
  for(int i = -2; i < 2; i ++ ) {
    result -= drain(xyz - offset * vec3(i, - 1, 1) - vec3(6.25, 6.25, 0));
    result -= drain(xyz - offset * vec3(i, 0, 1) - vec3(6.25, 6.25, 0));
  }
  
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Add 12 to the Z value to center the object vertically.
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3(0, 0, 12));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; }
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn sphere(radius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn post(xyz: vec3f) -> f32 {
  let height = 5.0;
  let radius = 5.0;
  let result = sphere(radius, xyz - vec3f(0.0, 0.0, height));
  if (xyz.z > height || xyz.z < 0.0) { return result; }
  let r = length(xyz.xy);
  if (r > radius) { return result; }
  return 1.0;
}

fn drain(xyz: vec3f) -> f32 {
  let r3 = length(xyz - vec3f(0.0, 0.0, 0.5));
  if (r3 <= 3.0) { return 1.0; }
  let r = length(xyz.xy);
  if (r > 2.0) { return 0.0; }
  return 1.0;
}

fn soapdish(width: f32, depth: f32, height: f32, xyz: vec3f) -> f32 {
  let baseHeight = 4.0;
  let separation = 3.0;
  let squish = vec3f(1.0, width / depth, 1.0);
  var result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3f(0.0, 0.0, height - baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3f(0.0, 0.0, separation));
  
  let offset = vec3f(12.5, 12.5, separation);
  for(var i = -1; i <= 1; i ++ ) {
    for(var j = -1; j <= 1; j ++ ) {
      result += post(xyz - offset * vec3f(f32(i), f32(j), 1.0));
    }
  }
  result += post(xyz - vec3f(-25.0, 0.0, separation));
  result += post(xyz - vec3f(25.0, 0.0, separation));
  
  for(var i = -2; i < 2; i ++ ) {
    result -= drain(xyz - offset * vec3f(f32(i), - 1.0, 1.0) - vec3f(6.25, 6.25, 0.0));
    result -= drain(xyz - offset * vec3f(f32(i), 0.0, 1.0) - vec3f(6.25, 6.25, 0.0));
  }
  
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3f(0.0, 0.0, 12.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [52.5,52.5,12],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (width, depth, height, p) => {
  const baseHeight = 4;
  const separation = 3.0;
  const q = p.mul(vec3(1, width / depth, 1));
  let result = cone(0.5 * width, height - baseHeight, q);
  result = result.add(
    halfTorus(0.5 * width - separation, separation, q.sub(vec3(0, 0, height - baseHeight)))
  );
  result = result.sub(cone(0.5 * width, height - baseHeight, q.sub(vec3(0, 0, separation))));

  const offset = vec3(12.5, 12.5, separation);
  for (let i = -1; i <= 1; i++) {
    for (let j = -1; j <= 1; j++) {
      result = result.add(post7(p.sub(offset.mul(vec3(i, j, 1)))));
    }
  }
  result = result.add(post7(p.sub(vec3(-25, 0, separation))));
  result = result.add(post7(p.sub(vec3(25, 0, separation))));

  for (let i = -2; i < 2; i++) {
    result = result.sub(drain(p.sub(offset.mul(vec3(i, -1, 1))).sub(vec3(6.25, 6.25, 0))));
    result = result.sub(drain(p.sub(offset.mul(vec3(i, 0, 1))).sub(vec3(6.25, 6.25, 0))));
  }

  return result;
};

const mainModel4 = (xyz) => {
  return [soapdish(105.0, 82.0, 24.0, xyz.add(vec3(0, 0, 12)))];
};
```

* Try loading [soapdish-step-09.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-09.irmf) now in the IRMF editor!
* Try loading [soapdish-step-09-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-09-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-09-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-09-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

## soapdish-step-10.irmf

Now we need to focus on the base to let the water drain out.

It's really a squished cylinder that can be squished with the rest of
the bowl shape, so let's do that.

We'll go ahead and add the cylindrical drain cutouts for the base at
the same time.

For some reason, I had to tweak the MBB settings, but here is the final
model.

![soapdish-step-10.png](soapdish-step-10.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["PLA1"],
  max: [52.5,52.5,18],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

#define M_PI 3.1415926535897932384626433832795

float halfTorus(float majorRadius, float minorRadius, in vec3 xyz) {
  float r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; } // Just to top half.
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  float angle = atan(xyz.y, xyz.x);
  vec3 center = vec3(majorRadius * cos(angle), majorRadius * sin(angle), 0);
  vec3 v = xyz - center;
  float r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

float cone(float radius, float height, in vec3 xyz) {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

float sphere(float radius, in vec3 xyz) {
  float r = length(xyz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

float post(in vec3 xyz) {
  const float height = 5.0;
  const float radius = 5.0;
  float result = sphere(radius, xyz - vec3(0, 0, height)); // Top the post with a sphere.
  if (xyz.z > height || xyz.z < 0.0) { return result; }
  float r = length(xyz.xy);
  if (r > radius) { return result; }
  return 1.0;
}

float drain(in vec3 xyz) {
  float r3 = length(xyz - vec3(0, 0, 0.5));
  if (r3 <= 3.0) { return 1.0; } // Top the drain off with a sphere.
  float r = length(xyz.xy);
  if (r > 2.0) { return 0.0; }
  return 1.0;
}

float base(float radius, float baseHeight, in vec3 xyz) {
  if (xyz.z > baseHeight || xyz.z < 0.0) { return 0.0; }
  float r = length(xyz.xy);
  if (r > radius) { return 0.0; }
  return 1.0;
}

float baseDrain(float radius, in vec3 xyz) {
  float r = length(xyz.xz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

float soapdish(float width, float depth, float height, in vec3 xyz) {
  const float baseHeight = 6.0;
  const float separation = 3.0;
  vec3 squish = vec3(1, width / depth, 1);
  float result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3(0, 0, height - baseHeight));
  result += base(0.5 * width - height + baseHeight, baseHeight, xyz * squish + vec3(0, 0, baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3(0, 0, separation));
  
  vec3 offset = vec3(12.5, 12.5, separation);
  for(int i = -1; i <= 1; i ++ ) {
    for(int j = -1; j <= 1; j ++ ) {
      result += post(xyz - offset * vec3(i, j, 1));
    }
  }
  result += post(xyz - vec3(-25, 0, separation));
  result += post(xyz - vec3(25, 0, separation));
  
  for(int i = -2; i < 2; i ++ ) {
    result -= drain(xyz - offset * vec3(i, - 1, 1) - vec3(6.25, 6.25, 0));
    result -= drain(xyz - offset * vec3(i, 0, 1) - vec3(6.25, 6.25, 0));
    result -= baseDrain(4.0, xyz - offset * vec3(i, 0, 0) + vec3(-6.25, 0, baseHeight));
  }
  
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  // Center the object vertically.
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3(0, 0, 6));
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["PLA1"],
  max: [52.5,52.5,18],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/

const M_PI = 3.1415926535897932384626433832795;

fn halfTorus(majorRadius: f32, minorRadius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (xyz.z > minorRadius || xyz.z < 0.0) { return 0.0; }
  if (r > majorRadius + minorRadius || r < majorRadius - minorRadius) { return 0.0; }
  
  let angle = atan2(xyz.y, xyz.x);
  let center = vec3f(majorRadius * cos(angle), majorRadius * sin(angle), 0.0);
  let v = xyz - center;
  let r2 = length(v);
  if (r2 > minorRadius) { return 0.0; }
  
  return 1.0;
}

fn cone(radius: f32, height: f32, xyz: vec3f) -> f32 {
  if (xyz.z > height || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius - (height - xyz.z)) { return 0.0; }
  return 1.0;
}

fn sphere(radius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn post(xyz: vec3f) -> f32 {
  let height = 5.0;
  let radius = 5.0;
  let result = sphere(radius, xyz - vec3f(0.0, 0.0, height));
  if (xyz.z > height || xyz.z < 0.0) { return result; }
  let r = length(xyz.xy);
  if (r > radius) { return result; }
  return 1.0;
}

fn drain(xyz: vec3f) -> f32 {
  let r3 = length(xyz - vec3f(0.0, 0.0, 0.5));
  if (r3 <= 3.0) { return 1.0; }
  let r = length(xyz.xy);
  if (r > 2.0) { return 0.0; }
  return 1.0;
}

fn base(radius: f32, baseHeight: f32, xyz: vec3f) -> f32 {
  if (xyz.z > baseHeight || xyz.z < 0.0) { return 0.0; }
  let r = length(xyz.xy);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn baseDrain(radius: f32, xyz: vec3f) -> f32 {
  let r = length(xyz.xz);
  if (r > radius) { return 0.0; }
  return 1.0;
}

fn soapdish(width: f32, depth: f32, height: f32, xyz: vec3f) -> f32 {
  let baseHeight = 6.0;
  let separation = 3.0;
  let squish = vec3f(1.0, width / depth, 1.0);
  var result = cone(0.5 * width, height - baseHeight, xyz * squish);
  result += halfTorus(0.5 * width - separation, separation, xyz * squish - vec3f(0.0, 0.0, height - baseHeight));
  result += base(0.5 * width - height + baseHeight, baseHeight, xyz * squish + vec3f(0.0, 0.0, baseHeight));
  result -= cone(0.5 * width, height - baseHeight, xyz * squish - vec3f(0.0, 0.0, separation));
  
  let offset = vec3f(12.5, 12.5, separation);
  for(var i = -1; i <= 1; i ++ ) {
    for(var j = -1; j <= 1; j ++ ) {
      result += post(xyz - offset * vec3f(f32(i), f32(j), 1.0));
    }
  }
  result += post(xyz - vec3f(-25.0, 0.0, separation));
  result += post(xyz - vec3f(25.0, 0.0, separation));
  
  for(var i = -2; i < 2; i ++ ) {
    result -= drain(xyz - offset * vec3f(f32(i), - 1.0, 1.0) - vec3f(6.25, 6.25, 0.0));
    result -= drain(xyz - offset * vec3f(f32(i), 0.0, 1.0) - vec3f(6.25, 6.25, 0.0));
    result -= baseDrain(4.0, xyz - offset * vec3f(f32(i), 0.0, 0.0) + vec3f(-6.25, 0.0, baseHeight));
  }
  
  return result;
}

fn mainModel4(xyz: vec3f) -> vec4f {
  var materials = vec4f(0.0);
  materials[0] = soapdish(105.0, 82.0, 24.0, xyz + vec3f(0.0, 0.0, 6.0));
  return materials;
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["PLA1"],
  max: [52.5,52.5,18],
  min: [-52.5,-52.5,-12],
  units: "mm",
}*/


const cone = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(float(radius).sub(float(height).sub(p.z)));
  return select(zOK.and(rOK), float(1.0), float(0.0));
};

const halfTorus = (majorRadius, minorRadius, p) => {
  const r = length(p);
  const ringOK = r.lessThanEqual(majorRadius + minorRadius).and(
    r.greaterThanEqual(majorRadius - minorRadius)
  );
  const angle = atan(p.y, p.x);
  const center = vec3(cos(angle).mul(majorRadius), sin(angle).mul(majorRadius), float(0));
  const r2 = length(p.sub(center));
  const zOK = p.z.lessThanEqual(minorRadius).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(ringOK).and(r2.lessThanEqual(minorRadius)), float(1.0), float(0.0));
};

const sphere = (radius, p) => {
  return select(length(p).lessThanEqual(radius), float(1.0), float(0.0));
};

const cylinderZ = (radius, height, p) => {
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const post6 = (p) => cylinderZ(5.0, 5.0, p);

const post7 = (p) => {
  const height = 5.0;
  const radius = 5.0;
  const cap = sphere(radius, p.sub(vec3(0, 0, height))); // Top the post with a sphere.
  const zOK = p.z.lessThanEqual(height).and(p.z.greaterThanEqual(0.0));
  const rOK = length(p.xy).lessThanEqual(radius);
  return select(zOK.and(rOK), float(1.0), cap);
};

const drain = (p) => {
  const top = length(p.sub(vec3(0, 0, 0.5))).lessThanEqual(3.0); // Top the drain off with a sphere.
  return select(top.or(length(p.xy).lessThanEqual(2.0)), float(1.0), float(0.0));
};

const base = (radius, baseHeight, p) => {
  const zOK = p.z.lessThanEqual(baseHeight).and(p.z.greaterThanEqual(0.0));
  return select(zOK.and(length(p.xy).lessThanEqual(radius)), float(1.0), float(0.0));
};

const baseDrain = (radius, p) => {
  return select(length(p.xz).lessThanEqual(radius), float(1.0), float(0.0));
};

const soapdish = (width, depth, height, p) => {
  const baseHeight = 6;
  const separation = 3.0;
  const q = p.mul(vec3(1, width / depth, 1));
  let result = cone(0.5 * width, height - baseHeight, q);
  result = result.add(
    halfTorus(0.5 * width - separation, separation, q.sub(vec3(0, 0, height - baseHeight)))
  );
  result = result.add(base(0.5 * width - height + baseHeight, baseHeight, q.add(vec3(0, 0, baseHeight))));
  result = result.sub(cone(0.5 * width, height - baseHeight, q.sub(vec3(0, 0, separation))));

  const offset = vec3(12.5, 12.5, separation);
  for (let i = -1; i <= 1; i++) {
    for (let j = -1; j <= 1; j++) {
      result = result.add(post7(p.sub(offset.mul(vec3(i, j, 1)))));
    }
  }
  result = result.add(post7(p.sub(vec3(-25, 0, separation))));
  result = result.add(post7(p.sub(vec3(25, 0, separation))));

  for (let i = -2; i < 2; i++) {
    result = result.sub(drain(p.sub(offset.mul(vec3(i, -1, 1))).sub(vec3(6.25, 6.25, 0))));
    result = result.sub(drain(p.sub(offset.mul(vec3(i, 0, 1))).sub(vec3(6.25, 6.25, 0))));
    result = result.sub(baseDrain(4.0, p.sub(offset.mul(vec3(i, 0, 0))).add(vec3(-6.25, 0, baseHeight))));
  }

  return result;
};

const mainModel4 = (xyz) => {
  return [soapdish(105.0, 82.0, 24.0, xyz.add(vec3(0, 0, 6)))];
};
```

* Try loading [soapdish-step-10.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-10.irmf) now in the IRMF editor!
* Try loading [soapdish-step-10-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-10-wgsl.irmf) now in the IRMF editor!
* Try loading [soapdish-step-10-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/015-soapdish/soapdish-step-10-tsl.irmf) now in the IRMF editor!

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
