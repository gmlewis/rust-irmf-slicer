# 036-utah-teapot

The Utah Teapot has an [amazing history](https://graphics.cs.utah.edu/teapot/).

I downloaded "teapot3.scl" and "teapot-bezier.scl" from the [Alpha_1 Model Repository](
https://www-old.cs.utah.edu/gdc/projects/alpha1/help/man/html/model_repo/model_teapot/model_teapot.html)
and then converted it to IRMF using Google's "gemini-3-flash-preview" model
within the [Google Gemini CLI](https://geminicli.com/).

Here is the solid version of the Utah Teapot ("teapot3.scl") written in IRMF:

## utah-teapot-wgsl.irmf

![utah-teapot.png](https://raw.githubusercontent.com/gmlewis/rust-irmf-slicer/master/examples/assets/036-utah-teapot/utah-teapot.png)

**GLSL**

```glsl
/*{
  irmf: "1.0",
  language: "glsl",
  materials: ["porcelain"],
  min: [-3.3,-2.1,0],
  max: [3.6,2.1,3.3],
  units: "mm",
}*/

precision highp float;

// Cubic Bezier evaluation
vec2 bezier(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t) {
    float u = 1.0 - t;
    return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}

// Ray-casting intersection for 2D profile
int intersectBezier(vec2 p, vec2 p0, vec2 p1, vec2 p2, vec2 p3) {
    int count = 0;
    vec2 prev = p0;
    const int samples = 20;
    for (int i = 1; i <= samples; i++) {
        vec2 curr = bezier(p0, p1, p2, p3, float(i) / float(samples));
        if (((prev.y <= p.y && p.y < curr.y) || (curr.y <= p.y && p.y < prev.y))) {
            float x = prev.x + (curr.x - prev.x) * (p.y - prev.y) / (curr.y - prev.y);
            if (x > p.x) count++;
        }
        prev = curr;
    }
    return count;
}

int intersectLine(vec2 p, vec2 p0, vec2 p1) {
    if (((p0.y <= p.y && p.y < p1.y) || (p1.y <= p.y && p.y < p0.y))) {
        float x = p0.x + (p1.x - p0.x) * (p.y - p0.y) / (p1.y - p0.y);
        if (x > p.x) return 1;
    }
    return 0;
}

// B-Spline evaluation (clamped cubic)
vec2 bspline(vec2 p[9], float t) {
    float mt = t * 6.0;
    int seg = int(floor(mt));
    if (seg < 0) seg = 0; if (seg > 5) seg = 5;
    float f = mt - float(seg);
    if (seg == 0) return bezier(p[0], p[1], p[2], (p[2]+p[3])*0.5, f);
    if (seg == 5) return bezier((p[5]+p[6])*0.5, p[6], p[7], p[8], f);
    vec2 c0 = (p[seg]+p[seg+1])*0.5; vec2 c1 = p[seg+1]; vec2 c2 = p[seg+2]; vec2 c3 = (p[seg+2]+p[seg+3])*0.5;
    return bezier(c0, c1, c2, c3, f);
}

int intersectBSpline(vec2 p, vec2 ctrl[9]) {
    int count = 0;
    vec2 prev = ctrl[0];
    const int samples = 40;
    for (int i = 1; i <= samples; i++) {
        vec2 curr = bspline(ctrl, float(i) / float(samples));
        if (((prev.y <= p.y && p.y < curr.y) || (curr.y <= p.y && p.y < prev.y))) {
            float x = prev.x + (curr.x - prev.x) * (p.y - prev.y) / (curr.y - prev.y);
            if (x > p.x) count++;
        }
        prev = curr;
    }
    return count;
}

vec2 bsplineLid(vec2 p[12], float t) {
    float mt = t * 9.0;
    int seg = int(floor(mt));
    if (seg < 0) seg = 0; if (seg > 8) seg = 8;
    float f = mt - float(seg);
    if (seg == 0) return bezier(p[0], p[1], p[2], (p[2]+p[3])*0.5, f);
    if (seg == 8) return bezier((p[8]+p[9])*0.5, p[9], p[10], p[11], f);
    vec2 c0 = (p[seg]+p[seg+1])*0.5; vec2 c1 = p[seg+1]; vec2 c2 = p[seg+2]; vec2 c3 = (p[seg+2]+p[seg+3])*0.5;
    return bezier(c0, c1, c2, c3, f);
}

int intersectBSplineLid(vec2 p, vec2 ctrl[12]) {
    int count = 0;
    vec2 prev = ctrl[0];
    const int samples = 50;
    for (int i = 1; i <= samples; i++) {
        vec2 curr = bsplineLid(ctrl, float(i) / float(samples));
        if (((prev.y <= p.y && p.y < curr.y) || (curr.y <= p.y && p.y < prev.y))) {
            float x = prev.x + (curr.x - prev.x) * (p.y - prev.y) / (curr.y - prev.y);
            if (x > p.x) count++;
        }
        prev = curr;
    }
    return count;
}

bool isInsideBody(vec3 xyz, bool outside) {
    float r = length(xyz.xz);
    vec2 p = vec2(r, xyz.y);
    int count = 0;
    if (outside) {
        count += intersectBezier(p, vec2(1.4, 2.25), vec2(1.3375, 2.38125), vec2(1.4375, 2.38125), vec2(1.5, 2.25));
        count += intersectBezier(p, vec2(1.5, 2.25), vec2(1.75, 1.725), vec2(2.0, 1.2), vec2(2.0, 0.75));
        count += intersectBezier(p, vec2(2.0, 0.75), vec2(2.0, 0.3), vec2(1.5, 0.075), vec2(1.5, 0.0));
        count += intersectLine(p, vec2(1.5, 0.0), vec2(0.0, 0.0));
        count += intersectLine(p, vec2(0.0, 0.0), vec2(0.0, 2.25));
        count += intersectLine(p, vec2(0.0, 2.25), vec2(1.4, 2.25));
    } else {
        vec2 ctrl[9];
        ctrl[0] = vec2(1.4, 2.25); ctrl[1] = vec2(1.3, 2.25); ctrl[2] = vec2(1.2, 2.25);
        ctrl[3] = vec2(1.2, 2.15); ctrl[4] = vec2(1.5, 2.12); ctrl[5] = vec2(2.0, 0.6);
        ctrl[6] = vec2(1.4, 0.2); ctrl[7] = vec2(1.3, 0.15); ctrl[8] = vec2(1.1, 0.15);
        count += intersectBSpline(p, ctrl);
        count += intersectLine(p, vec2(1.1, 0.15), vec2(0.0, 0.15));
        count += intersectLine(p, vec2(0.0, 0.15), vec2(0.0, 2.25));
        count += intersectLine(p, vec2(0.0, 2.25), vec2(1.4, 2.25));
    }
    return (count % 2) != 0;
}

bool isInsideLid(vec3 xyz, bool outside) {
    float r = length(xyz.xz);
    vec2 p = vec2(r, xyz.y);
    int count = 0;
    if (outside) {
        count += intersectBezier(p, vec2(0.01, 3.0), vec2(0.8, 3.0), vec2(0.0, 2.7), vec2(0.2, 2.55));
        count += intersectBezier(p, vec2(0.2, 2.55), vec2(0.4, 2.4), vec2(1.3, 2.4), vec2(1.3, 2.25));
        count += intersectLine(p, vec2(1.3, 2.25), vec2(0.0, 2.25));
        count += intersectLine(p, vec2(0.0, 2.25), vec2(0.0, 3.0));
        count += intersectLine(p, vec2(0.0, 3.0), vec2(0.01, 3.0));
    } else {
        vec2 ctrl[12];
        ctrl[0] = vec2(1.3, 2.25); ctrl[1] = vec2(1.2, 2.25); ctrl[2] = vec2(1.18, 2.25);
        ctrl[3] = vec2(1.15, 2.15); ctrl[4] = vec2(1.1, 2.2); ctrl[5] = vec2(1.0, 2.32);
        ctrl[6] = vec2(0.5, 2.35); ctrl[7] = vec2(0.07, 2.48); ctrl[8] = vec2(0.07, 2.66);
        ctrl[9] = vec2(0.26, 2.87); ctrl[10] = vec2(0.2, 2.91); ctrl[11] = vec2(0.05, 2.91);
        count += intersectBSplineLid(p, ctrl);
        count += intersectLine(p, vec2(0.05, 2.91), vec2(0.0, 2.91));
        count += intersectLine(p, vec2(0.0, 2.91), vec2(0.0, 2.25));
        count += intersectLine(p, vec2(0.0, 2.25), vec2(1.3, 2.25));
    }
    return (count % 2) != 0;
}

float projectToBezier(vec2 q, vec2 p0, vec2 p1, vec2 p2, vec2 p3) {
    float bestT = 0.5; float minDist = 1e10;
    for (int i = 0; i <= 10; i++) {
        float t = float(i) / 10.0;
        float d = length(q - bezier(p0, p1, p2, p3, t));
        if (d < minDist) { minDist = d; bestT = t; }
    }
    float delta = 0.05;
    for (int j = 0; j < 3; j++) {
        float t1 = clamp(bestT - delta, 0.0, 1.0); float t2 = clamp(bestT + delta, 0.0, 1.0);
        float d1 = length(q - bezier(p0, p1, p2, p3, t1)); float d2 = length(q - bezier(p0, p1, p2, p3, t2));
        if (d1 < d2) { if (d1 < minDist) { minDist = d1; bestT = t1; } }
        else { if (d2 < minDist) { minDist = d2; bestT = t2; } }
        delta *= 0.5;
    }
    return bestT;
}

bool isInsideHandle(vec3 xyz) {
    // Average path of Row 0 and Row 3 from Handle patches
    vec2 p0 = vec2(-1.55, 1.9875); vec2 p1 = vec2(-2.4, 1.9875); vec2 p2 = vec2(-2.85, 1.9875); vec2 p3 = vec2(-2.85, 1.65);
    vec2 p4 = vec2(-2.85, 1.3125); vec2 p5 = vec2(-2.575, 0.88125); vec2 p6 = vec2(-1.95, 0.6);
    float t1 = projectToBezier(xyz.xy, p0, p1, p2, p3); float d1 = length(xyz.xy - bezier(p0, p1, p2, p3, t1));
    float t2 = projectToBezier(xyz.xy, p3, p4, p5, p6); float d2 = length(xyz.xy - bezier(p3, p4, p5, p6, t2));
    float d = min(d1, d2);
    float r_xy = 0.15; float r_z = 0.225; // Based on control points
    return (d*d)/(r_xy*r_xy) + (xyz.z*xyz.z)/(r_z*r_z) < 1.0;
}

float bezier1D(float p0, float p1, float p2, float p3, float t) {
    float u = 1.0 - t;
    return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}

float distToSegment(vec2 p, vec2 a, vec2 b, out float w) {
    vec2 ba = b - a;
    float l2 = dot(ba, ba);
    if (l2 < 1e-6) { w = 0.0; return length(p - a); }
    w = clamp(dot(p - a, ba) / l2, 0.0, 1.0);
    return length(p - (a + w * ba));
}

bool testHump(vec3 p, vec2 q0[4], vec2 q3[4], float zc[4], float xyScale, float zScale) {
    float bestV = 0.5; float minDist = 1e10; float bestW = 0.0;
    for (int i = 0; i <= 24; i++) {
        float v = float(i) / 24.0;
        vec2 A = bezier(q0[0], q0[1], q0[2], q0[3], v);
        vec2 B = bezier(q3[0], q3[1], q3[2], q3[3], v);
        if (xyScale != 1.0) {
            vec2 C = (A + B) * 0.5;
            A = C + (A - C) * xyScale;
            B = C + (B - C) * xyScale;
        }
        float w; float d = distToSegment(p.xy, A, B, w);
        if (d < minDist) { minDist = d; bestV = v; bestW = w; }
    }
    float delta = 1.0 / 48.0;
    for (int j = 0; j < 3; j++) {
        float v1 = clamp(bestV - delta, 0.0, 1.0); float v2 = clamp(bestV + delta, 0.0, 1.0);
        vec2 A1 = bezier(q0[0], q0[1], q0[2], q0[3], v1); vec2 B1 = bezier(q3[0], q3[1], q3[2], q3[3], v1);
        vec2 A2 = bezier(q0[0], q0[1], q0[2], q0[3], v2); vec2 B2 = bezier(q3[0], q3[1], q3[2], q3[3], v2);
        if (xyScale != 1.0) {
            vec2 C1 = (A1 + B1) * 0.5; A1 = C1 + (A1 - C1) * xyScale; B1 = C1 + (B1 - C1) * xyScale;
            vec2 C2 = (A2 + B2) * 0.5; A2 = C2 + (A2 - C2) * xyScale; B2 = C2 + (B2 - C2) * xyScale;
        }
        float w1, w2; float d1 = distToSegment(p.xy, A1, B1, w1); float d2 = distToSegment(p.xy, A2, B2, w2);
        if (d1 < d2) { if (d1 < minDist) { minDist = d1; bestV = v1; bestW = w1; } }
        else { if (d2 < minDist) { minDist = d2; bestV = v2; bestW = w2; } }
        delta *= 0.5;
    }
    if (minDist > 0.03) return false;
    float Zv = bezier1D(zc[0], zc[1], zc[2], zc[3], bestV) * zScale;
    float u = 0.5 + sin(asin(clamp(2.0*bestW - 1.0, -1.0, 1.0)) / 3.0);
    return abs(p.z) < Zv * 3.0 * u * (1.0 - u);
}

bool isInsideSpout(vec3 xyz, bool outside) {
    vec2 b0[4]; b0[0]=vec2(1.7, 1.275); b0[1]=vec2(2.6, 1.275); b0[2]=vec2(2.3, 1.95); b0[3]=vec2(2.7, 2.25);
    vec2 b3[4]; b3[0]=vec2(1.7, 0.45); b3[1]=vec2(3.1, 0.675); b3[2]=vec2(2.4, 1.875); b3[3]=vec2(3.3, 2.25);
    float bz[4]; bz[0]=0.66; bz[1]=0.66; bz[2]=0.25; bz[3]=0.25;

    if (!outside) {
        return testHump(xyz, b0, b3, bz, 0.666667, 0.6);
    }

    if (testHump(xyz, b0, b3, bz, 1.0, 1.0)) return true;

    vec2 t0[4]; t0[0]=vec2(2.7, 2.25); t0[1]=vec2(2.8, 2.325); t0[2]=vec2(2.9, 2.325); t0[3]=vec2(2.8, 2.25);
    vec2 t3[4]; t3[0]=vec2(3.3, 2.25); t3[1]=vec2(3.525, 2.34375); t3[2]=vec2(3.45, 2.3625); t3[3]=vec2(3.2, 2.25);
    float tz[4]; tz[0]=0.25; tz[1]=0.25; tz[2]=0.15; tz[3]=0.15;

    return testHump(xyz, t0, t3, tz, 1.0, 1.0);
}

void mainModel4(out vec4 materials, in vec3 pos) {
    vec3 xyz = pos.xzy;
    bool spoutSolid = isInsideSpout(xyz, true) && !isInsideSpout(xyz, false);
    bool bodySolid = isInsideBody(xyz, true) && !isInsideBody(xyz, false);
    bool lidSolid = isInsideLid(xyz, true) && !isInsideLid(xyz, false);
    bool handleSolid = isInsideHandle(xyz);
    
    materials = vec4(0.0);
    if (bodySolid || spoutSolid || lidSolid || handleSolid) materials[0] = 1.0;
}
```

**WGSL**

```wgsl
/*{
  irmf: "1.0",
  language: "wgsl",
  materials: ["porcelain"],
  min: [-3.3,-2.1,0],
  max: [3.6,2.1,3.3],
  units: "mm",
}*/

fn bezier(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let u = 1.0 - t;
    return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}

fn intersectBezier(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> i32 {
    var count: i32 = 0; var prev = p0; let samples: f32 = 20.0;
    for (var i: i32 = 1; i <= 20; i = i + 1) {
        let curr = bezier(p0, p1, p2, p3, f32(i) / samples);
        if (((prev.y <= p.y && p.y < curr.y) || (curr.y <= p.y && p.y < prev.y))) {
            let x = prev.x + (curr.x - prev.x) * (p.y - prev.y) / (curr.y - prev.y);
            if (x > p.x) { count = count + 1; }
        }
        prev = curr;
    }
    return count;
}

fn intersectLine(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>) -> i32 {
    if (((p0.y <= p.y && p.y < p1.y) || (p1.y <= p.y && p.y < p0.y))) {
        let x = p0.x + (p1.x - p0.x) * (p.y - p0.y) / (p1.y - p0.y);
        if (x > p.x) { return 1; }
    }
    return 0;
}

fn bspline(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, p4: vec2<f32>, p5: vec2<f32>, p6: vec2<f32>, p7: vec2<f32>, p8: vec2<f32>, t: f32) -> vec2<f32> {
    let mt = t * 6.0; let seg = i32(floor(mt)); let f = mt - f32(seg);
    if (seg == 0) { return bezier(p0, p1, p2, (p2+p3)*0.5, f); }
    if (seg == 5) { return bezier((p5+p6)*0.5, p6, p7, p8, f); }
    var pts = array<vec2<f32>, 9>(p0, p1, p2, p3, p4, p5, p6, p7, p8);
    let c0 = (pts[seg]+pts[seg+1])*0.5; let c1 = pts[seg+1]; let c2 = pts[seg+2]; let c3 = (pts[seg+2]+pts[seg+3])*0.5;
    return bezier(c0, c1, c2, c3, f);
}

fn intersectBSpline(p: vec2<f32>) -> i32 {
    var count: i32 = 0;
    let p0 = vec2<f32>(1.4, 2.25); let p1 = vec2<f32>(1.3, 2.25); let p2 = vec2<f32>(1.2, 2.25);
    let p3 = vec2<f32>(1.2, 2.15); let p4 = vec2<f32>(1.5, 2.12); let p5 = vec2<f32>(2.0, 0.6);
    let p6 = vec2<f32>(1.4, 0.2); let p7 = vec2<f32>(1.3, 0.15); let p8 = vec2<f32>(1.1, 0.15);
    var prev = p0; let samples: f32 = 40.0;
    for (var i: i32 = 1; i <= 40; i = i + 1) {
        let curr = bspline(p0, p1, p2, p3, p4, p5, p6, p7, p8, f32(i) / samples);
        if (((prev.y <= p.y && p.y < curr.y) || (curr.y <= p.y && p.y < prev.y))) {
            let x = prev.x + (curr.x - prev.x) * (p.y - prev.y) / (curr.y - prev.y);
            if (x > p.x) { count = count + 1; }
        }
        prev = curr;
    }
    return count;
}

fn bsplineLid(t: f32) -> vec2<f32> {
    let p0 = vec2<f32>(1.3, 2.25); let p1 = vec2<f32>(1.2, 2.25); let p2 = vec2<f32>(1.18, 2.25);
    let p3 = vec2<f32>(1.15, 2.15); let p4 = vec2<f32>(1.1, 2.2); let p5 = vec2<f32>(1.0, 2.32);
    let p6 = vec2<f32>(0.5, 2.35); let p7 = vec2<f32>(0.07, 2.48); let p8 = vec2<f32>(0.07, 2.66);
    let p9 = vec2<f32>(0.26, 2.87); let p10 = vec2<f32>(0.2, 2.91); let p11 = vec2<f32>(0.05, 2.91);
    let mt = t * 9.0; let seg = i32(floor(mt)); let f = mt - f32(seg);
    if (seg == 0) { return bezier(p0, p1, p2, (p2+p3)*0.5, f); }
    if (seg == 8) { return bezier((p8+p9)*0.5, p9, p10, p11, f); }
    var pts = array<vec2<f32>, 12>(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
    let c0 = (pts[seg]+pts[seg+1])*0.5; let c1 = pts[seg+1]; let c2 = pts[seg+2]; let c3 = (pts[seg+2]+pts[seg+3])*0.5;
    return bezier(c0, c1, c2, c3, f);
}

fn intersectBSplineLid(p: vec2<f32>) -> i32 {
    var count: i32 = 0; var prev = vec2<f32>(1.3, 2.25); let samples: f32 = 50.0;
    for (var i: i32 = 1; i <= 50; i = i + 1) {
        let curr = bsplineLid(f32(i) / samples);
        if (((prev.y <= p.y && p.y < curr.y) || (curr.y <= p.y && p.y < prev.y))) {
            let x = prev.x + (curr.x - prev.x) * (p.y - prev.y) / (curr.y - prev.y);
            if (x > p.x) { count = count + 1; }
        }
        prev = curr;
    }
    return count;
}

fn isInsideBody(xyz: vec3<f32>, outside: bool) -> bool {
    let r = length(xyz.xz); let p = vec2<f32>(r, xyz.y); var count: i32 = 0;
    if (outside) {
        count = count + intersectBezier(p, vec2<f32>(1.4, 2.25), vec2<f32>(1.3375, 2.38125), vec2<f32>(1.4375, 2.38125), vec2<f32>(1.5, 2.25));
        count = count + intersectBezier(p, vec2<f32>(1.5, 2.25), vec2<f32>(1.75, 1.725), vec2<f32>(2.0, 1.2), vec2<f32>(2.0, 0.75));
        count = count + intersectBezier(p, vec2<f32>(2.0, 0.75), vec2<f32>(2.0, 0.3), vec2<f32>(1.5, 0.075), vec2<f32>(1.5, 0.0));
        count = count + intersectLine(p, vec2<f32>(1.5, 0.0), vec2<f32>(0.0, 0.0));
        count = count + intersectLine(p, vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 2.25));
        count = count + intersectLine(p, vec2<f32>(0.0, 2.25), vec2<f32>(1.4, 2.25));
    } else {
        count = count + intersectBSpline(p);
        count = count + intersectLine(p, vec2<f32>(1.1, 0.15), vec2<f32>(0.0, 0.15));
        count = count + intersectLine(p, vec2<f32>(0.0, 0.15), vec2<f32>(0.0, 2.25));
        count = count + intersectLine(p, vec2<f32>(0.0, 2.25), vec2<f32>(1.4, 2.25));
    }
    return (count % 2) != 0;
}

fn isInsideLid(xyz: vec3<f32>, outside: bool) -> bool {
    let r = length(xyz.xz); let p = vec2<f32>(r, xyz.y); var count: i32 = 0;
    if (outside) {
        count = count + intersectBezier(p, vec2<f32>(0.01, 3.0), vec2<f32>(0.8, 3.0), vec2<f32>(0.0, 2.7), vec2<f32>(0.2, 2.55));
        count = count + intersectBezier(p, vec2<f32>(0.2, 2.55), vec2<f32>(0.4, 2.4), vec2<f32>(1.3, 2.4), vec2<f32>(1.3, 2.25));
        count = count + intersectLine(p, vec2<f32>(1.3, 2.25), vec2<f32>(0.0, 2.25));
        count = count + intersectLine(p, vec2<f32>(0.0, 2.25), vec2<f32>(0.0, 3.0));
        count = count + intersectLine(p, vec2<f32>(0.0, 3.0), vec2<f32>(0.01, 3.0));
    } else {
        count = count + intersectBSplineLid(p);
        count = count + intersectLine(p, vec2<f32>(0.05, 2.91), vec2<f32>(0.0, 2.91));
        count = count + intersectLine(p, vec2<f32>(0.0, 2.91), vec2<f32>(0.0, 2.25));
        count = count + intersectLine(p, vec2<f32>(0.0, 2.25), vec2<f32>(1.3, 2.25));
    }
    return (count % 2) != 0;
}

fn projectToBezier(q: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> f32 {
    var bestT: f32 = 0.5; var minDist: f32 = 1e10;
    for (var i: i32 = 0; i <= 10; i = i + 1) {
        let t = f32(i) / 10.0; let d = length(q - bezier(p0, p1, p2, p3, t));
        if (d < minDist) { minDist = d; bestT = t; }
    }
    var delta: f32 = 0.05;
    for (var j: i32 = 0; j < 3; j = j + 1) {
        let t1 = clamp(bestT - delta, 0.0, 1.0); let t2 = clamp(bestT + delta, 0.0, 1.0);
        let d1 = length(q - bezier(p0, p1, p2, p3, t1)); let d2 = length(q - bezier(p0, p1, p2, p3, t2));
        if (d1 < d2) { if (d1 < minDist) { minDist = d1; bestT = t1; } }
        else { if (d2 < minDist) { minDist = d2; bestT = t2; } }
        delta = delta * 0.5;
    }
    return bestT;
}

fn isInsideHandle(xyz: vec3<f32>) -> bool {
    let p0 = vec2<f32>(-1.55, 1.9875); let p1 = vec2<f32>(-2.4, 1.9875); let p2 = vec2<f32>(-2.85, 1.9875); let p3 = vec2<f32>(-2.85, 1.65);
    let p4 = vec2<f32>(-2.85, 1.3125); let p5 = vec2<f32>(-2.575, 0.88125); let p6 = vec2<f32>(-1.95, 0.6);
    let t1 = projectToBezier(xyz.xy, p0, p1, p2, p3); let d1 = length(xyz.xy - bezier(p0, p1, p2, p3, t1));
    let t2 = projectToBezier(xyz.xy, p3, p4, p5, p6); let d2 = length(xyz.xy - bezier(p3, p4, p5, p6, t2));
    let d = min(d1, d2); let r_xy = 0.15; let r_z = 0.225;
    return (d*d)/(r_xy*r_xy) + (xyz.z*xyz.z)/(r_z*r_z) < 1.0;
}

fn bezier1D(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let u = 1.0 - t;
    return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}

fn distToSegment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, w: ptr<function, f32>) -> f32 {
    let ba = b - a;
    let l2 = dot(ba, ba);
    if (l2 < 1e-6) { *w = 0.0; return length(p - a); }
    *w = clamp(dot(p - a, ba) / l2, 0.0, 1.0);
    return length(p - (a + (*w) * ba));
}

fn testHump(p: vec3<f32>, q0: array<vec2<f32>, 4>, q3: array<vec2<f32>, 4>, zc: array<f32, 4>, xyScale: f32, zScale: f32) -> bool {
    var bestV: f32 = 0.5; var minDist: f32 = 1e10; var bestW: f32 = 0.0;
    for (var i: i32 = 0; i <= 24; i = i + 1) {
        let v = f32(i) / 24.0;
        var A = bezier(q0[0], q0[1], q0[2], q0[3], v);
        var B = bezier(q3[0], q3[1], q3[2], q3[3], v);
        if (xyScale != 1.0) {
            let C = (A + B) * 0.5;
            A = C + (A - C) * xyScale;
            B = C + (B - C) * xyScale;
        }
        var curW: f32 = 0.0;
        let d = distToSegment(p.xy, A, B, &curW);
        if (d < minDist) { minDist = d; bestV = v; bestW = curW; }
    }
    var delta: f32 = 1.0 / 48.0;
    for (var j: i32 = 0; j < 3; j = j + 1) {
        let v1 = clamp(bestV - delta, 0.0, 1.0); let v2 = clamp(bestV + delta, 0.0, 1.0);
        var A1 = bezier(q0[0], q0[1], q0[2], q0[3], v1); var B1 = bezier(q3[0], q3[1], q3[2], q3[3], v1);
        var A2 = bezier(q0[0], q0[1], q0[2], q0[3], v2); var B2 = bezier(q3[0], q3[1], q3[2], q3[3], v2);
        if (xyScale != 1.0) {
            let C1 = (A1 + B1) * 0.5; A1 = C1 + (A1 - C1) * xyScale; B1 = C1 + (B1 - C1) * xyScale;
            let C2 = (A2 + B2) * 0.5; A2 = C2 + (A2 - C2) * xyScale; B2 = C2 + (B2 - C2) * xyScale;
        }
        var w1: f32 = 0.0; var w2: f32 = 0.0;
        let d1 = distToSegment(p.xy, A1, B1, &w1); let d2 = distToSegment(p.xy, A2, B2, &w2);
        if (d1 < d2) { if (d1 < minDist) { minDist = d1; bestV = v1; bestW = w1; } }
        else { if (d2 < minDist) { minDist = d2; bestV = v2; bestW = w2; } }
        delta = delta * 0.5;
    }
    if (minDist > 0.03) { return false; }
    let Zv = bezier1D(zc[0], zc[1], zc[2], zc[3], bestV) * zScale;
    let u = 0.5 + sin(asin(clamp(2.0*bestW - 1.0, -1.0, 1.0)) / 3.0);
    return abs(p.z) < Zv * 3.0 * u * (1.0 - u);
}

fn isInsideSpout(xyz: vec3<f32>, outside: bool) -> bool {
    let b0 = array<vec2<f32>, 4>(vec2<f32>(1.7, 1.275), vec2<f32>(2.6, 1.275), vec2<f32>(2.3, 1.95), vec2<f32>(2.7, 2.25));
    let b3 = array<vec2<f32>, 4>(vec2<f32>(1.7, 0.45), vec2<f32>(3.1, 0.675), vec2<f32>(2.4, 1.875), vec2<f32>(3.3, 2.25));
    let bz = array<f32, 4>(0.66, 0.66, 0.25, 0.25);

    if (!outside) {
        return testHump(xyz, b0, b3, bz, 0.666667, 0.6);
    }

    if (testHump(xyz, b0, b3, bz, 1.0, 1.0)) { return true; }

    let t0 = array<vec2<f32>, 4>(vec2<f32>(2.7, 2.25), vec2<f32>(2.8, 2.325), vec2<f32>(2.9, 2.325), vec2<f32>(2.8, 2.25));
    let t3 = array<vec2<f32>, 4>(vec2<f32>(3.3, 2.25), vec2<f32>(3.525, 2.34375), vec2<f32>(3.45, 2.3625), vec2<f32>(3.2, 2.25));
    let tz = array<f32, 4>(0.25, 0.25, 0.15, 0.15);

    return testHump(xyz, t0, t3, tz, 1.0, 1.0);
}

fn mainModel4(pos: vec3<f32>) -> vec4<f32> {
    let xyz = pos.xzy;
    let spoutSolid = isInsideSpout(xyz, true) && !isInsideSpout(xyz, false);
    let bodySolid = isInsideBody(xyz, true) && !isInsideBody(xyz, false);
    let lidSolid = isInsideLid(xyz, true) && !isInsideLid(xyz, false);
    let handleSolid = isInsideHandle(xyz);
    if (bodySolid || spoutSolid || lidSolid || handleSolid) { return vec4<f32>(1.0, 0.0, 0.0, 0.0); }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
```

**TSL**

```tsl
/*{
  irmf: "1.0",
  language: "tsl",
  tslVersion: "0.180.0",
  materials: ["porcelain"],
  min: [-3.3,-2.1,0],
  max: [3.6,2.1,3.3],
  units: "mm",
}*/

// Ported from utah-teapot-glsl.irmf. All ray-casting loops have constant
// bounds, so they are unrolled at graph-build time; only the sample point and
// the refined curve parameters are runtime nodes.

const bezier2 = (p0, p1, p2, p3, t) => {
  if (typeof t === 'number') {
    // All-constant evaluation: fold in JS so the node graph stays small.
    const u = 1.0 - t;
    const b0 = u * u * u;
    const b1 = 3.0 * u * u * t;
    const b2 = 3.0 * u * t * t;
    const b3 = t * t * t;
    return vec2(
      Math.fround(p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3),
      Math.fround(p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3)
    );
  }
  const u = float(1.0).sub(t);
  const b0 = u.mul(u).mul(u);
  const b1 = float(3.0).mul(u).mul(u).mul(t);
  const b2 = float(3.0).mul(u).mul(t).mul(t);
  const b3 = t.mul(t).mul(t);
  return vec2(p0[0], p0[1]).mul(b0)
    .add(vec2(p1[0], p1[1]).mul(b1))
    .add(vec2(p2[0], p2[1]).mul(b2))
    .add(vec2(p3[0], p3[1]).mul(b3));
};

const intersectBezier = (p, c) => {
  const samples = 20;
  const acc = float(0.0).toVar();
  const prev = bezier2(c[0], c[1], c[2], c[3], 0.0).toVar();
  for (let i = 1; i <= samples; i++) {
    const curr = bezier2(c[0], c[1], c[2], c[3], i / samples);
    const cond = prev.y.lessThanEqual(p.y).and(p.y.lessThan(curr.y))
      .or(curr.y.lessThanEqual(p.y).and(p.y.lessThan(prev.y)));
    const x = prev.x.add(curr.x.sub(prev.x).mul(p.y.sub(prev.y)).div(curr.y.sub(prev.y)));
    acc.assign(acc.add(select(cond.and(x.greaterThan(p.x)), float(1.0), float(0.0))));
    prev.assign(curr);
  }
  return acc;
};

const intersectLine = (p, p0, p1) => {
  const cond = float(p0[1]).lessThanEqual(p.y).and(p.y.lessThan(p1[1]))
    .or(float(p1[1]).lessThanEqual(p.y).and(p.y.lessThan(p0[1])));
  const x = float(p0[0]).add(
    float(p1[0] - p0[0]).mul(p.y.sub(p0[1])).div(p1[1] - p0[1])
  );
  return select(cond.and(x.greaterThan(p.x)), float(1.0), float(0.0));
};

const mid = (a, b) => [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];

const bspline = (ctrl, t) => {
  const mt = t * 6.0;
  let seg = Math.floor(mt);
  if (seg < 0) seg = 0;
  if (seg > 5) seg = 5;
  const f = mt - seg;
  if (seg === 0) return bezier2(ctrl[0], ctrl[1], ctrl[2], mid(ctrl[2], ctrl[3]), f);
  if (seg === 5) return bezier2(mid(ctrl[5], ctrl[6]), ctrl[6], ctrl[7], ctrl[8], f);
  return bezier2(
    mid(ctrl[seg], ctrl[seg + 1]),
    ctrl[seg + 1],
    ctrl[seg + 2],
    mid(ctrl[seg + 2], ctrl[seg + 3]),
    f
  );
};

const intersectBSpline = (p, ctrl) => {
  const samples = 40;
  const acc = float(0.0).toVar();
  const prev = bezier2(ctrl[0], ctrl[1], ctrl[2], mid(ctrl[2], ctrl[3]), 0.0).toVar();
  for (let i = 1; i <= samples; i++) {
    const curr = bspline(ctrl, i / samples);
    const cond = prev.y.lessThanEqual(p.y).and(p.y.lessThan(curr.y))
      .or(curr.y.lessThanEqual(p.y).and(p.y.lessThan(prev.y)));
    const x = prev.x.add(curr.x.sub(prev.x).mul(p.y.sub(prev.y)).div(curr.y.sub(prev.y)));
    acc.assign(acc.add(select(cond.and(x.greaterThan(p.x)), float(1.0), float(0.0))));
    prev.assign(curr);
  }
  return acc;
};

const bsplineLid = (ctrl, t) => {
  const mt = t * 9.0;
  let seg = Math.floor(mt);
  if (seg < 0) seg = 0;
  if (seg > 8) seg = 8;
  const f = mt - seg;
  if (seg === 0) return bezier2(ctrl[0], ctrl[1], ctrl[2], mid(ctrl[2], ctrl[3]), f);
  if (seg === 8) return bezier2(mid(ctrl[8], ctrl[9]), ctrl[9], ctrl[10], ctrl[11], f);
  return bezier2(
    mid(ctrl[seg], ctrl[seg + 1]),
    ctrl[seg + 1],
    ctrl[seg + 2],
    mid(ctrl[seg + 2], ctrl[seg + 3]),
    f
  );
};

const intersectBSplineLid = (p, ctrl) => {
  const samples = 50;
  const acc = float(0.0).toVar();
  const prev = bezier2(ctrl[0], ctrl[1], ctrl[2], mid(ctrl[2], ctrl[3]), 0.0).toVar();
  for (let i = 1; i <= samples; i++) {
    const curr = bsplineLid(ctrl, i / samples);
    const cond = prev.y.lessThanEqual(p.y).and(p.y.lessThan(curr.y))
      .or(curr.y.lessThanEqual(p.y).and(p.y.lessThan(prev.y)));
    const x = prev.x.add(curr.x.sub(prev.x).mul(p.y.sub(prev.y)).div(curr.y.sub(prev.y)));
    acc.assign(acc.add(select(cond.and(x.greaterThan(p.x)), float(1.0), float(0.0))));
    prev.assign(curr);
  }
  return acc;
};

const isInsideBody = (xyz, outside) => {
  const p = vec2(length(xyz.xz), xyz.y);
  let count = float(0.0);
  if (outside) {
    count = count.add(intersectBezier(p, [[1.4, 2.25], [1.3375, 2.38125], [1.4375, 2.38125], [1.5, 2.25]]));
    count = count.add(intersectBezier(p, [[1.5, 2.25], [1.75, 1.725], [2.0, 1.2], [2.0, 0.75]]));
    count = count.add(intersectBezier(p, [[2.0, 0.75], [2.0, 0.3], [1.5, 0.075], [1.5, 0.0]]));
    count = count.add(intersectLine(p, [1.5, 0.0], [0.0, 0.0]));
    count = count.add(intersectLine(p, [0.0, 0.0], [0.0, 2.25]));
    count = count.add(intersectLine(p, [0.0, 2.25], [1.4, 2.25]));
  } else {
    const ctrl = [
      [1.4, 2.25], [1.3, 2.25], [1.2, 2.25],
      [1.2, 2.15], [1.5, 2.12], [2.0, 0.6],
      [1.4, 0.2], [1.3, 0.15], [1.1, 0.15],
    ];
    count = count.add(intersectBSpline(p, ctrl));
    count = count.add(intersectLine(p, [1.1, 0.15], [0.0, 0.15]));
    count = count.add(intersectLine(p, [0.0, 0.15], [0.0, 2.25]));
    count = count.add(intersectLine(p, [0.0, 2.25], [1.4, 2.25]));
  }
  return mod(count, 2.0).notEqual(0.0);
};

const isInsideLid = (xyz, outside) => {
  const p = vec2(length(xyz.xz), xyz.y);
  let count = float(0.0);
  if (outside) {
    count = count.add(intersectBezier(p, [[0.01, 3.0], [0.8, 3.0], [0.0, 2.7], [0.2, 2.55]]));
    count = count.add(intersectBezier(p, [[0.2, 2.55], [0.4, 2.4], [1.3, 2.4], [1.3, 2.25]]));
    count = count.add(intersectLine(p, [1.3, 2.25], [0.0, 2.25]));
    count = count.add(intersectLine(p, [0.0, 2.25], [0.0, 3.0]));
    count = count.add(intersectLine(p, [0.0, 3.0], [0.01, 3.0]));
  } else {
    const ctrl = [
      [1.3, 2.25], [1.2, 2.25], [1.18, 2.25],
      [1.15, 2.15], [1.1, 2.2], [1.0, 2.32],
      [0.5, 2.35], [0.07, 2.48], [0.07, 2.66],
      [0.26, 2.87], [0.2, 2.91], [0.05, 2.91],
    ];
    count = count.add(intersectBSplineLid(p, ctrl));
    count = count.add(intersectLine(p, [0.05, 2.91], [0.0, 2.91]));
    count = count.add(intersectLine(p, [0.0, 2.91], [0.0, 2.25]));
    count = count.add(intersectLine(p, [0.0, 2.25], [1.3, 2.25]));
  }
  return mod(count, 2.0).notEqual(0.0);
};

const projectToBezier = (q, c) => {
  const bestT = float(0.5).toVar();
  const minDist = float(1e10).toVar();
  Loop({ start: 0, end: 11, name: 'pi' }, ({ pi }) => {
    const t = float(pi).div(10.0);
    const d = length(q.sub(bezier2(c[0], c[1], c[2], c[3], t)));
    const better = d.lessThan(minDist).toVar();
    minDist.assign(select(better, d, minDist));
    bestT.assign(select(better, t, bestT));
  });
  const delta = float(0.05).toVar();
  Loop({ start: 0, end: 3, name: 'pj' }, () => {
    const t1 = clamp(bestT.sub(delta), 0.0, 1.0);
    const t2 = clamp(bestT.add(delta), 0.0, 1.0);
    const d1 = length(q.sub(bezier2(c[0], c[1], c[2], c[3], t1)));
    const d2 = length(q.sub(bezier2(c[0], c[1], c[2], c[3], t2)));
    const useD1 = d1.lessThan(d2);
    const candD = select(useD1, d1, d2);
    const candT = select(useD1, t1, t2);
    const better = candD.lessThan(minDist).toVar();
    minDist.assign(select(better, candD, minDist));
    bestT.assign(select(better, candT, bestT));
    delta.assign(delta.mul(0.5));
  });
  return bestT;
};

const isInsideHandle = (xyz) => {
  const p0 = [-1.55, 1.9875], p1 = [-2.4, 1.9875], p2 = [-2.85, 1.9875], p3 = [-2.85, 1.65];
  const p4 = [-2.85, 1.3125], p5 = [-2.575, 0.88125], p6 = [-1.95, 0.6];
  const t1 = projectToBezier(xyz.xy, [p0, p1, p2, p3]);
  const d1 = length(xyz.xy.sub(bezier2(p0, p1, p2, p3, t1)));
  const t2 = projectToBezier(xyz.xy, [p3, p4, p5, p6]);
  const d2 = length(xyz.xy.sub(bezier2(p3, p4, p5, p6, t2)));
  const d = min(d1, d2);
  const rxy = 0.15;
  const rz = 0.225;
  return d.mul(d).div(rxy * rxy)
    .add(xyz.z.mul(xyz.z).div(rz * rz))
    .lessThan(1.0);
};

const bezier1D = (p0, p1, p2, p3, t) => {
  const u = float(1.0).sub(t);
  return u.mul(u).mul(u).mul(p0)
    .add(float(3.0).mul(u).mul(u).mul(t).mul(p1))
    .add(float(3.0).mul(u).mul(t).mul(t).mul(p2))
    .add(t.mul(t).mul(t).mul(p3));
};

const distToSegment = (p, a, b) => {
  const ba = b.sub(a);
  const l2 = dot(ba, ba);
  const tiny = l2.lessThan(1e-6);
  const w = select(tiny, float(0.0), clamp(dot(p.sub(a), ba).div(l2), 0.0, 1.0));
  const d = select(tiny, length(p.sub(a)), length(p.sub(a.add(ba.mul(w)))));
  return { d, w };
};

const testHump = (p, q0, q3, zc, xyScale, zScale) => {
  const shrink = (A, B) => {
    if (xyScale === 1.0) return [A, B];
    const C = A.add(B).mul(0.5);
    return [C.add(A.sub(C).mul(xyScale)), C.add(B.sub(C).mul(xyScale))];
  };
  const bestV = float(0.5).toVar();
  const minDist = float(1e10).toVar();
  const bestW = float(0.0).toVar();
  Loop({ start: 0, end: 25, name: 'hi' }, ({ hi }) => {
    const v = float(hi).div(24.0);
    let A = bezier2(q0[0], q0[1], q0[2], q0[3], v);
    let B = bezier2(q3[0], q3[1], q3[2], q3[3], v);
    [A, B] = shrink(A, B);
    const r = distToSegment(p.xy, A, B);
    const better = r.d.lessThan(minDist).toVar();
    minDist.assign(select(better, r.d, minDist));
    bestV.assign(select(better, v, bestV));
    bestW.assign(select(better, r.w, bestW));
  });
  const delta = float(1.0 / 48.0).toVar();
  Loop({ start: 0, end: 3, name: 'hj' }, () => {
    const v1 = clamp(bestV.sub(delta), 0.0, 1.0);
    const v2 = clamp(bestV.add(delta), 0.0, 1.0);
    let A1 = bezier2(q0[0], q0[1], q0[2], q0[3], v1);
    let B1 = bezier2(q3[0], q3[1], q3[2], q3[3], v1);
    let A2 = bezier2(q0[0], q0[1], q0[2], q0[3], v2);
    let B2 = bezier2(q3[0], q3[1], q3[2], q3[3], v2);
    [A1, B1] = shrink(A1, B1);
    [A2, B2] = shrink(A2, B2);
    const r1 = distToSegment(p.xy, A1, B1);
    const r2 = distToSegment(p.xy, A2, B2);
    const useD1 = r1.d.lessThan(r2.d);
    const candD = select(useD1, r1.d, r2.d);
    const candV = select(useD1, v1, v2);
    const candW = select(useD1, r1.w, r2.w);
    const better = candD.lessThan(minDist).toVar();
    minDist.assign(select(better, candD, minDist));
    bestV.assign(select(better, candV, bestV));
    bestW.assign(select(better, candW, bestW));
    delta.assign(delta.mul(0.5));
  });
  const Zv = bezier1D(zc[0], zc[1], zc[2], zc[3], bestV).mul(zScale);
  const u = float(0.5).add(
    sin(asin(clamp(bestW.mul(2.0).sub(1.0), -1.0, 1.0)).div(3.0))
  );
  const near = minDist.lessThanEqual(0.03);
  const thin = p.z.abs().lessThan(Zv.mul(3.0).mul(u).mul(float(1.0).sub(u)));
  return near.and(thin);
};

const isInsideSpout = (xyz, outside) => {
  const b0 = [[1.7, 1.275], [2.6, 1.275], [2.3, 1.95], [2.7, 2.25]];
  const b3 = [[1.7, 0.45], [3.1, 0.675], [2.4, 1.875], [3.3, 2.25]];
  const bz = [0.66, 0.66, 0.25, 0.25];
  if (!outside) return testHump(xyz, b0, b3, bz, 0.666667, 0.6);
  const h1 = testHump(xyz, b0, b3, bz, 1.0, 1.0);
  const t0 = [[2.7, 2.25], [2.8, 2.325], [2.9, 2.325], [2.8, 2.25]];
  const t3 = [[3.3, 2.25], [3.525, 2.34375], [3.45, 2.3625], [3.2, 2.25]];
  const tz = [0.25, 0.25, 0.15, 0.15];
  return h1.or(testHump(xyz, t0, t3, tz, 1.0, 1.0));
};

const mainModel4 = (pos) => {
  const xyz = pos.xzy;
  const spoutSolid = isInsideSpout(xyz, true).and(isInsideSpout(xyz, false).not());
  const bodySolid = isInsideBody(xyz, true).and(isInsideBody(xyz, false).not());
  const lidSolid = isInsideLid(xyz, true).and(isInsideLid(xyz, false).not());
  const handleSolid = isInsideHandle(xyz);
  return [select(
    bodySolid.or(spoutSolid).or(lidSolid).or(handleSolid),
    float(1.0),
    float(0.0)
  )];
};
```

* Try loading [utah-teapot-glsl.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/036-utah-teapot/utah-teapot-glsl.irmf) now in the IRMF editor!
* Try loading [utah-teapot-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/036-utah-teapot/utah-teapot-wgsl.irmf) now in the IRMF editor!
* Try loading [utah-teapot-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/036-utah-teapot/utah-teapot-tsl.irmf) now in the IRMF editor!

* Use [irmf-slicer](https://github.com/gmlewis/irmf-slicer) to generate an STL or voxel approximation.

----------------------------------------------------------------------

# License

Copyright 2026 Glenn M. Lewis. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
