# 016-text

## text-1.irmf

Text is a bit trickier, and requires conversion from SVG web fonts.
An experimental [font2irmf](https://github.com/gmlewis/go-fonts) converter
is in the works, and is starting to show some promise.

![text-1.png](text-1.png)

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

float blinnLoop(vec2 A, vec2 B, vec2 C) {
  vec2 v0 = C - A;
  vec2 v1 = B - A;
  vec2 v2 = vec2(0.75,0.5) - A;
  // Compute dot products
  float dot00 = dot(v0, v0);
  float dot01 = dot(v0, v1);
  float dot02 = dot(v0, v2);
  float dot11 = dot(v1, v1);
  float dot12 = dot(v1, v2);
  // Compute barycentric coordinates
  float invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
  float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
  // use the blinn and loop method
  float w = (1.0 - u - v);
  return w;
}

float interpLine(vec2 A, vec2 B, float y) {
  float p = (y - A.y) / (B.y - A.y);
  return p*(B.x-A.x) + A.x;
}

float interpQuadratic(vec2 p0, vec2 p1, vec2 p2, float y) {
  float a = p2.y + p0.y - 2.0*p1.y;
  float b = 2.0 * (p1.y - p0.y);
  float c = p0.y - y;
  if (b*b < 4.0*a*c) { return 0.0; } // bad (imaginary) quadratic
  float det = sqrt(b*b - 4.0*a*c);
  float t = (-b + det) / (2.0 * a);
  float t2 = (-b - det) / (2.0 * a);
  if (t2 >= 0.0 && t2 <= 1.0) {
    t = t2;
  }
  float x = (1.0-t)*(1.0-t)*p0.x + 2.0*(1.0-t)*t*p1.x + t*t*p2.x;
  return x;
}

float glyph_F_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(904.00,1467.00)))) { return 0.0; }

  if (xyz.y >= 1306.00 && xyz.y <= 1467.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2(904.00,1306.00),vec2(904.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 793.00 && xyz.y < 1306.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2(200.00,793.00),vec2(200.00,1306.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 631.00 && xyz.y < 793.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2(802.00,631.00),vec2(802.00,793.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 631.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2(200.00,0.00),vec2(200.00,631.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_F(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(178.00,0.00,0.0))) || any(greaterThan(xyz, vec3(1082.00,1467.00,height)))) { return 0.0; }
  xyz -= vec3(178.00,0.00,0.0);
  float result = glyph_F_1(xyz);
  return result;
}

float glyph_I_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(199.00,1467.00)))) { return 0.0; }

  if (xyz.y >= 0.00 && xyz.y <= 1467.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2(199.00,0.00),vec2(199.00,1467.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_I(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(215.00,0.00,0.0))) || any(greaterThan(xyz, vec3(414.00,1467.00,height)))) { return 0.0; }
  xyz -= vec3(215.00,0.00,0.0);
  float result = glyph_I_1(xyz);
  return result;
}

float glyph_M_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(1527.00,1467.00)))) { return 0.0; }

  if (xyz.y >= 1462.50 && xyz.y <= 1467.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2(188.00,1462.00),vec2(174.00,1467.00),vec2(147.00,1467.00),xyz.y) && xyz.x < interpQuadratic(vec2(1339.50,1462.50),vec2(1354.00,1467.00),vec2(1381.00,1467.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1462.00 && xyz.y < 1462.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2(188.00,1462.00),vec2(174.00,1467.00),vec2(147.00,1467.00),xyz.y) && xyz.x < interpQuadratic(vec2(1312.00,1434.00),vec2(1325.00,1458.00),vec2(1339.50,1462.50),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1434.00 && xyz.y < 1462.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2(215.00,1434.00),vec2(202.00,1457.00),vec2(188.00,1462.00),xyz.y) && xyz.x < interpQuadratic(vec2(1312.00,1434.00),vec2(1325.00,1458.00),vec2(1339.50,1462.50),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1178.00 && xyz.y < 1434.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(722.00,530.00),vec2(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2(815.00,531.00),vec2(1312.00,1434.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1175.00 && xyz.y < 1178.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2(172.50,1125.50),vec2(171.00,1151.00),vec2(168.00,1178.00),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2(722.00,530.00),vec2(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2(815.00,531.00),vec2(1312.00,1434.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1125.50 && xyz.y < 1175.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2(172.50,1125.50),vec2(171.00,1151.00),vec2(168.00,1178.00),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2(722.00,530.00),vec2(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2(815.00,531.00),vec2(1312.00,1434.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpQuadratic(vec2(1354.00,1124.50),vec2(1355.00,1149.00),vec2(1357.00,1175.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1124.50 && xyz.y < 1125.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2(174.00,1078.00),vec2(174.00,1100.00),vec2(172.50,1125.50),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2(722.00,530.00),vec2(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2(815.00,531.00),vec2(1312.00,1434.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpQuadratic(vec2(1354.00,1124.50),vec2(1355.00,1149.00),vec2(1357.00,1175.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1078.00 && xyz.y < 1124.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2(174.00,1078.00),vec2(174.00,1100.00),vec2(172.50,1125.50),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2(722.00,530.00),vec2(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2(815.00,531.00),vec2(1312.00,1434.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpQuadratic(vec2(1353.00,1078.00),vec2(1353.00,1100.00),vec2(1354.00,1124.50),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 531.00 && xyz.y < 1078.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(174.00,0.00),vec2(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2(722.00,530.00),vec2(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2(815.00,531.00),vec2(1312.00,1434.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2(1353.00,0.00),vec2(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 530.00 && xyz.y < 531.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(174.00,0.00),vec2(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2(722.00,530.00),vec2(215.00,1434.00),xyz.y) && xyz.x < interpQuadratic(vec2(789.50,479.00),vec2(801.00,505.00),vec2(815.00,531.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2(1353.00,0.00),vec2(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 479.00 && xyz.y < 530.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(174.00,0.00),vec2(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpQuadratic(vec2(747.00,478.50),vec2(736.00,505.00),vec2(722.00,530.00),xyz.y) && xyz.x < interpQuadratic(vec2(789.50,479.00),vec2(801.00,505.00),vec2(815.00,531.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2(1353.00,0.00),vec2(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 478.50 && xyz.y < 479.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(174.00,0.00),vec2(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpQuadratic(vec2(747.00,478.50),vec2(736.00,505.00),vec2(722.00,530.00),xyz.y) && xyz.x < interpQuadratic(vec2(768.00,424.00),vec2(778.00,453.00),vec2(789.50,479.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2(1353.00,0.00),vec2(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 424.00 && xyz.y < 478.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(174.00,0.00),vec2(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpQuadratic(vec2(768.00,424.00),vec2(758.00,452.00),vec2(747.00,478.50),xyz.y) && xyz.x < interpQuadratic(vec2(768.00,424.00),vec2(778.00,453.00),vec2(789.50,479.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2(1353.00,0.00),vec2(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 257.00 && xyz.y < 424.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(174.00,0.00),vec2(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2(682.00,257.00),vec2(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2(854.00,257.00),vec2(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2(1353.00,0.00),vec2(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 211.00 && xyz.y < 257.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(174.00,0.00),vec2(174.00,1078.00),xyz.y) && xyz.x < interpQuadratic(vec2(754.00,211.00),vec2(708.00,211.00),vec2(682.00,257.00),xyz.y)) || (xyz.x > interpQuadratic(vec2(783.00,211.00),vec2(828.00,211.00),vec2(854.00,257.00),xyz.y) && xyz.x < interpLine(vec2(1353.00,0.00),vec2(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 211.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(174.00,0.00),vec2(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2(1353.00,0.00),vec2(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2(1527.00,0.00),vec2(1527.00,1467.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_M(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(178.00,0.00,0.0))) || any(greaterThan(xyz, vec3(1705.00,1467.00,height)))) { return 0.0; }
  xyz -= vec3(178.00,0.00,0.0);
  float result = glyph_M_1(xyz);
  return result;
}

float glyph_R_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(1084.00,1467.00)))) { return 0.0; }

  if (xyz.y >= 1439.00 && xyz.y <= 1467.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(654.50,1439.00),vec2(553.00,1467.00),vec2(414.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1358.00 && xyz.y < 1439.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(822.00,1358.00),vec2(756.00,1411.00),vec2(654.50,1439.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1229.50 && xyz.y < 1358.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(920.00,1229.50),vec2(888.00,1305.00),vec2(822.00,1358.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1061.00 && xyz.y < 1229.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(952.00,1061.00),vec2(952.00,1154.00),vec2(920.00,1229.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 915.50 && xyz.y < 1061.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(927.50,915.50),vec2(952.00,983.00),vec2(952.00,1061.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 794.00 && xyz.y < 915.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(856.50,794.00),vec2(903.00,848.00),vec2(927.50,915.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 702.50 && xyz.y < 794.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(742.50,702.50),vec2(810.00,740.00),vec2(856.50,794.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 645.00 && xyz.y < 702.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(591.00,645.00),vec2(675.00,665.00),vec2(742.50,702.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 612.00 && xyz.y < 645.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2(656.00,583.00),vec2(628.00,623.00),vec2(591.00,645.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 601.50 && xyz.y < 612.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(197.00,0.00),vec2(197.00,612.00),xyz.y) && xyz.x < interpQuadratic(vec2(409.50,601.50),vec2(389.00,612.00),vec2(348.00,612.00),xyz.y)) || xyz.x > interpQuadratic(vec2(656.00,583.00),vec2(628.00,623.00),vec2(591.00,645.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 583.00 && xyz.y < 601.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(197.00,0.00),vec2(197.00,612.00),xyz.y) && xyz.x < interpQuadratic(vec2(447.00,566.00),vec2(430.00,591.00),vec2(409.50,601.50),xyz.y)) || xyz.x > interpQuadratic(vec2(656.00,583.00),vec2(628.00,623.00),vec2(591.00,645.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 566.00 && xyz.y < 583.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(197.00,0.00),vec2(197.00,612.00),xyz.y) && xyz.x < interpQuadratic(vec2(447.00,566.00),vec2(430.00,591.00),vec2(409.50,601.50),xyz.y)) || xyz.x > interpLine(vec2(1084.00,0.00),vec2(656.00,583.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 42.00 && xyz.y < 566.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(197.00,0.00),vec2(197.00,612.00),xyz.y) && xyz.x < interpLine(vec2(828.00,42.00),vec2(447.00,566.00),xyz.y)) || xyz.x > interpLine(vec2(1084.00,0.00),vec2(656.00,583.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 42.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2(197.00,0.00),vec2(197.00,612.00),xyz.y) && xyz.x < interpQuadratic(vec2(908.00,0.00),vec2(854.00,0.00),vec2(828.00,42.00),xyz.y)) || xyz.x > interpLine(vec2(1084.00,0.00),vec2(656.00,583.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_R_2(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(197.00,757.00))) || any(greaterThan(xyz.xy, vec2(758.00,1311.00)))) { return 0.0; }

  if (xyz.y >= 1244.50 && xyz.y <= 1311.00 && (xyz.x < interpLine(vec2(197.00,757.00),vec2(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2(671.50,1244.50),vec2(585.00,1311.00),vec2(414.00,1311.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1047.00 && xyz.y < 1244.50 && (xyz.x < interpLine(vec2(197.00,757.00),vec2(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2(758.00,1047.00),vec2(758.00,1178.00),vec2(671.50,1244.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 929.00 && xyz.y < 1047.00 && (xyz.x < interpLine(vec2(197.00,757.00),vec2(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2(735.50,929.00),vec2(758.00,982.00),vec2(758.00,1047.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 837.50 && xyz.y < 929.00 && (xyz.x < interpLine(vec2(197.00,757.00),vec2(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2(668.50,837.50),vec2(713.00,876.00),vec2(735.50,929.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 778.00 && xyz.y < 837.50 && (xyz.x < interpLine(vec2(197.00,757.00),vec2(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2(558.00,778.00),vec2(624.00,799.00),vec2(668.50,837.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 757.00 && xyz.y < 778.00 && (xyz.x < interpLine(vec2(197.00,757.00),vec2(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2(405.00,757.00),vec2(492.00,757.00),vec2(558.00,778.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_R(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(199.00,0.00,0.0))) || any(greaterThan(xyz, vec3(1283.00,1467.00,height)))) { return 0.0; }
  xyz -= vec3(199.00,0.00,0.0);
  float result = glyph_R_1(xyz);
  result -= glyph_R_2(xyz);
  return result;
}

float glyph_f_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(649.00,1489.00)))) { return 0.0; }

  if (xyz.y >= 1468.00 && xyz.y <= 1489.00 && (xyz.x < interpQuadratic(vec2(372.50,1464.50),vec2(438.00,1489.00),vec2(520.00,1489.00),xyz.y) || xyz.x > interpQuadratic(vec2(649.00,1468.00),vec2(590.00,1489.00),vec2(520.00,1489.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1464.50 && xyz.y < 1468.00 && (xyz.x < interpQuadratic(vec2(372.50,1464.50),vec2(438.00,1489.00),vec2(520.00,1489.00),xyz.y) || xyz.x > interpLine(vec2(645.00,1377.00),vec2(649.00,1468.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1392.50 && xyz.y < 1464.50 && (xyz.x < interpQuadratic(vec2(260.50,1392.50),vec2(307.00,1440.00),vec2(372.50,1464.50),xyz.y) || xyz.x > interpLine(vec2(645.00,1377.00),vec2(649.00,1468.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1377.00 && xyz.y < 1392.50 && (xyz.x < interpQuadratic(vec2(188.50,1275.50),vec2(214.00,1345.00),vec2(260.50,1392.50),xyz.y) || xyz.x > interpLine(vec2(645.00,1377.00),vec2(649.00,1468.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1353.00 && xyz.y < 1377.00 && (xyz.x < interpQuadratic(vec2(188.50,1275.50),vec2(214.00,1345.00),vec2(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2(627.50,1353.00),vec2(644.00,1357.00),vec2(645.00,1377.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1349.00 && xyz.y < 1353.00 && (xyz.x < interpQuadratic(vec2(188.50,1275.50),vec2(214.00,1345.00),vec2(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2(581.00,1349.00),vec2(611.00,1349.00),vec2(627.50,1353.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1336.50 && xyz.y < 1349.00 && (xyz.x < interpQuadratic(vec2(188.50,1275.50),vec2(214.00,1345.00),vec2(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2(464.50,1336.50),vec2(503.00,1349.00),vec2(550.00,1349.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1296.50 && xyz.y < 1336.50 && (xyz.x < interpQuadratic(vec2(188.50,1275.50),vec2(214.00,1345.00),vec2(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2(398.00,1296.50),vec2(426.00,1324.00),vec2(464.50,1336.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 1275.50 && xyz.y < 1296.50 && (xyz.x < interpQuadratic(vec2(188.50,1275.50),vec2(214.00,1345.00),vec2(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2(355.50,1224.00),vec2(370.00,1269.00),vec2(398.00,1296.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 1224.00 && xyz.y < 1275.50 && (xyz.x < interpQuadratic(vec2(163.00,1117.00),vec2(163.00,1206.00),vec2(188.50,1275.50),xyz.y) || xyz.x > interpQuadratic(vec2(355.50,1224.00),vec2(370.00,1269.00),vec2(398.00,1296.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 1117.00 && xyz.y < 1224.00 && (xyz.x < interpQuadratic(vec2(163.00,1117.00),vec2(163.00,1206.00),vec2(188.50,1275.50),xyz.y) || xyz.x > interpQuadratic(vec2(341.00,1112.00),vec2(341.00,1179.00),vec2(355.50,1224.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1112.00 && xyz.y < 1117.00 && (xyz.x < interpLine(vec2(163.00,1017.00),vec2(163.00,1117.00),xyz.y) || xyz.x > interpQuadratic(vec2(341.00,1112.00),vec2(341.00,1179.00),vec2(355.50,1224.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1017.00 && xyz.y < 1112.00 && (xyz.x < interpLine(vec2(163.00,1017.00),vec2(163.00,1117.00),xyz.y) || xyz.x > interpLine(vec2(341.00,1017.00),vec2(341.00,1112.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 942.00 && xyz.y < 1017.00 && (xyz.x < interpLine(vec2(0.00,942.00),vec2(0.00,1017.00),xyz.y) || xyz.x > interpLine(vec2(641.00,885.00),vec2(641.00,1017.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 911.00 && xyz.y < 942.00 && (xyz.x < interpQuadratic(vec2(13.50,911.00),vec2(0.00,922.00),vec2(0.00,942.00),xyz.y) || xyz.x > interpLine(vec2(641.00,885.00),vec2(641.00,1017.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 895.00 && xyz.y < 911.00 && (xyz.x < interpQuadratic(vec2(49.00,895.00),vec2(27.00,900.00),vec2(13.50,911.00),xyz.y) || xyz.x > interpLine(vec2(641.00,885.00),vec2(641.00,1017.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 885.00 && xyz.y < 895.00 && (xyz.x < interpLine(vec2(163.00,882.00),vec2(49.00,895.00),xyz.y) || xyz.x > interpLine(vec2(641.00,885.00),vec2(641.00,1017.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 882.00 && xyz.y < 885.00 && (xyz.x < interpLine(vec2(163.00,882.00),vec2(49.00,895.00),xyz.y) || xyz.x > interpLine(vec2(347.00,0.00),vec2(347.00,885.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 882.00 && (xyz.x < interpLine(vec2(163.00,0.00),vec2(163.00,882.00),xyz.y) || xyz.x > interpLine(vec2(347.00,0.00),vec2(347.00,885.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_f(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(27.00,0.00,0.0))) || any(greaterThan(xyz, vec3(676.00,1489.00,height)))) { return 0.0; }
  xyz -= vec3(27.00,0.00,0.0);
  float result = glyph_f_1(xyz);
  return result;
}

float glyph_n_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(863.00,1054.00)))) { return 0.0; }

  if (xyz.y >= 1037.00 && xyz.y <= 1054.00 && (xyz.x < interpQuadratic(vec2(323.00,1008.00),vec2(406.00,1054.00),vec2(516.00,1054.00),xyz.y) || xyz.x > interpQuadratic(vec2(666.00,1025.50),vec2(601.00,1054.00),vec2(516.00,1054.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1025.50 && xyz.y < 1037.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2(157.00,999.00),vec2(147.00,1037.00),vec2(108.00,1037.00),xyz.y) && xyz.x < interpQuadratic(vec2(323.00,1008.00),vec2(406.00,1054.00),vec2(516.00,1054.00),xyz.y)) || xyz.x > interpQuadratic(vec2(666.00,1025.50),vec2(601.00,1054.00),vec2(516.00,1054.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1008.00 && xyz.y < 1025.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2(157.00,999.00),vec2(147.00,1037.00),vec2(108.00,1037.00),xyz.y) && xyz.x < interpQuadratic(vec2(323.00,1008.00),vec2(406.00,1054.00),vec2(516.00,1054.00),xyz.y)) || xyz.x > interpQuadratic(vec2(774.50,945.50),vec2(731.00,997.00),vec2(666.00,1025.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 999.00 && xyz.y < 1008.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2(157.00,999.00),vec2(147.00,1037.00),vec2(108.00,1037.00),xyz.y) && xyz.x < interpQuadratic(vec2(172.00,887.00),vec2(240.00,962.00),vec2(323.00,1008.00),xyz.y)) || xyz.x > interpQuadratic(vec2(774.50,945.50),vec2(731.00,997.00),vec2(666.00,1025.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 945.50 && xyz.y < 999.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2(172.00,887.00),vec2(157.00,999.00),xyz.y) && xyz.x < interpQuadratic(vec2(172.00,887.00),vec2(240.00,962.00),vec2(323.00,1008.00),xyz.y)) || xyz.x > interpQuadratic(vec2(774.50,945.50),vec2(731.00,997.00),vec2(666.00,1025.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 908.00 && xyz.y < 945.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2(172.00,887.00),vec2(157.00,999.00),xyz.y) && xyz.x < interpQuadratic(vec2(172.00,887.00),vec2(240.00,962.00),vec2(323.00,1008.00),xyz.y)) || xyz.x > interpQuadratic(vec2(840.50,821.50),vec2(818.00,894.00),vec2(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 887.00 && xyz.y < 908.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2(172.00,887.00),vec2(157.00,999.00),xyz.y) && xyz.x < interpQuadratic(vec2(172.00,887.00),vec2(240.00,962.00),vec2(323.00,1008.00),xyz.y)) || (xyz.x > interpQuadratic(vec2(311.00,869.00),vec2(381.00,908.00),vec2(462.00,908.00),xyz.y) && xyz.x < interpQuadratic(vec2(626.50,843.00),vec2(573.00,908.00),vec2(462.00,908.00),xyz.y)) || xyz.x > interpQuadratic(vec2(840.50,821.50),vec2(818.00,894.00),vec2(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 869.00 && xyz.y < 887.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2(311.00,869.00),vec2(381.00,908.00),vec2(462.00,908.00),xyz.y) && xyz.x < interpQuadratic(vec2(626.50,843.00),vec2(573.00,908.00),vec2(462.00,908.00),xyz.y)) || xyz.x > interpQuadratic(vec2(840.50,821.50),vec2(818.00,894.00),vec2(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 843.00 && xyz.y < 869.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2(182.00,764.00),vec2(241.00,830.00),vec2(311.00,869.00),xyz.y) && xyz.x < interpQuadratic(vec2(626.50,843.00),vec2(573.00,908.00),vec2(462.00,908.00),xyz.y)) || xyz.x > interpQuadratic(vec2(840.50,821.50),vec2(818.00,894.00),vec2(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 821.50 && xyz.y < 843.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2(182.00,764.00),vec2(241.00,830.00),vec2(311.00,869.00),xyz.y) && xyz.x < interpQuadratic(vec2(680.00,660.00),vec2(680.00,778.00),vec2(626.50,843.00),xyz.y)) || xyz.x > interpQuadratic(vec2(840.50,821.50),vec2(818.00,894.00),vec2(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 764.00 && xyz.y < 821.50 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2(182.00,764.00),vec2(241.00,830.00),vec2(311.00,869.00),xyz.y) && xyz.x < interpQuadratic(vec2(680.00,660.00),vec2(680.00,778.00),vec2(626.50,843.00),xyz.y)) || xyz.x > interpQuadratic(vec2(863.00,660.00),vec2(863.00,749.00),vec2(840.50,821.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 660.00 && xyz.y < 764.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2(182.00,0.00),vec2(182.00,764.00),xyz.y) && xyz.x < interpQuadratic(vec2(680.00,660.00),vec2(680.00,778.00),vec2(626.50,843.00),xyz.y)) || xyz.x > interpQuadratic(vec2(863.00,660.00),vec2(863.00,749.00),vec2(840.50,821.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 660.00 && (xyz.x < interpLine(vec2(0.00,0.00),vec2(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2(182.00,0.00),vec2(182.00,764.00),xyz.y) && xyz.x < interpLine(vec2(680.00,0.00),vec2(680.00,660.00),xyz.y)) || xyz.x > interpLine(vec2(863.00,0.00),vec2(863.00,660.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_n(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(150.00,0.00,0.0))) || any(greaterThan(xyz, vec3(1013.00,1054.00,height)))) { return 0.0; }
  xyz -= vec3(150.00,0.00,0.0);
  float result = glyph_n_1(xyz);
  return result;
}

float glyph_o_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(989.00,1068.00)))) { return 0.0; }

  if (xyz.y >= 1030.00 && xyz.y <= 1068.00 && (xyz.x < interpQuadratic(vec2(290.00,1030.00),vec2(382.00,1068.00),vec2(495.00,1068.00),xyz.y) || xyz.x > interpQuadratic(vec2(700.50,1030.00),vec2(609.00,1068.00),vec2(495.00,1068.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 922.50 && xyz.y < 1030.00 && (xyz.x < interpQuadratic(vec2(133.50,922.50),vec2(198.00,992.00),vec2(290.00,1030.00),xyz.y) || xyz.x > interpQuadratic(vec2(856.00,922.50),vec2(792.00,992.00),vec2(700.50,1030.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 754.00 && xyz.y < 922.50 && (xyz.x < interpQuadratic(vec2(34.50,754.00),vec2(69.00,853.00),vec2(133.50,922.50),xyz.y) || xyz.x > interpQuadratic(vec2(954.50,754.00),vec2(920.00,853.00),vec2(856.00,922.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 533.00 && xyz.y < 754.00 && (xyz.x < interpQuadratic(vec2(0.00,533.00),vec2(0.00,655.00),vec2(34.50,754.00),xyz.y) || xyz.x > interpQuadratic(vec2(989.00,533.00),vec2(989.00,655.00),vec2(954.50,754.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 312.00 && xyz.y < 533.00 && (xyz.x < interpQuadratic(vec2(34.50,312.00),vec2(0.00,410.00),vec2(0.00,533.00),xyz.y) || xyz.x > interpQuadratic(vec2(954.50,312.00),vec2(989.00,410.00),vec2(989.00,533.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 144.00 && xyz.y < 312.00 && (xyz.x < interpQuadratic(vec2(133.50,144.00),vec2(69.00,214.00),vec2(34.50,312.00),xyz.y) || xyz.x > interpQuadratic(vec2(856.00,144.00),vec2(920.00,214.00),vec2(954.50,312.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 37.00 && xyz.y < 144.00 && (xyz.x < interpQuadratic(vec2(289.50,37.00),vec2(198.00,74.00),vec2(133.50,144.00),xyz.y) || xyz.x > interpQuadratic(vec2(700.50,37.00),vec2(792.00,74.00),vec2(856.00,144.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 37.00 && (xyz.x < interpQuadratic(vec2(495.00,0.00),vec2(381.00,0.00),vec2(289.50,37.00),xyz.y) || xyz.x > interpQuadratic(vec2(495.00,0.00),vec2(609.00,0.00),vec2(700.50,37.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_o_2(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(188.00,142.00))) || any(greaterThan(xyz.xy, vec2(800.00,924.00)))) { return 0.0; }

  if (xyz.y >= 897.50 && xyz.y <= 924.00 && (xyz.x < interpQuadratic(vec2(360.00,897.50),vec2(417.00,924.00),vec2(495.00,924.00),xyz.y) || xyz.x > interpQuadratic(vec2(724.50,820.50),vec2(649.00,924.00),vec2(495.00,924.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 821.00 && xyz.y < 897.50 && (xyz.x < interpQuadratic(vec2(264.50,821.00),vec2(303.00,871.00),vec2(360.00,897.50),xyz.y) || xyz.x > interpQuadratic(vec2(724.50,820.50),vec2(649.00,924.00),vec2(495.00,924.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 820.50 && xyz.y < 821.00 && (xyz.x < interpQuadratic(vec2(207.00,697.50),vec2(226.00,771.00),vec2(264.50,821.00),xyz.y) || xyz.x > interpQuadratic(vec2(724.50,820.50),vec2(649.00,924.00),vec2(495.00,924.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 697.50 && xyz.y < 820.50 && (xyz.x < interpQuadratic(vec2(207.00,697.50),vec2(226.00,771.00),vec2(264.50,821.00),xyz.y) || xyz.x > interpQuadratic(vec2(800.00,532.00),vec2(800.00,717.00),vec2(724.50,820.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 532.00 && xyz.y < 697.50 && (xyz.x < interpQuadratic(vec2(188.00,532.00),vec2(188.00,624.00),vec2(207.00,697.50),xyz.y) || xyz.x > interpQuadratic(vec2(800.00,532.00),vec2(800.00,717.00),vec2(724.50,820.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 367.50 && xyz.y < 532.00 && (xyz.x < interpQuadratic(vec2(207.00,367.50),vec2(188.00,440.00),vec2(188.00,532.00),xyz.y) || xyz.x > interpQuadratic(vec2(724.50,245.00),vec2(800.00,348.00),vec2(800.00,532.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 245.00 && xyz.y < 367.50 && (xyz.x < interpQuadratic(vec2(264.50,245.00),vec2(226.00,295.00),vec2(207.00,367.50),xyz.y) || xyz.x > interpQuadratic(vec2(724.50,245.00),vec2(800.00,348.00),vec2(800.00,532.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 168.50 && xyz.y < 245.00 && (xyz.x < interpQuadratic(vec2(360.50,168.50),vec2(303.00,195.00),vec2(264.50,245.00),xyz.y) || xyz.x > interpQuadratic(vec2(495.00,142.00),vec2(649.00,142.00),vec2(724.50,245.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 142.00 && xyz.y < 168.50 && (xyz.x < interpQuadratic(vec2(495.00,142.00),vec2(418.00,142.00),vec2(360.50,168.50),xyz.y) || xyz.x > interpQuadratic(vec2(495.00,142.00),vec2(649.00,142.00),vec2(724.50,245.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_o(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(74.00,-14.00,0.0))) || any(greaterThan(xyz, vec3(1063.00,1054.00,height)))) { return 0.0; }
  xyz -= vec3(74.00,-14.00,0.0);
  float result = glyph_o_1(xyz);
  result -= glyph_o_2(xyz);
  return result;
}

float glyph_s_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(739.00,1070.00)))) { return 0.0; }

  if (xyz.y >= 1045.50 && xyz.y <= 1070.00 && (xyz.x < interpQuadratic(vec2(240.00,1045.50),vec2(309.00,1070.00),vec2(397.00,1070.00),xyz.y) || xyz.x > interpQuadratic(vec2(580.50,1037.50),vec2(499.00,1070.00),vec2(397.00,1070.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1037.50 && xyz.y < 1045.50 && (xyz.x < interpQuadratic(vec2(124.00,979.50),vec2(171.00,1021.00),vec2(240.00,1045.50),xyz.y) || xyz.x > interpQuadratic(vec2(580.50,1037.50),vec2(499.00,1070.00),vec2(397.00,1070.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 979.50 && xyz.y < 1037.50 && (xyz.x < interpQuadratic(vec2(124.00,979.50),vec2(171.00,1021.00),vec2(240.00,1045.50),xyz.y) || xyz.x > interpQuadratic(vec2(721.00,949.00),vec2(662.00,1005.00),vec2(580.50,1037.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 949.00 && xyz.y < 979.50 && (xyz.x < interpQuadratic(vec2(52.50,883.00),vec2(77.00,938.00),vec2(124.00,979.50),xyz.y) || xyz.x > interpQuadratic(vec2(721.00,949.00),vec2(662.00,1005.00),vec2(580.50,1037.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 934.00 && xyz.y < 949.00 && (xyz.x < interpQuadratic(vec2(52.50,883.00),vec2(77.00,938.00),vec2(124.00,979.50),xyz.y) || xyz.x > interpLine(vec2(680.00,882.00),vec2(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 922.00 && xyz.y < 934.00 && (xyz.x < interpQuadratic(vec2(52.50,883.00),vec2(77.00,938.00),vec2(124.00,979.50),xyz.y) || (xyz.x > interpQuadratic(vec2(318.00,922.00),vec2(355.00,934.00),vec2(401.00,934.00),xyz.y) && xyz.x < interpQuadratic(vec2(493.00,922.00),vec2(454.00,934.00),vec2(401.00,934.00),xyz.y)) || xyz.x > interpLine(vec2(680.00,882.00),vec2(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 896.00 && xyz.y < 922.00 && (xyz.x < interpQuadratic(vec2(52.50,883.00),vec2(77.00,938.00),vec2(124.00,979.50),xyz.y) || (xyz.x > interpQuadratic(vec2(255.00,889.50),vec2(281.00,910.00),vec2(318.00,922.00),xyz.y) && xyz.x < interpQuadratic(vec2(560.00,896.00),vec2(532.00,910.00),vec2(493.00,922.00),xyz.y)) || xyz.x > interpLine(vec2(680.00,882.00),vec2(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 889.50 && xyz.y < 896.00 && (xyz.x < interpQuadratic(vec2(52.50,883.00),vec2(77.00,938.00),vec2(124.00,979.50),xyz.y) || (xyz.x > interpQuadratic(vec2(255.00,889.50),vec2(281.00,910.00),vec2(318.00,922.00),xyz.y) && xyz.x < interpQuadratic(vec2(608.00,871.00),vec2(588.00,882.00),vec2(560.00,896.00),xyz.y)) || xyz.x > interpLine(vec2(680.00,882.00),vec2(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 883.00 && xyz.y < 889.50 && (xyz.x < interpQuadratic(vec2(52.50,883.00),vec2(77.00,938.00),vec2(124.00,979.50),xyz.y) || (xyz.x > interpQuadratic(vec2(215.00,842.00),vec2(229.00,869.00),vec2(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2(608.00,871.00),vec2(588.00,882.00),vec2(560.00,896.00),xyz.y)) || xyz.x > interpLine(vec2(680.00,882.00),vec2(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 882.00 && xyz.y < 883.00 && (xyz.x < interpQuadratic(vec2(28.00,769.00),vec2(28.00,828.00),vec2(52.50,883.00),xyz.y) || (xyz.x > interpQuadratic(vec2(215.00,842.00),vec2(229.00,869.00),vec2(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2(608.00,871.00),vec2(588.00,882.00),vec2(560.00,896.00),xyz.y)) || xyz.x > interpLine(vec2(680.00,882.00),vec2(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 871.00 && xyz.y < 882.00 && (xyz.x < interpQuadratic(vec2(28.00,769.00),vec2(28.00,828.00),vec2(52.50,883.00),xyz.y) || (xyz.x > interpQuadratic(vec2(215.00,842.00),vec2(229.00,869.00),vec2(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2(608.00,871.00),vec2(588.00,882.00),vec2(560.00,896.00),xyz.y)) || xyz.x > interpQuadratic(vec2(643.00,860.00),vec2(668.00,859.00),vec2(680.00,882.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 860.00 && xyz.y < 871.00 && (xyz.x < interpQuadratic(vec2(28.00,769.00),vec2(28.00,828.00),vec2(52.50,883.00),xyz.y) || (xyz.x > interpQuadratic(vec2(215.00,842.00),vec2(229.00,869.00),vec2(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2(643.00,860.00),vec2(628.00,860.00),vec2(608.00,871.00),xyz.y)) || xyz.x > interpQuadratic(vec2(643.00,860.00),vec2(668.00,859.00),vec2(680.00,882.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 859.00 && xyz.y < 860.00 && (xyz.x < interpQuadratic(vec2(28.00,769.00),vec2(28.00,828.00),vec2(52.50,883.00),xyz.y) || (xyz.x > interpQuadratic(vec2(215.00,842.00),vec2(229.00,869.00),vec2(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2(643.00,860.00),vec2(668.00,859.00),vec2(680.00,882.00),xyz.y)) || xyz.x > interpQuadratic(vec2(643.00,860.00),vec2(668.00,859.00),vec2(680.00,882.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 842.00 && xyz.y < 859.00 && (xyz.x < interpQuadratic(vec2(28.00,769.00),vec2(28.00,828.00),vec2(52.50,883.00),xyz.y) || xyz.x > interpQuadratic(vec2(215.00,842.00),vec2(229.00,869.00),vec2(255.00,889.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 783.00 && xyz.y < 842.00 && (xyz.x < interpQuadratic(vec2(28.00,769.00),vec2(28.00,828.00),vec2(52.50,883.00),xyz.y) || xyz.x > interpQuadratic(vec2(201.00,783.00),vec2(201.00,815.00),vec2(215.00,842.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 769.00 && xyz.y < 783.00 && (xyz.x < interpQuadratic(vec2(28.00,769.00),vec2(28.00,828.00),vec2(52.50,883.00),xyz.y) || xyz.x > interpQuadratic(vec2(224.00,716.50),vec2(201.00,743.00),vec2(201.00,783.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 716.50 && xyz.y < 769.00 && (xyz.x < interpQuadratic(vec2(51.00,655.50),vec2(28.00,702.00),vec2(28.00,769.00),xyz.y) || xyz.x > interpQuadratic(vec2(224.00,716.50),vec2(201.00,743.00),vec2(201.00,783.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 670.50 && xyz.y < 716.50 && (xyz.x < interpQuadratic(vec2(51.00,655.50),vec2(28.00,702.00),vec2(28.00,769.00),xyz.y) || xyz.x > interpQuadratic(vec2(285.00,670.50),vec2(247.00,690.00),vec2(224.00,716.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 655.50 && xyz.y < 670.50 && (xyz.x < interpQuadratic(vec2(51.00,655.50),vec2(28.00,702.00),vec2(28.00,769.00),xyz.y) || xyz.x > interpQuadratic(vec2(371.00,636.00),vec2(323.00,651.00),vec2(285.00,670.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 636.00 && xyz.y < 655.50 && (xyz.x < interpQuadratic(vec2(112.00,577.00),vec2(74.00,609.00),vec2(51.00,655.50),xyz.y) || xyz.x > interpQuadratic(vec2(371.00,636.00),vec2(323.00,651.00),vec2(285.00,670.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 604.00 && xyz.y < 636.00 && (xyz.x < interpQuadratic(vec2(112.00,577.00),vec2(74.00,609.00),vec2(51.00,655.50),xyz.y) || xyz.x > interpQuadratic(vec2(470.00,604.00),vec2(419.00,621.00),vec2(371.00,636.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 577.00 && xyz.y < 604.00 && (xyz.x < interpQuadratic(vec2(112.00,577.00),vec2(74.00,609.00),vec2(51.00,655.50),xyz.y) || xyz.x > interpQuadratic(vec2(569.00,566.50),vec2(521.00,587.00),vec2(470.00,604.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 566.50 && xyz.y < 577.00 && (xyz.x < interpQuadratic(vec2(198.50,524.00),vec2(150.00,545.00),vec2(112.00,577.00),xyz.y) || xyz.x > interpQuadratic(vec2(569.00,566.50),vec2(521.00,587.00),vec2(470.00,604.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 524.00 && xyz.y < 566.50 && (xyz.x < interpQuadratic(vec2(198.50,524.00),vec2(150.00,545.00),vec2(112.00,577.00),xyz.y) || xyz.x > interpQuadratic(vec2(655.00,515.50),vec2(617.00,546.00),vec2(569.00,566.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 515.50 && xyz.y < 524.00 && (xyz.x < interpQuadratic(vec2(298.00,487.00),vec2(247.00,503.00),vec2(198.50,524.00),xyz.y) || xyz.x > interpQuadratic(vec2(655.00,515.50),vec2(617.00,546.00),vec2(569.00,566.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 487.00 && xyz.y < 515.50 && (xyz.x < interpQuadratic(vec2(298.00,487.00),vec2(247.00,503.00),vec2(198.50,524.00),xyz.y) || xyz.x > interpQuadratic(vec2(716.00,440.50),vec2(693.00,485.00),vec2(655.00,515.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 456.00 && xyz.y < 487.00 && (xyz.x < interpQuadratic(vec2(397.50,456.00),vec2(349.00,471.00),vec2(298.00,487.00),xyz.y) || xyz.x > interpQuadratic(vec2(716.00,440.50),vec2(693.00,485.00),vec2(655.00,515.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 440.50 && xyz.y < 456.00 && (xyz.x < interpQuadratic(vec2(484.00,421.00),vec2(446.00,441.00),vec2(397.50,456.00),xyz.y) || xyz.x > interpQuadratic(vec2(716.00,440.50),vec2(693.00,485.00),vec2(655.00,515.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 421.00 && xyz.y < 440.50 && (xyz.x < interpQuadratic(vec2(484.00,421.00),vec2(446.00,441.00),vec2(397.50,456.00),xyz.y) || xyz.x > interpQuadratic(vec2(739.00,333.00),vec2(739.00,396.00),vec2(716.00,440.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 373.00 && xyz.y < 421.00 && (xyz.x < interpQuadratic(vec2(545.00,373.00),vec2(522.00,401.00),vec2(484.00,421.00),xyz.y) || xyz.x > interpQuadratic(vec2(739.00,333.00),vec2(739.00,396.00),vec2(716.00,440.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 333.00 && xyz.y < 373.00 && (xyz.x < interpQuadratic(vec2(568.00,302.00),vec2(568.00,345.00),vec2(545.00,373.00),xyz.y) || xyz.x > interpQuadratic(vec2(739.00,333.00),vec2(739.00,396.00),vec2(716.00,440.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 302.00 && xyz.y < 333.00 && (xyz.x < interpQuadratic(vec2(568.00,302.00),vec2(568.00,345.00),vec2(545.00,373.00),xyz.y) || xyz.x > interpQuadratic(vec2(713.50,200.50),vec2(739.00,261.00),vec2(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 236.00 && xyz.y < 302.00 && (xyz.x < interpQuadratic(vec2(555.00,236.00),vec2(568.00,267.00),vec2(568.00,302.00),xyz.y) || xyz.x > interpQuadratic(vec2(713.50,200.50),vec2(739.00,261.00),vec2(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 223.00 && xyz.y < 236.00 && (xyz.x < interpQuadratic(vec2(515.50,181.50),vec2(542.00,205.00),vec2(555.00,236.00),xyz.y) || xyz.x > interpQuadratic(vec2(713.50,200.50),vec2(739.00,261.00),vec2(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 215.50 && xyz.y < 223.00 && (xyz.x < interpQuadratic(vec2(62.50,215.50),vec2(74.00,223.00),vec2(93.00,223.00),xyz.y) || (xyz.x > interpQuadratic(vec2(131.50,208.50),vec2(111.00,223.00),vec2(93.00,223.00),xyz.y) && xyz.x < interpQuadratic(vec2(515.50,181.50),vec2(542.00,205.00),vec2(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2(713.50,200.50),vec2(739.00,261.00),vec2(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 208.50 && xyz.y < 215.50 && (xyz.x < interpQuadratic(vec2(43.00,195.00),vec2(51.00,208.00),vec2(62.50,215.50),xyz.y) || (xyz.x > interpQuadratic(vec2(131.50,208.50),vec2(111.00,223.00),vec2(93.00,223.00),xyz.y) && xyz.x < interpQuadratic(vec2(515.50,181.50),vec2(542.00,205.00),vec2(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2(713.50,200.50),vec2(739.00,261.00),vec2(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 200.50 && xyz.y < 208.50 && (xyz.x < interpQuadratic(vec2(43.00,195.00),vec2(51.00,208.00),vec2(62.50,215.50),xyz.y) || (xyz.x > interpQuadratic(vec2(181.00,176.50),vec2(152.00,194.00),vec2(131.50,208.50),xyz.y) && xyz.x < interpQuadratic(vec2(515.50,181.50),vec2(542.00,205.00),vec2(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2(713.50,200.50),vec2(739.00,261.00),vec2(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 195.00 && xyz.y < 200.50 && (xyz.x < interpQuadratic(vec2(43.00,195.00),vec2(51.00,208.00),vec2(62.50,215.50),xyz.y) || (xyz.x > interpQuadratic(vec2(181.00,176.50),vec2(152.00,194.00),vec2(131.50,208.50),xyz.y) && xyz.x < interpQuadratic(vec2(515.50,181.50),vec2(542.00,205.00),vec2(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2(637.50,95.50),vec2(688.00,140.00),vec2(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 181.50 && xyz.y < 195.00 && (xyz.x < interpLine(vec2(0.00,126.00),vec2(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2(181.00,176.50),vec2(152.00,194.00),vec2(131.50,208.50),xyz.y) && xyz.x < interpQuadratic(vec2(515.50,181.50),vec2(542.00,205.00),vec2(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2(637.50,95.50),vec2(688.00,140.00),vec2(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 176.50 && xyz.y < 181.50 && (xyz.x < interpLine(vec2(0.00,126.00),vec2(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2(181.00,176.50),vec2(152.00,194.00),vec2(131.50,208.50),xyz.y) && xyz.x < interpQuadratic(vec2(449.00,144.50),vec2(489.00,158.00),vec2(515.50,181.50),xyz.y)) || xyz.x > interpQuadratic(vec2(637.50,95.50),vec2(688.00,140.00),vec2(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 145.00 && xyz.y < 176.50 && (xyz.x < interpLine(vec2(0.00,126.00),vec2(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2(251.50,145.00),vec2(210.00,159.00),vec2(181.00,176.50),xyz.y) && xyz.x < interpQuadratic(vec2(449.00,144.50),vec2(489.00,158.00),vec2(515.50,181.50),xyz.y)) || xyz.x > interpQuadratic(vec2(637.50,95.50),vec2(688.00,140.00),vec2(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 144.50 && xyz.y < 145.00 && (xyz.x < interpLine(vec2(0.00,126.00),vec2(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2(356.00,131.00),vec2(293.00,131.00),vec2(251.50,145.00),xyz.y) && xyz.x < interpQuadratic(vec2(449.00,144.50),vec2(489.00,158.00),vec2(515.50,181.50),xyz.y)) || xyz.x > interpQuadratic(vec2(637.50,95.50),vec2(688.00,140.00),vec2(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 131.00 && xyz.y < 144.50 && (xyz.x < interpLine(vec2(0.00,126.00),vec2(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2(356.00,131.00),vec2(293.00,131.00),vec2(251.50,145.00),xyz.y) && xyz.x < interpQuadratic(vec2(356.00,131.00),vec2(409.00,131.00),vec2(449.00,144.50),xyz.y)) || xyz.x > interpQuadratic(vec2(637.50,95.50),vec2(688.00,140.00),vec2(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 126.00 && xyz.y < 131.00 && (xyz.x < interpLine(vec2(0.00,126.00),vec2(43.00,195.00),xyz.y) || xyz.x > interpQuadratic(vec2(637.50,95.50),vec2(688.00,140.00),vec2(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 95.50 && xyz.y < 126.00 && (xyz.x < interpQuadratic(vec2(149.50,35.50),vec2(61.00,71.00),vec2(0.00,126.00),xyz.y) || xyz.x > interpQuadratic(vec2(637.50,95.50),vec2(688.00,140.00),vec2(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 35.50 && xyz.y < 95.50 && (xyz.x < interpQuadratic(vec2(149.50,35.50),vec2(61.00,71.00),vec2(0.00,126.00),xyz.y) || xyz.x > interpQuadratic(vec2(514.50,25.50),vec2(587.00,51.00),vec2(637.50,95.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 25.50 && xyz.y < 35.50 && (xyz.x < interpQuadratic(vec2(347.00,0.00),vec2(238.00,0.00),vec2(149.50,35.50),xyz.y) || xyz.x > interpQuadratic(vec2(514.50,25.50),vec2(587.00,51.00),vec2(637.50,95.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 25.50 && (xyz.x < interpQuadratic(vec2(347.00,0.00),vec2(238.00,0.00),vec2(149.50,35.50),xyz.y) || xyz.x > interpQuadratic(vec2(347.00,0.00),vec2(442.00,0.00),vec2(514.50,25.50),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_s(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(63.00,-16.00,0.0))) || any(greaterThan(xyz, vec3(802.00,1054.00,height)))) { return 0.0; }
  xyz -= vec3(63.00,-16.00,0.0);
  float result = glyph_s_1(xyz);
  return result;
}

float glyph_t_1(vec3 xyz) {
  if (any(lessThan(xyz.xy, vec2(0.00,0.00))) || any(greaterThan(xyz.xy, vec2(670.00,1390.00)))) { return 0.0; }

  if (xyz.y >= 1380.00 && xyz.y <= 1390.00 && (xyz.x < interpQuadratic(vec2(225.50,1380.00),vec2(237.00,1390.00),vec2(254.00,1390.00),xyz.y) || xyz.x > interpLine(vec2(346.00,1033.00),vec2(346.00,1390.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1355.00 && xyz.y < 1380.00 && (xyz.x < interpQuadratic(vec2(212.00,1355.00),vec2(214.00,1370.00),vec2(225.50,1380.00),xyz.y) || xyz.x > interpLine(vec2(346.00,1033.00),vec2(346.00,1390.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1035.00 && xyz.y < 1355.00 && (xyz.x < interpLine(vec2(170.00,1035.00),vec2(212.00,1355.00),xyz.y) || xyz.x > interpLine(vec2(346.00,1033.00),vec2(346.00,1390.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1033.00 && xyz.y < 1035.00 && (xyz.x < interpLine(vec2(0.00,1013.00),vec2(170.00,1035.00),xyz.y) || xyz.x > interpLine(vec2(346.00,1033.00),vec2(346.00,1390.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1013.00 && xyz.y < 1033.00 && (xyz.x < interpLine(vec2(0.00,1013.00),vec2(170.00,1035.00),xyz.y) || xyz.x > interpLine(vec2(643.00,901.00),vec2(643.00,1033.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 941.00 && xyz.y < 1013.00 && (xyz.x < interpLine(vec2(0.00,941.00),vec2(0.00,1013.00),xyz.y) || xyz.x > interpLine(vec2(643.00,901.00),vec2(643.00,1033.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 911.00 && xyz.y < 941.00 && (xyz.x < interpQuadratic(vec2(11.50,911.00),vec2(0.00,921.00),vec2(0.00,941.00),xyz.y) || xyz.x > interpLine(vec2(643.00,901.00),vec2(643.00,1033.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 901.00 && xyz.y < 911.00 && (xyz.x < interpQuadratic(vec2(39.00,901.00),vec2(23.00,901.00),vec2(11.50,911.00),xyz.y) || xyz.x > interpLine(vec2(643.00,901.00),vec2(643.00,1033.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 278.00 && xyz.y < 901.00 && (xyz.x < interpLine(vec2(164.00,266.00),vec2(164.00,901.00),xyz.y) || xyz.x > interpLine(vec2(346.00,278.00),vec2(346.00,901.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 266.00 && xyz.y < 278.00 && (xyz.x < interpLine(vec2(164.00,266.00),vec2(164.00,901.00),xyz.y) || xyz.x > interpQuadratic(vec2(378.00,180.50),vec2(346.00,212.00),vec2(346.00,278.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 198.00 && xyz.y < 266.00 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || xyz.x > interpQuadratic(vec2(378.00,180.50),vec2(346.00,212.00),vec2(346.00,278.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 190.50 && xyz.y < 198.00 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2(378.00,180.50),vec2(346.00,212.00),vec2(346.00,278.00),xyz.y) && xyz.x < interpQuadratic(vec2(572.00,190.50),vec2(583.00,198.00),vec2(591.00,198.00),xyz.y)) || xyz.x > interpQuadratic(vec2(617.00,181.00),vec2(605.00,198.00),vec2(591.00,198.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 181.00 && xyz.y < 190.50 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2(378.00,180.50),vec2(346.00,212.00),vec2(346.00,278.00),xyz.y) && xyz.x < interpQuadratic(vec2(545.50,173.50),vec2(561.00,183.00),vec2(572.00,190.50),xyz.y)) || xyz.x > interpQuadratic(vec2(617.00,181.00),vec2(605.00,198.00),vec2(591.00,198.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 180.50 && xyz.y < 181.00 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2(378.00,180.50),vec2(346.00,212.00),vec2(346.00,278.00),xyz.y) && xyz.x < interpQuadratic(vec2(545.50,173.50),vec2(561.00,183.00),vec2(572.00,190.50),xyz.y)) || xyz.x > interpLine(vec2(670.00,94.00),vec2(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 173.50 && xyz.y < 180.50 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2(460.00,149.00),vec2(410.00,149.00),vec2(378.00,180.50),xyz.y) && xyz.x < interpQuadratic(vec2(545.50,173.50),vec2(561.00,183.00),vec2(572.00,190.50),xyz.y)) || xyz.x > interpLine(vec2(670.00,94.00),vec2(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 156.50 && xyz.y < 173.50 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2(460.00,149.00),vec2(410.00,149.00),vec2(378.00,180.50),xyz.y) && xyz.x < interpQuadratic(vec2(509.50,156.50),vec2(530.00,164.00),vec2(545.50,173.50),xyz.y)) || xyz.x > interpLine(vec2(670.00,94.00),vec2(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 149.00 && xyz.y < 156.50 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2(460.00,149.00),vec2(410.00,149.00),vec2(378.00,180.50),xyz.y) && xyz.x < interpQuadratic(vec2(460.00,149.00),vec2(489.00,149.00),vec2(509.50,156.50),xyz.y)) || xyz.x > interpLine(vec2(670.00,94.00),vec2(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 94.00 && xyz.y < 149.00 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || xyz.x > interpLine(vec2(670.00,94.00),vec2(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 68.50 && xyz.y < 94.00 && (xyz.x < interpQuadratic(vec2(230.00,68.50),vec2(164.00,137.00),vec2(164.00,266.00),xyz.y) || xyz.x > interpQuadratic(vec2(556.50,25.00),vec2(623.00,50.00),vec2(670.00,94.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 25.00 && xyz.y < 68.50 && (xyz.x < interpQuadratic(vec2(419.00,0.00),vec2(296.00,0.00),vec2(230.00,68.50),xyz.y) || xyz.x > interpQuadratic(vec2(556.50,25.00),vec2(623.00,50.00),vec2(670.00,94.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 25.00 && (xyz.x < interpQuadratic(vec2(419.00,0.00),vec2(296.00,0.00),vec2(230.00,68.50),xyz.y) || xyz.x > interpQuadratic(vec2(419.00,0.00),vec2(490.00,0.00),vec2(556.50,25.00),xyz.y))) { return 0.0; }
  return 1.0;
}

float glyph_t(float height, vec3 xyz) {
  if (any(lessThan(xyz, vec3(45.00,-16.00,0.0))) || any(greaterThan(xyz, vec3(715.00,1374.00,height)))) { return 0.0; }
  xyz -= vec3(45.00,-16.00,0.0);
  float result = glyph_t_1(xyz);
  return result;
}

float textMessage(float emSize, float height, in vec3 xyz) {
  xyz *= vec3(1649.00/emSize,1649.00/emSize,1.0);
  float result = 0.0;
  result += glyph_I(height, xyz-vec3(0.00,0,0));
  result += glyph_R(height, xyz-vec3(628.00,0,0));
  result += glyph_M(height, xyz-vec3(2146.00,0,0));
  result += glyph_F(height, xyz-vec3(4030.00,0,0));
  result += glyph_f(height, xyz-vec3(5584.00,0,0));
  result += glyph_o(height, xyz-vec3(6274.00,0,0));
  result += glyph_n(height, xyz-vec3(7412.00,0,0));
  result += glyph_t(height, xyz-vec3(8550.00,0,0));
  result += glyph_s(height, xyz-vec3(9313.00,0,0));
  return result;
}

void mainModel4(out vec4 materials, in vec3 xyz) {
  xyz += vec3(3.0, 0.0, 1.5);
  materials[0] = textMessage(1.0, 0.5, xyz);
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

fn blinnLoop(A: vec2f, B: vec2f, C: vec2f) -> f32 {
  let v0 = C - A;
  let v1 = B - A;
  let v2 = vec2f(0.75,0.5) - A;
  let dot00 = dot(v0, v0);
  let dot01 = dot(v0, v1);
  let dot02 = dot(v0, v2);
  let dot11 = dot(v1, v1);
  let dot12 = dot(v1, v2);
  let invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
  let u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  let v = (dot00 * dot12 - dot01 * dot02) * invDenom;
  let w = (1.0 - u - v);
  return w;
}

fn interpLine(A: vec2f, B: vec2f, y: f32) -> f32 {
  let p = (y - A.y) / (B.y - A.y);
  return p*(B.x-A.x) + A.x;
}

fn interpQuadratic(p0: vec2f, p1: vec2f, p2: vec2f, y: f32) -> f32 {
  let a = p2.y + p0.y - 2.0*p1.y;
  let b = 2.0 * (p1.y - p0.y);
  let c = p0.y - y;
  if (b*b < 4.0*a*c) { return 0.0; } 
  let det = sqrt(b*b - 4.0*a*c);
  var t = (-b + det) / (2.0 * a);
  let t2 = (-b - det) / (2.0 * a);
  if (t2 >= 0.0 && t2 <= 1.0) {
    t = t2;
  }
  let x = (1.0-t)*(1.0-t)*p0.x + 2.0*(1.0-t)*t*p1.x + t*t*p2.x;
  return x;
}

fn glyph_F_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(904.00,1467.00))) { return 0.0; }

  if (xyz.y >= 1306.00 && xyz.y <= 1467.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2f(904.00,1306.00),vec2f(904.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 793.00 && xyz.y < 1306.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2f(200.00,793.00),vec2f(200.00,1306.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 631.00 && xyz.y < 793.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2f(802.00,631.00),vec2f(802.00,793.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 631.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2f(200.00,0.00),vec2f(200.00,631.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_F(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(178.00,0.00,0.0)) || any(xyz_in > vec3f(1082.00,1467.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(178.00,0.00,0.0);
  return glyph_F_1(xyz);
}

fn glyph_I_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(199.00,1467.00))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y <= 1467.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpLine(vec2f(199.00,0.00),vec2f(199.00,1467.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_I(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(215.00,0.00,0.0)) || any(xyz_in > vec3f(414.00,1467.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(215.00,0.00,0.0);
  return glyph_I_1(xyz);
}

fn glyph_M_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(1527.00,1467.00))) { return 0.0; }

  if (xyz.y >= 1462.50 && xyz.y <= 1467.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(188.00,1462.00),vec2f(174.00,1467.00),vec2f(147.00,1467.00),xyz.y) && xyz.x < interpQuadratic(vec2f(1339.50,1462.50),vec2f(1354.00,1467.00),vec2f(1381.00,1467.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1462.00 && xyz.y < 1462.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(188.00,1462.00),vec2f(174.00,1467.00),vec2f(147.00,1467.00),xyz.y) && xyz.x < interpQuadratic(vec2f(1312.00,1434.00),vec2f(1325.00,1458.00),vec2f(1339.50,1462.50),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1434.00 && xyz.y < 1462.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(215.00,1434.00),vec2f(202.00,1457.00),vec2f(188.00,1462.00),xyz.y) && xyz.x < interpQuadratic(vec2f(1312.00,1434.00),vec2f(1325.00,1458.00),vec2f(1339.50,1462.50),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1178.00 && xyz.y < 1434.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(722.00,530.00),vec2f(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2f(815.00,531.00),vec2f(1312.00,1434.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1175.00 && xyz.y < 1178.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(172.50,1125.50),vec2f(171.00,1151.00),vec2f(168.00,1178.00),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2f(722.00,530.00),vec2f(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2f(815.00,531.00),vec2f(1312.00,1434.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1125.50 && xyz.y < 1175.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(172.50,1125.50),vec2f(171.00,1151.00),vec2f(168.00,1178.00),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2f(722.00,530.00),vec2f(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2f(815.00,531.00),vec2f(1312.00,1434.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpQuadratic(vec2f(1354.00,1124.50),vec2f(1355.00,1149.00),vec2f(1357.00,1175.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1124.50 && xyz.y < 1125.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(174.00,1078.00),vec2f(174.00,1100.00),vec2f(172.50,1125.50),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2f(722.00,530.00),vec2f(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2f(815.00,531.00),vec2f(1312.00,1434.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpQuadratic(vec2f(1354.00,1124.50),vec2f(1355.00,1149.00),vec2f(1357.00,1175.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1078.00 && xyz.y < 1124.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(174.00,1078.00),vec2f(174.00,1100.00),vec2f(172.50,1125.50),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2f(722.00,530.00),vec2f(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2f(815.00,531.00),vec2f(1312.00,1434.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpQuadratic(vec2f(1353.00,1078.00),vec2f(1353.00,1100.00),vec2f(1354.00,1124.50),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 531.00 && xyz.y < 1078.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(174.00,0.00),vec2f(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2f(722.00,530.00),vec2f(215.00,1434.00),xyz.y) && xyz.x < interpLine(vec2f(815.00,531.00),vec2f(1312.00,1434.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2f(1353.00,0.00),vec2f(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 530.00 && xyz.y < 531.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(174.00,0.00),vec2f(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2f(722.00,530.00),vec2f(215.00,1434.00),xyz.y) && xyz.x < interpQuadratic(vec2f(789.50,479.00),vec2f(801.00,505.00),vec2f(815.00,531.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2f(1353.00,0.00),vec2f(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 479.00 && xyz.y < 530.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(174.00,0.00),vec2f(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpQuadratic(vec2f(747.00,478.50),vec2f(736.00,505.00),vec2f(722.00,530.00),xyz.y) && xyz.x < interpQuadratic(vec2f(789.50,479.00),vec2f(801.00,505.00),vec2f(815.00,531.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2f(1353.00,0.00),vec2f(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 478.50 && xyz.y < 479.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(174.00,0.00),vec2f(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpQuadratic(vec2f(747.00,478.50),vec2f(736.00,505.00),vec2f(722.00,530.00),xyz.y) && xyz.x < interpQuadratic(vec2f(768.00,424.00),vec2f(778.00,453.00),vec2f(789.50,479.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2f(1353.00,0.00),vec2f(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 424.00 && xyz.y < 478.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(174.00,0.00),vec2f(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpQuadratic(vec2f(768.00,424.00),vec2f(758.00,452.00),vec2f(747.00,478.50),xyz.y) && xyz.x < interpQuadratic(vec2f(768.00,424.00),vec2f(778.00,453.00),vec2f(789.50,479.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2f(1353.00,0.00),vec2f(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 257.00 && xyz.y < 424.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(174.00,0.00),vec2f(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2f(682.00,257.00),vec2f(168.00,1178.00),xyz.y)) || (xyz.x > interpLine(vec2f(854.00,257.00),vec2f(1357.00,1175.00),xyz.y) && xyz.x < interpLine(vec2f(1353.00,0.00),vec2f(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 211.00 && xyz.y < 257.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(174.00,0.00),vec2f(174.00,1078.00),xyz.y) && xyz.x < interpQuadratic(vec2f(754.00,211.00),vec2f(708.00,211.00),vec2f(682.00,257.00),xyz.y)) || (xyz.x > interpQuadratic(vec2f(783.00,211.00),vec2f(828.00,211.00),vec2f(854.00,257.00),xyz.y) && xyz.x < interpLine(vec2f(1353.00,0.00),vec2f(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 211.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(174.00,0.00),vec2f(174.00,1078.00),xyz.y) && xyz.x < interpLine(vec2f(1353.00,0.00),vec2f(1353.00,1078.00),xyz.y)) || xyz.x > interpLine(vec2f(1527.00,0.00),vec2f(1527.00,1467.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_M(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(178.00,0.00,0.0)) || any(xyz_in > vec3f(1705.00,1467.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(178.00,0.00,0.0);
  return glyph_M_1(xyz);
}

fn glyph_R_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(1084.00,1467.00))) { return 0.0; }

  if (xyz.y >= 1439.00 && xyz.y <= 1467.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(654.50,1439.00),vec2f(553.00,1467.00),vec2f(414.00,1467.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1358.00 && xyz.y < 1439.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(822.00,1358.00),vec2f(756.00,1411.00),vec2f(654.50,1439.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1229.50 && xyz.y < 1358.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(920.00,1229.50),vec2f(888.00,1305.00),vec2f(822.00,1358.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1061.00 && xyz.y < 1229.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(952.00,1061.00),vec2f(952.00,1154.00),vec2f(920.00,1229.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 915.50 && xyz.y < 1061.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(927.50,915.50),vec2f(952.00,983.00),vec2f(952.00,1061.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 794.00 && xyz.y < 915.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(856.50,794.00),vec2f(903.00,848.00),vec2f(927.50,915.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 702.50 && xyz.y < 794.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(742.50,702.50),vec2f(810.00,740.00),vec2f(856.50,794.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 645.00 && xyz.y < 702.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(591.00,645.00),vec2f(675.00,665.00),vec2f(742.50,702.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 612.00 && xyz.y < 645.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || xyz.x > interpQuadratic(vec2f(656.00,583.00),vec2f(628.00,623.00),vec2f(591.00,645.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 601.50 && xyz.y < 612.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(197.00,0.00),vec2f(197.00,612.00),xyz.y) && xyz.x < interpQuadratic(vec2f(409.50,601.50),vec2f(389.00,612.00),vec2f(348.00,612.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(656.00,583.00),vec2f(628.00,623.00),vec2f(591.00,645.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 583.00 && xyz.y < 601.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(197.00,0.00),vec2f(197.00,612.00),xyz.y) && xyz.x < interpQuadratic(vec2f(447.00,566.00),vec2f(430.00,591.00),vec2f(409.50,601.50),xyz.y)) || xyz.x > interpQuadratic(vec2f(656.00,583.00),vec2f(628.00,623.00),vec2f(591.00,645.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 566.00 && xyz.y < 583.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(197.00,0.00),vec2f(197.00,612.00),xyz.y) && xyz.x < interpQuadratic(vec2f(447.00,566.00),vec2f(430.00,591.00),vec2f(409.50,601.50),xyz.y)) || xyz.x > interpLine(vec2f(1084.00,0.00),vec2f(656.00,583.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 42.00 && xyz.y < 566.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(197.00,0.00),vec2f(197.00,612.00),xyz.y) && xyz.x < interpLine(vec2f(828.00,42.00),vec2f(447.00,566.00),xyz.y)) || xyz.x > interpLine(vec2f(1084.00,0.00),vec2f(656.00,583.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 42.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1467.00),xyz.y) || (xyz.x > interpLine(vec2f(197.00,0.00),vec2f(197.00,612.00),xyz.y) && xyz.x < interpQuadratic(vec2f(908.00,0.00),vec2f(854.00,0.00),vec2f(828.00,42.00),xyz.y)) || xyz.x > interpLine(vec2f(1084.00,0.00),vec2f(656.00,583.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_R_2(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(197.00,757.00)) || any(xyz.xy > vec2f(758.00,1311.00))) { return 0.0; }

  if (xyz.y >= 1244.50 && xyz.y <= 1311.00 && (xyz.x < interpLine(vec2f(197.00,757.00),vec2f(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2f(671.50,1244.50),vec2f(585.00,1311.00),vec2f(414.00,1311.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1047.00 && xyz.y < 1244.50 && (xyz.x < interpLine(vec2f(197.00,757.00),vec2f(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2f(758.00,1047.00),vec2f(758.00,1178.00),vec2f(671.50,1244.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 929.00 && xyz.y < 1047.00 && (xyz.x < interpLine(vec2f(197.00,757.00),vec2f(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2f(735.50,929.00),vec2f(758.00,982.00),vec2f(758.00,1047.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 837.50 && xyz.y < 929.00 && (xyz.x < interpLine(vec2f(197.00,757.00),vec2f(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2f(668.50,837.50),vec2f(713.00,876.00),vec2f(735.50,929.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 778.00 && xyz.y < 837.50 && (xyz.x < interpLine(vec2f(197.00,757.00),vec2f(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2f(558.00,778.00),vec2f(624.00,799.00),vec2f(668.50,837.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 757.00 && xyz.y < 778.00 && (xyz.x < interpLine(vec2f(197.00,757.00),vec2f(197.00,1311.00),xyz.y) || xyz.x > interpQuadratic(vec2f(405.00,757.00),vec2f(492.00,757.00),vec2f(558.00,778.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_R(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(199.00,0.00,0.0)) || any(xyz_in > vec3f(1283.00,1467.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(199.00,0.00,0.0);
  let result = glyph_R_1(xyz);
  return result - glyph_R_2(xyz);
}

fn glyph_f_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(649.00,1489.00))) { return 0.0; }

  if (xyz.y >= 1468.00 && xyz.y <= 1489.00 && (xyz.x < interpQuadratic(vec2f(372.50,1464.50),vec2f(438.00,1489.00),vec2f(520.00,1489.00),xyz.y) || xyz.x > interpQuadratic(vec2f(649.00,1468.00),vec2f(590.00,1489.00),vec2f(520.00,1489.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1464.50 && xyz.y < 1468.00 && (xyz.x < interpQuadratic(vec2f(372.50,1464.50),vec2f(438.00,1489.00),vec2f(520.00,1489.00),xyz.y) || xyz.x > interpLine(vec2f(645.00,1377.00),vec2f(649.00,1468.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1392.50 && xyz.y < 1464.50 && (xyz.x < interpQuadratic(vec2f(260.50,1392.50),vec2f(307.00,1440.00),vec2f(372.50,1464.50),xyz.y) || xyz.x > interpLine(vec2f(645.00,1377.00),vec2f(649.00,1468.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1377.00 && xyz.y < 1392.50 && (xyz.x < interpQuadratic(vec2f(188.50,1275.50),vec2f(214.00,1345.00),vec2f(260.50,1392.50),xyz.y) || xyz.x > interpLine(vec2f(645.00,1377.00),vec2f(649.00,1468.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1353.00 && xyz.y < 1377.00 && (xyz.x < interpQuadratic(vec2f(188.50,1275.50),vec2f(214.00,1345.00),vec2f(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2f(627.50,1353.00),vec2f(644.00,1357.00),vec2f(645.00,1377.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1349.00 && xyz.y < 1353.00 && (xyz.x < interpQuadratic(vec2f(188.50,1275.50),vec2f(214.00,1345.00),vec2f(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2f(581.00,1349.00),vec2f(611.00,1349.00),vec2f(627.50,1353.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1336.50 && xyz.y < 1349.00 && (xyz.x < interpQuadratic(vec2f(188.50,1275.50),vec2f(214.00,1345.00),vec2f(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2f(464.50,1336.50),vec2f(503.00,1349.00),vec2f(550.00,1349.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1296.50 && xyz.y < 1336.50 && (xyz.x < interpQuadratic(vec2f(188.50,1275.50),vec2f(214.00,1345.00),vec2f(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2f(398.00,1296.50),vec2f(426.00,1324.00),vec2f(464.50,1336.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 1275.50 && xyz.y < 1296.50 && (xyz.x < interpQuadratic(vec2f(188.50,1275.50),vec2f(214.00,1345.00),vec2f(260.50,1392.50),xyz.y) || xyz.x > interpQuadratic(vec2f(355.50,1224.00),vec2f(370.00,1269.00),vec2f(398.00,1296.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 1224.00 && xyz.y < 1275.50 && (xyz.x < interpQuadratic(vec2f(163.00,1117.00),vec2f(163.00,1206.00),vec2f(188.50,1275.50),xyz.y) || xyz.x > interpQuadratic(vec2f(355.50,1224.00),vec2f(370.00,1269.00),vec2f(398.00,1296.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 1117.00 && xyz.y < 1224.00 && (xyz.x < interpQuadratic(vec2f(163.00,1117.00),vec2f(163.00,1206.00),vec2f(188.50,1275.50),xyz.y) || xyz.x > interpQuadratic(vec2f(341.00,1112.00),vec2f(341.00,1179.00),vec2f(355.50,1224.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1112.00 && xyz.y < 1117.00 && (xyz.x < interpLine(vec2f(163.00,1017.00),vec2f(163.00,1117.00),xyz.y) || xyz.x > interpQuadratic(vec2f(341.00,1112.00),vec2f(341.00,1179.00),vec2f(355.50,1224.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1017.00 && xyz.y < 1112.00 && (xyz.x < interpLine(vec2f(163.00,1017.00),vec2f(163.00,1117.00),xyz.y) || xyz.x > interpLine(vec2f(341.00,1017.00),vec2f(341.00,1112.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 942.00 && xyz.y < 1017.00 && (xyz.x < interpLine(vec2f(0.00,942.00),vec2f(0.00,1017.00),xyz.y) || xyz.x > interpLine(vec2f(641.00,885.00),vec2f(641.00,1017.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 911.00 && xyz.y < 942.00 && (xyz.x < interpQuadratic(vec2f(13.50,911.00),vec2f(0.00,922.00),vec2f(0.00,942.00),xyz.y) || xyz.x > interpLine(vec2f(641.00,885.00),vec2f(641.00,1017.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 895.00 && xyz.y < 911.00 && (xyz.x < interpQuadratic(vec2f(49.00,895.00),vec2f(27.00,900.00),vec2f(13.50,911.00),xyz.y) || xyz.x > interpLine(vec2f(641.00,885.00),vec2f(641.00,1017.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 885.00 && xyz.y < 895.00 && (xyz.x < interpLine(vec2f(163.00,882.00),vec2f(49.00,895.00),xyz.y) || xyz.x > interpLine(vec2f(641.00,885.00),vec2f(641.00,1017.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 882.00 && xyz.y < 885.00 && (xyz.x < interpLine(vec2f(163.00,882.00),vec2f(49.00,895.00),xyz.y) || xyz.x > interpLine(vec2f(347.00,0.00),vec2f(347.00,885.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 882.00 && (xyz.x < interpLine(vec2f(163.00,0.00),vec2f(163.00,882.00),xyz.y) || xyz.x > interpLine(vec2f(347.00,0.00),vec2f(347.00,885.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_f(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(27.00,0.00,0.0)) || any(xyz_in > vec3f(676.00,1489.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(27.00,0.00,0.0);
  return glyph_f_1(xyz);
}

fn glyph_n_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(863.00,1054.00))) { return 0.0; }

  if (xyz.y >= 1037.00 && xyz.y <= 1054.00 && (xyz.x < interpQuadratic(vec2f(323.00,1008.00),vec2f(406.00,1054.00),vec2f(516.00,1054.00),xyz.y) || xyz.x > interpQuadratic(vec2f(666.00,1025.50),vec2f(601.00,1054.00),vec2f(516.00,1054.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1025.50 && xyz.y < 1037.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(157.00,999.00),vec2f(147.00,1037.00),vec2f(108.00,1037.00),xyz.y) && xyz.x < interpQuadratic(vec2f(323.00,1008.00),vec2f(406.00,1054.00),vec2f(516.00,1054.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(666.00,1025.50),vec2f(601.00,1054.00),vec2f(516.00,1054.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1008.00 && xyz.y < 1025.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(157.00,999.00),vec2f(147.00,1037.00),vec2f(108.00,1037.00),xyz.y) && xyz.x < interpQuadratic(vec2f(323.00,1008.00),vec2f(406.00,1054.00),vec2f(516.00,1054.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(774.50,945.50),vec2f(731.00,997.00),vec2f(666.00,1025.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 999.00 && xyz.y < 1008.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(157.00,999.00),vec2f(147.00,1037.00),vec2f(108.00,1037.00),xyz.y) && xyz.x < interpQuadratic(vec2f(172.00,887.00),vec2f(240.00,962.00),vec2f(323.00,1008.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(774.50,945.50),vec2f(731.00,997.00),vec2f(666.00,1025.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 945.50 && xyz.y < 999.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2f(172.00,887.00),vec2f(157.00,999.00),xyz.y) && xyz.x < interpQuadratic(vec2f(172.00,887.00),vec2f(240.00,962.00),vec2f(323.00,1008.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(774.50,945.50),vec2f(731.00,997.00),vec2f(666.00,1025.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 908.00 && xyz.y < 945.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2f(172.00,887.00),vec2f(157.00,999.00),xyz.y) && xyz.x < interpQuadratic(vec2f(172.00,887.00),vec2f(240.00,962.00),vec2f(323.00,1008.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(840.50,821.50),vec2f(818.00,894.00),vec2f(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 887.00 && xyz.y < 908.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2f(172.00,887.00),vec2f(157.00,999.00),xyz.y) && xyz.x < interpQuadratic(vec2f(172.00,887.00),vec2f(240.00,962.00),vec2f(323.00,1008.00),xyz.y)) || (xyz.x > interpQuadratic(vec2f(311.00,869.00),vec2f(381.00,908.00),vec2f(462.00,908.00),xyz.y) && xyz.x < interpQuadratic(vec2f(626.50,843.00),vec2f(573.00,908.00),vec2f(462.00,908.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(840.50,821.50),vec2f(818.00,894.00),vec2f(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 869.00 && xyz.y < 887.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(311.00,869.00),vec2f(381.00,908.00),vec2f(462.00,908.00),xyz.y) && xyz.x < interpQuadratic(vec2f(626.50,843.00),vec2f(573.00,908.00),vec2f(462.00,908.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(840.50,821.50),vec2f(818.00,894.00),vec2f(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 843.00 && xyz.y < 869.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(182.00,764.00),vec2f(241.00,830.00),vec2f(311.00,869.00),xyz.y) && xyz.x < interpQuadratic(vec2f(626.50,843.00),vec2f(573.00,908.00),vec2f(462.00,908.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(840.50,821.50),vec2f(818.00,894.00),vec2f(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 821.50 && xyz.y < 843.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(182.00,764.00),vec2f(241.00,830.00),vec2f(311.00,869.00),xyz.y) && xyz.x < interpQuadratic(vec2f(680.00,660.00),vec2f(680.00,778.00),vec2f(626.50,843.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(840.50,821.50),vec2f(818.00,894.00),vec2f(774.50,945.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 764.00 && xyz.y < 821.50 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(182.00,764.00),vec2f(241.00,830.00),vec2f(311.00,869.00),xyz.y) && xyz.x < interpQuadratic(vec2f(680.00,660.00),vec2f(680.00,778.00),vec2f(626.50,843.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(863.00,660.00),vec2f(863.00,749.00),vec2f(840.50,821.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 660.00 && xyz.y < 764.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2f(182.00,0.00),vec2f(182.00,764.00),xyz.y) && xyz.x < interpQuadratic(vec2f(680.00,660.00),vec2f(680.00,778.00),vec2f(626.50,843.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(863.00,660.00),vec2f(863.00,749.00),vec2f(840.50,821.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 660.00 && (xyz.x < interpLine(vec2f(0.00,0.00),vec2f(0.00,1037.00),xyz.y) || (xyz.x > interpLine(vec2f(182.00,0.00),vec2f(182.00,764.00),xyz.y) && xyz.x < interpLine(vec2f(680.00,0.00),vec2f(680.00,660.00),xyz.y)) || xyz.x > interpLine(vec2f(863.00,0.00),vec2f(863.00,660.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_n(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(150.00,0.00,0.0)) || any(xyz_in > vec3f(1013.00,1054.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(150.00,0.00,0.0);
  return glyph_n_1(xyz);
}

fn glyph_o_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(989.00,1068.00))) { return 0.0; }

  if (xyz.y >= 1030.00 && xyz.y <= 1068.00 && (xyz.x < interpQuadratic(vec2f(290.00,1030.00),vec2f(382.00,1068.00),vec2f(495.00,1068.00),xyz.y) || xyz.x > interpQuadratic(vec2f(700.50,1030.00),vec2f(609.00,1068.00),vec2f(495.00,1068.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 922.50 && xyz.y < 1030.00 && (xyz.x < interpQuadratic(vec2f(133.50,922.50),vec2f(198.00,992.00),vec2f(290.00,1030.00),xyz.y) || xyz.x > interpQuadratic(vec2f(856.00,922.50),vec2f(792.00,992.00),vec2f(700.50,1030.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 754.00 && xyz.y < 922.50 && (xyz.x < interpQuadratic(vec2f(34.50,754.00),vec2f(69.00,853.00),vec2f(133.50,922.50),xyz.y) || xyz.x > interpQuadratic(vec2f(954.50,754.00),vec2f(920.00,853.00),vec2f(856.00,922.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 533.00 && xyz.y < 754.00 && (xyz.x < interpQuadratic(vec2f(0.00,533.00),vec2f(0.00,655.00),vec2f(34.50,754.00),xyz.y) || xyz.x > interpQuadratic(vec2f(989.00,533.00),vec2f(989.00,655.00),vec2f(954.50,754.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 312.00 && xyz.y < 533.00 && (xyz.x < interpQuadratic(vec2f(34.50,312.00),vec2f(0.00,410.00),vec2f(0.00,533.00),xyz.y) || xyz.x > interpQuadratic(vec2f(954.50,312.00),vec2f(989.00,410.00),vec2f(989.00,533.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 144.00 && xyz.y < 312.00 && (xyz.x < interpQuadratic(vec2f(133.50,144.00),vec2f(69.00,214.00),vec2f(34.50,312.00),xyz.y) || xyz.x > interpQuadratic(vec2f(856.00,144.00),vec2f(920.00,214.00),vec2f(856.00,144.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 37.00 && xyz.y < 144.00 && (xyz.x < interpQuadratic(vec2f(289.50,37.00),vec2f(198.00,74.00),vec2f(133.50,144.00),xyz.y) || xyz.x > interpQuadratic(vec2f(700.50,37.00),vec2f(792.00,74.00),vec2f(856.00,144.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 37.00 && (xyz.x < interpQuadratic(vec2f(495.00,0.00),vec2f(381.00,0.00),vec2f(289.50,37.00),xyz.y) || xyz.x > interpQuadratic(vec2f(495.00,0.00),vec2f(609.00,0.00),vec2f(700.50,37.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_o_2(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(188.00,142.00)) || any(xyz.xy > vec2f(800.00,924.00))) { return 0.0; }

  if (xyz.y >= 897.50 && xyz.y <= 924.00 && (xyz.x < interpQuadratic(vec2f(360.00,897.50),vec2f(417.00,924.00),vec2f(495.00,924.00),xyz.y) || xyz.x > interpQuadratic(vec2f(724.50,820.50),vec2f(649.00,924.00),vec2f(495.00,924.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 821.00 && xyz.y < 897.50 && (xyz.x < interpQuadratic(vec2f(264.50,821.00),vec2f(303.00,871.00),vec2f(360.00,897.50),xyz.y) || xyz.x > interpQuadratic(vec2f(724.50,820.50),vec2f(649.00,924.00),vec2f(495.00,924.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 820.50 && xyz.y < 821.00 && (xyz.x < interpQuadratic(vec2f(207.00,697.50),vec2f(226.00,771.00),vec2f(264.50,821.00),xyz.y) || xyz.x > interpQuadratic(vec2f(724.50,820.50),vec2f(649.00,924.00),vec2f(495.00,924.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 697.50 && xyz.y < 820.50 && (xyz.x < interpQuadratic(vec2f(207.00,697.50),vec2f(226.00,771.00),vec2f(264.50,821.00),xyz.y) || xyz.x > interpQuadratic(vec2f(800.00,532.00),vec2f(800.00,717.00),vec2f(724.50,820.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 532.00 && xyz.y < 697.50 && (xyz.x < interpQuadratic(vec2f(188.00,532.00),vec2f(188.00,624.00),vec2f(207.00,697.50),xyz.y) || xyz.x > interpQuadratic(vec2f(800.00,532.00),vec2f(800.00,717.00),vec2f(724.50,820.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 367.50 && xyz.y < 532.00 && (xyz.x < interpQuadratic(vec2f(207.00,367.50),vec2f(188.00,440.00),vec2f(188.00,532.00),xyz.y) || xyz.x > interpQuadratic(vec2f(724.50,245.00),vec2f(800.00,348.00),vec2f(800.00,532.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 245.00 && xyz.y < 367.50 && (xyz.x < interpQuadratic(vec2f(264.50,245.00),vec2f(226.00,295.00),vec2f(207.00,367.50),xyz.y) || xyz.x > interpQuadratic(vec2f(724.50,245.00),vec2f(800.00,348.00),vec2f(800.00,532.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 168.50 && xyz.y < 245.00 && (xyz.x < interpQuadratic(vec2f(360.50,168.50),vec2f(303.00,195.00),vec2f(264.50,245.00),xyz.y) || xyz.x > interpQuadratic(vec2f(495.00,142.00),vec2f(649.00,142.00),vec2f(724.50,245.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 142.00 && xyz.y < 168.50 && (xyz.x < interpQuadratic(vec2f(495.00,142.00),vec2f(418.00,142.00),vec2f(360.50,168.50),xyz.y) || xyz.x > interpQuadratic(vec2f(495.00,142.00),vec2f(649.00,142.00),vec2f(724.50,245.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_o(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(74.00,-14.00,0.0)) || any(xyz_in > vec3f(1063.00,1054.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(74.00,-14.00,0.0);
  let result = glyph_o_1(xyz);
  return result - glyph_o_2(xyz);
}

fn glyph_s_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(739.00,1070.00))) { return 0.0; }

  if (xyz.y >= 1045.50 && xyz.y <= 1070.00 && (xyz.x < interpQuadratic(vec2f(240.00,1045.50),vec2f(309.00,1070.00),vec2f(397.00,1070.00),xyz.y) || xyz.x > interpQuadratic(vec2f(580.50,1037.50),vec2f(499.00,1070.00),vec2f(397.00,1070.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1037.50 && xyz.y < 1045.50 && (xyz.x < interpQuadratic(vec2f(124.00,979.50),vec2f(171.00,1021.00),vec2f(240.00,1045.50),xyz.y) || xyz.x > interpQuadratic(vec2f(580.50,1037.50),vec2f(499.00,1070.00),vec2f(397.00,1070.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 979.50 && xyz.y < 1037.50 && (xyz.x < interpQuadratic(vec2f(124.00,979.50),vec2f(171.00,1021.00),vec2f(240.00,1045.50),xyz.y) || xyz.x > interpQuadratic(vec2f(721.00,949.00),vec2f(662.00,1005.00),vec2f(580.50,1037.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 949.00 && xyz.y < 979.50 && (xyz.x < interpQuadratic(vec2f(52.50,883.00),vec2f(77.00,938.00),vec2f(124.00,979.50),xyz.y) || xyz.x > interpQuadratic(vec2f(721.00,949.00),vec2f(662.00,1005.00),vec2f(580.50,1037.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 934.00 && xyz.y < 949.00 && (xyz.x < interpQuadratic(vec2f(52.50,883.00),vec2f(77.00,938.00),vec2f(124.00,979.50),xyz.y) || xyz.x > interpLine(vec2f(680.00,882.00),vec2f(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 922.00 && xyz.y < 934.00 && (xyz.x < interpQuadratic(vec2f(52.50,883.00),vec2f(77.00,938.00),vec2f(124.00,979.50),xyz.y) || (xyz.x > interpQuadratic(vec2f(318.00,922.00),vec2f(355.00,934.00),vec2f(401.00,934.00),xyz.y) && xyz.x < interpQuadratic(vec2f(493.00,922.00),vec2f(454.00,934.00),vec2f(401.00,934.00),xyz.y)) || xyz.x > interpLine(vec2f(680.00,882.00),vec2f(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 896.00 && xyz.y < 922.00 && (xyz.x < interpQuadratic(vec2f(52.50,883.00),vec2f(77.00,938.00),vec2f(124.00,979.50),xyz.y) || (xyz.x > interpQuadratic(vec2f(255.00,889.50),vec2f(281.00,910.00),vec2f(318.00,922.00),xyz.y) && xyz.x < interpQuadratic(vec2f(560.00,896.00),vec2f(532.00,910.00),vec2f(493.00,922.00),xyz.y)) || xyz.x > interpLine(vec2f(680.00,882.00),vec2f(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 889.50 && xyz.y < 896.00 && (xyz.x < interpQuadratic(vec2f(52.50,883.00),vec2f(77.00,938.00),vec2f(124.00,979.50),xyz.y) || (xyz.x > interpQuadratic(vec2f(255.00,889.50),vec2f(281.00,910.00),vec2f(318.00,922.00),xyz.y) && xyz.x < interpQuadratic(vec2f(608.00,871.00),vec2f(588.00,882.00),vec2f(560.00,896.00),xyz.y)) || xyz.x > interpLine(vec2f(680.00,882.00),vec2f(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 883.00 && xyz.y < 889.50 && (xyz.x < interpQuadratic(vec2f(52.50,883.00),vec2f(77.00,938.00),vec2f(124.00,979.50),xyz.y) || (xyz.x > interpQuadratic(vec2f(215.00,842.00),vec2f(229.00,869.00),vec2f(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2f(608.00,871.00),vec2f(588.00,882.00),vec2f(560.00,896.00),xyz.y)) || xyz.x > interpLine(vec2f(680.00,882.00),vec2f(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 882.00 && xyz.y < 883.00 && (xyz.x < interpQuadratic(vec2f(28.00,769.00),vec2f(28.00,828.00),vec2f(52.50,883.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(215.00,842.00),vec2f(229.00,869.00),vec2f(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2f(608.00,871.00),vec2f(588.00,882.00),vec2f(560.00,896.00),xyz.y)) || xyz.x > interpLine(vec2f(680.00,882.00),vec2f(721.00,949.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 871.00 && xyz.y < 882.00 && (xyz.x < interpQuadratic(vec2f(28.00,769.00),vec2f(28.00,828.00),vec2f(52.50,883.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(215.00,842.00),vec2f(229.00,869.00),vec2f(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2f(608.00,871.00),vec2f(588.00,882.00),vec2f(560.00,896.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(643.00,860.00),vec2f(668.00,859.00),vec2f(680.00,882.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 860.00 && xyz.y < 871.00 && (xyz.x < interpQuadratic(vec2f(28.00,769.00),vec2f(28.00,828.00),vec2f(52.50,883.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(215.00,842.00),vec2f(229.00,869.00),vec2f(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2f(643.00,860.00),vec2f(628.00,860.00),vec2f(608.00,871.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(643.00,860.00),vec2f(668.00,859.00),vec2f(680.00,882.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 859.00 && xyz.y < 860.00 && (xyz.x < interpQuadratic(vec2f(28.00,769.00),vec2f(28.00,828.00),vec2f(52.50,883.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(215.00,842.00),vec2f(229.00,869.00),vec2f(255.00,889.50),xyz.y) && xyz.x < interpQuadratic(vec2f(643.00,860.00),vec2f(668.00,859.00),vec2f(680.00,882.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(643.00,860.00),vec2f(668.00,859.00),vec2f(680.00,882.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 842.00 && xyz.y < 859.00 && (xyz.x < interpQuadratic(vec2f(28.00,769.00),vec2f(28.00,828.00),vec2f(52.50,883.00),xyz.y) || xyz.x > interpQuadratic(vec2f(215.00,842.00),vec2f(229.00,869.00),vec2f(255.00,889.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 783.00 && xyz.y < 842.00 && (xyz.x < interpQuadratic(vec2f(28.00,769.00),vec2f(28.00,828.00),vec2f(52.50,883.00),xyz.y) || xyz.x > interpQuadratic(vec2f(201.00,783.00),vec2f(201.00,815.00),vec2f(215.00,842.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 769.00 && xyz.y < 783.00 && (xyz.x < interpQuadratic(vec2f(28.00,769.00),vec2f(28.00,828.00),vec2f(52.50,883.00),xyz.y) || xyz.x > interpQuadratic(vec2f(224.00,716.50),vec2f(201.00,743.00),vec2f(201.00,783.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 716.50 && xyz.y < 769.00 && (xyz.x < interpQuadratic(vec2f(51.00,655.50),vec2f(28.00,702.00),vec2f(28.00,769.00),xyz.y) || xyz.x > interpQuadratic(vec2f(224.00,716.50),vec2f(201.00,743.00),vec2f(201.00,783.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 670.50 && xyz.y < 716.50 && (xyz.x < interpQuadratic(vec2f(51.00,655.50),vec2f(28.00,702.00),vec2f(28.00,769.00),xyz.y) || xyz.x > interpQuadratic(vec2f(285.00,670.50),vec2f(247.00,690.00),vec2f(224.00,716.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 655.50 && xyz.y < 670.50 && (xyz.x < interpQuadratic(vec2f(51.00,655.50),vec2f(28.00,702.00),vec2f(28.00,769.00),xyz.y) || xyz.x > interpQuadratic(vec2f(371.00,636.00),vec2f(323.00,651.00),vec2f(285.00,670.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 636.00 && xyz.y < 655.50 && (xyz.x < interpQuadratic(vec2f(112.00,577.00),vec2f(74.00,609.00),vec2f(51.00,655.50),xyz.y) || xyz.x > interpQuadratic(vec2f(371.00,636.00),vec2f(323.00,651.00),vec2f(285.00,670.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 604.00 && xyz.y < 636.00 && (xyz.x < interpQuadratic(vec2f(112.00,577.00),vec2f(74.00,609.00),vec2f(51.00,655.50),xyz.y) || xyz.x > interpQuadratic(vec2f(470.00,604.00),vec2f(419.00,621.00),vec2f(371.00,636.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 577.00 && xyz.y < 604.00 && (xyz.x < interpQuadratic(vec2f(112.00,577.00),vec2f(74.00,609.00),vec2f(51.00,655.50),xyz.y) || xyz.x > interpQuadratic(vec2f(569.00,566.50),vec2f(521.00,587.00),vec2f(470.00,604.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 566.50 && xyz.y < 577.00 && (xyz.x < interpQuadratic(vec2f(198.50,524.00),vec2f(150.00,545.00),vec2f(112.00,577.00),xyz.y) || xyz.x > interpQuadratic(vec2f(569.00,566.50),vec2f(521.00,587.00),vec2f(470.00,604.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 524.00 && xyz.y < 566.50 && (xyz.x < interpQuadratic(vec2f(198.50,524.00),vec2f(150.00,545.00),vec2f(112.00,577.00),xyz.y) || xyz.x > interpQuadratic(vec2f(655.00,515.50),vec2f(617.00,546.00),vec2f(569.00,566.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 515.50 && xyz.y < 524.00 && (xyz.x < interpQuadratic(vec2f(298.00,487.00),vec2f(247.00,503.00),vec2f(198.50,524.00),xyz.y) || xyz.x > interpQuadratic(vec2f(655.00,515.50),vec2f(617.00,546.00),vec2f(569.00,566.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 487.00 && xyz.y < 515.50 && (xyz.x < interpQuadratic(vec2f(298.00,487.00),vec2f(247.00,503.00),vec2f(198.50,524.00),xyz.y) || xyz.x > interpQuadratic(vec2f(716.00,440.50),vec2f(693.00,485.00),vec2f(655.00,515.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 456.00 && xyz.y < 487.00 && (xyz.x < interpQuadratic(vec2f(397.50,456.00),vec2f(349.00,471.00),vec2f(298.00,487.00),xyz.y) || xyz.x > interpQuadratic(vec2f(716.00,440.50),vec2f(693.00,485.00),vec2f(655.00,515.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 440.50 && xyz.y < 456.00 && (xyz.x < interpQuadratic(vec2f(484.00,421.00),vec2f(446.00,441.00),vec2f(397.50,456.00),xyz.y) || xyz.x > interpQuadratic(vec2f(716.00,440.50),vec2f(693.00,485.00),vec2f(655.00,515.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 421.00 && xyz.y < 440.50 && (xyz.x < interpQuadratic(vec2f(484.00,421.00),vec2f(446.00,441.00),vec2f(397.50,456.00),xyz.y) || xyz.x > interpQuadratic(vec2f(739.00,333.00),vec2f(739.00,396.00),vec2f(716.00,440.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 373.00 && xyz.y < 421.00 && (xyz.x < interpQuadratic(vec2f(545.00,373.00),vec2f(522.00,401.00),vec2f(484.00,421.00),xyz.y) || xyz.x > interpQuadratic(vec2f(739.00,333.00),vec2f(739.00,396.00),vec2f(716.00,440.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 333.00 && xyz.y < 373.00 && (xyz.x < interpQuadratic(vec2f(568.00,302.00),vec2f(568.00,345.00),vec2f(545.00,373.00),xyz.y) || xyz.x > interpQuadratic(vec2f(739.00,333.00),vec2f(739.00,396.00),vec2f(716.00,440.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 302.00 && xyz.y < 333.00 && (xyz.x < interpQuadratic(vec2f(568.00,302.00),vec2f(568.00,345.00),vec2f(545.00,373.00),xyz.y) || xyz.x > interpQuadratic(vec2f(713.50,200.50),vec2f(739.00,261.00),vec2f(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 236.00 && xyz.y < 302.00 && (xyz.x < interpQuadratic(vec2f(555.00,236.00),vec2f(568.00,267.00),vec2f(568.00,302.00),xyz.y) || xyz.x > interpQuadratic(vec2f(713.50,200.50),vec2f(739.00,261.00),vec2f(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 223.00 && xyz.y < 236.00 && (xyz.x < interpQuadratic(vec2f(515.50,181.50),vec2f(542.00,205.00),vec2f(555.00,236.00),xyz.y) || xyz.x > interpQuadratic(vec2f(713.50,200.50),vec2f(739.00,261.00),vec2f(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 215.50 && xyz.y < 223.00 && (xyz.x < interpQuadratic(vec2f(62.50,215.50),vec2f(74.00,223.00),vec2f(93.00,223.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(131.50,208.50),vec2f(111.00,223.00),vec2f(93.00,223.00),xyz.y) && xyz.x < interpQuadratic(vec2f(515.50,181.50),vec2f(542.00,205.00),vec2f(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(713.50,200.50),vec2f(739.00,261.00),vec2f(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 208.50 && xyz.y < 215.50 && (xyz.x < interpQuadratic(vec2f(43.00,195.00),vec2f(51.00,208.00),vec2f(62.50,215.50),xyz.y) || (xyz.x > interpQuadratic(vec2f(131.50,208.50),vec2f(111.00,223.00),vec2f(93.00,223.00),xyz.y) && xyz.x < interpQuadratic(vec2f(515.50,181.50),vec2f(542.00,205.00),vec2f(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(713.50,200.50),vec2f(739.00,261.00),vec2f(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 200.50 && xyz.y < 208.50 && (xyz.x < interpQuadratic(vec2f(43.00,195.00),vec2f(51.00,208.00),vec2f(62.50,215.50),xyz.y) || (xyz.x > interpQuadratic(vec2f(181.00,176.50),vec2f(152.00,194.00),vec2f(131.50,208.50),xyz.y) && xyz.x < interpQuadratic(vec2f(515.50,181.50),vec2f(542.00,205.00),vec2f(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(713.50,200.50),vec2f(739.00,261.00),vec2f(739.00,333.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 195.00 && xyz.y < 200.50 && (xyz.x < interpQuadratic(vec2f(43.00,195.00),vec2f(51.00,208.00),vec2f(62.50,215.50),xyz.y) || (xyz.x > interpQuadratic(vec2f(181.00,176.50),vec2f(152.00,194.00),vec2f(131.50,208.50),xyz.y) && xyz.x < interpQuadratic(vec2f(515.50,181.50),vec2f(542.00,205.00),vec2f(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(637.50,95.50),vec2f(688.00,140.00),vec2f(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 181.50 && xyz.y < 195.00 && (xyz.x < interpLine(vec2f(0.00,126.00),vec2f(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(181.00,176.50),vec2f(152.00,194.00),vec2f(131.50,208.50),xyz.y) && xyz.x < interpQuadratic(vec2f(515.50,181.50),vec2f(542.00,205.00),vec2f(555.00,236.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(637.50,95.50),vec2f(688.00,140.00),vec2f(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 176.50 && xyz.y < 181.50 && (xyz.x < interpLine(vec2f(0.00,126.00),vec2f(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(181.00,176.50),vec2f(152.00,194.00),vec2f(131.50,208.50),xyz.y) && xyz.x < interpQuadratic(vec2f(449.00,144.50),vec2f(489.00,158.00),vec2f(515.50,181.50),xyz.y)) || xyz.x > interpQuadratic(vec2f(637.50,95.50),vec2f(688.00,140.00),vec2f(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 145.00 && xyz.y < 176.50 && (xyz.x < interpLine(vec2f(0.00,126.00),vec2f(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(251.50,145.00),vec2f(210.00,159.00),vec2f(181.00,176.50),xyz.y) && xyz.x < interpQuadratic(vec2f(449.00,144.50),vec2f(489.00,158.00),vec2f(515.50,181.50),xyz.y)) || xyz.x > interpQuadratic(vec2f(637.50,95.50),vec2f(688.00,140.00),vec2f(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 144.50 && xyz.y < 145.00 && (xyz.x < interpLine(vec2f(0.00,126.00),vec2f(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(356.00,131.00),vec2f(293.00,131.00),vec2f(251.50,145.00),xyz.y) && xyz.x < interpQuadratic(vec2f(449.00,144.50),vec2f(489.00,158.00),vec2f(515.50,181.50),xyz.y)) || xyz.x > interpQuadratic(vec2f(637.50,95.50),vec2f(688.00,140.00),vec2f(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 131.00 && xyz.y < 144.50 && (xyz.x < interpLine(vec2f(0.00,126.00),vec2f(43.00,195.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(356.00,131.00),vec2f(293.00,131.00),vec2f(251.50,145.00),xyz.y) && xyz.x < interpQuadratic(vec2f(356.00,131.00),vec2f(409.00,131.00),vec2f(449.00,144.50),xyz.y)) || xyz.x > interpQuadratic(vec2f(637.50,95.50),vec2f(688.00,140.00),vec2f(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 126.00 && xyz.y < 131.00 && (xyz.x < interpLine(vec2f(0.00,126.00),vec2f(43.00,195.00),xyz.y) || xyz.x > interpQuadratic(vec2f(637.50,95.50),vec2f(688.00,140.00),vec2f(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 95.50 && xyz.y < 126.00 && (xyz.x < interpQuadratic(vec2f(149.50,35.50),vec2f(61.00,71.00),vec2f(0.00,126.00),xyz.y) || xyz.x > interpQuadratic(vec2f(637.50,95.50),vec2f(688.00,140.00),vec2f(713.50,200.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 35.50 && xyz.y < 95.50 && (xyz.x < interpQuadratic(vec2f(149.50,35.50),vec2f(61.00,71.00),vec2f(0.00,126.00),xyz.y) || xyz.x > interpQuadratic(vec2f(514.50,25.50),vec2f(587.00,51.00),vec2f(637.50,95.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 25.50 && xyz.y < 35.50 && (xyz.x < interpQuadratic(vec2f(347.00,0.00),vec2f(238.00,0.00),vec2f(149.50,35.50),xyz.y) || xyz.x > interpQuadratic(vec2f(514.50,25.50),vec2f(587.00,51.00),vec2f(637.50,95.50),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 25.50 && (xyz.x < interpQuadratic(vec2f(347.00,0.00),vec2f(238.00,0.00),vec2f(149.50,35.50),xyz.y) || xyz.x > interpQuadratic(vec2f(347.00,0.00),vec2f(442.00,0.00),vec2f(514.50,25.50),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_s(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(63.00,-16.00,0.0)) || any(xyz_in > vec3f(802.00,1054.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(63.00,-16.00,0.0);
  return glyph_s_1(xyz);
}

fn glyph_t_1(xyz: vec3f) -> f32 {
  if (any(xyz.xy < vec2f(0.00,0.00)) || any(xyz.xy > vec2f(670.00,1390.00))) { return 0.0; }

  if (xyz.y >= 1380.00 && xyz.y <= 1390.00 && (xyz.x < interpQuadratic(vec2f(225.50,1380.00),vec2f(237.00,1390.00),vec2f(254.00,1390.00),xyz.y) || xyz.x > interpLine(vec2f(346.00,1033.00),vec2f(346.00,1390.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1355.00 && xyz.y < 1380.00 && (xyz.x < interpQuadratic(vec2f(212.00,1355.00),vec2f(214.00,1370.00),vec2f(225.50,1380.00),xyz.y) || xyz.x > interpLine(vec2f(346.00,1033.00),vec2f(346.00,1390.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1035.00 && xyz.y < 1355.00 && (xyz.x < interpLine(vec2f(170.00,1035.00),vec2f(212.00,1355.00),xyz.y) || xyz.x > interpLine(vec2f(346.00,1033.00),vec2f(346.00,1390.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1033.00 && xyz.y < 1035.00 && (xyz.x < interpLine(vec2f(0.00,1013.00),vec2f(170.00,1035.00),xyz.y) || xyz.x > interpLine(vec2f(346.00,1033.00),vec2f(346.00,1390.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 1013.00 && xyz.y < 1033.00 && (xyz.x < interpLine(vec2f(0.00,1013.00),vec2f(170.00,1035.00),xyz.y) || xyz.x > interpLine(vec2f(643.00,901.00),vec2f(643.00,1033.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 941.00 && xyz.y < 1013.00 && (xyz.x < interpLine(vec2f(0.00,941.00),vec2f(0.00,1013.00),xyz.y) || xyz.x > interpLine(vec2f(643.00,901.00),vec2f(643.00,1033.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 911.00 && xyz.y < 941.00 && (xyz.x < interpQuadratic(vec2f(11.50,911.00),vec2f(0.00,921.00),vec2f(0.00,941.00),xyz.y) || xyz.x > interpLine(vec2f(643.00,901.00),vec2f(643.00,1033.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 901.00 && xyz.y < 911.00 && (xyz.x < interpQuadratic(vec2f(39.00,901.00),vec2f(23.00,901.00),vec2f(11.50,911.00),xyz.y) || xyz.x > interpLine(vec2f(643.00,901.00),vec2f(643.00,1033.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 278.00 && xyz.y < 901.00 && (xyz.x < interpLine(vec2f(164.00,266.00),vec2f(164.00,901.00),xyz.y) || xyz.x > interpLine(vec2f(346.00,278.00),vec2f(346.00,901.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 266.00 && xyz.y < 278.00 && (xyz.x < interpLine(vec2f(164.00,266.00),vec2f(164.00,901.00),xyz.y) || xyz.x > interpQuadratic(vec2f(378.00,180.50),vec2f(346.00,212.00),vec2f(346.00,278.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 198.00 && xyz.y < 266.00 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || xyz.x > interpQuadratic(vec2f(378.00,180.50),vec2f(346.00,212.00),vec2f(346.00,278.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 190.50 && xyz.y < 198.00 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(378.00,180.50),vec2f(346.00,212.00),vec2f(346.00,278.00),xyz.y) && xyz.x < interpQuadratic(vec2f(572.00,190.50),vec2f(583.00,198.00),vec2f(591.00,198.00),xyz.y)) || xyz.x > interpQuadratic(vec2f(617.00,181.00),vec2f(605.00,198.00),vec2f(591.00,198.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 181.00 && xyz.y < 190.50 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(378.00,180.50),vec2f(346.00,212.00),vec2f(346.00,278.00),xyz.y) && xyz.x < interpQuadratic(vec2f(545.50,173.50),vec2f(561.00,183.00),vec2f(572.00,190.50),xyz.y)) || xyz.x > interpQuadratic(vec2f(617.00,181.00),vec2f(605.00,198.00),vec2f(591.00,198.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 180.50 && xyz.y < 181.00 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(378.00,180.50),vec2f(346.00,212.00),vec2f(346.00,278.00),xyz.y) && xyz.x < interpQuadratic(vec2f(545.50,173.50),vec2f(561.00,183.00),vec2f(572.00,190.50),xyz.y)) || xyz.x > interpLine(vec2f(670.00,94.00),vec2f(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 173.50 && xyz.y < 180.50 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(460.00,149.00),vec2f(410.00,149.00),vec2f(378.00,180.50),xyz.y) && xyz.x < interpQuadratic(vec2f(545.50,173.50),vec2f(561.00,183.00),vec2f(572.00,190.50),xyz.y)) || xyz.x > interpLine(vec2f(670.00,94.00),vec2f(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 156.50 && xyz.y < 173.50 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(460.00,149.00),vec2f(410.00,149.00),vec2f(378.00,180.50),xyz.y) && xyz.x < interpQuadratic(vec2f(509.50,156.50),vec2f(530.00,164.00),vec2f(545.50,173.50),xyz.y)) || xyz.x > interpLine(vec2f(670.00,94.00),vec2f(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 149.00 && xyz.y < 156.50 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || (xyz.x > interpQuadratic(vec2f(460.00,149.00),vec2f(410.00,149.00),vec2f(378.00,180.50),xyz.y) && xyz.x < interpQuadratic(vec2f(460.00,149.00),vec2f(489.00,149.00),vec2f(509.50,156.50),xyz.y)) || xyz.x > interpLine(vec2f(670.00,94.00),vec2f(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 94.00 && xyz.y < 149.00 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || xyz.x > interpLine(vec2f(670.00,94.00),vec2f(617.00,181.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 68.50 && xyz.y < 94.00 && (xyz.x < interpQuadratic(vec2f(230.00,68.50),vec2f(164.00,137.00),vec2f(164.00,266.00),xyz.y) || xyz.x > interpQuadratic(vec2f(556.50,25.00),vec2f(623.00,50.00),vec2f(670.00,94.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 25.00 && xyz.y < 68.50 && (xyz.x < interpQuadratic(vec2f(419.00,0.00),vec2f(296.00,0.00),vec2f(230.00,68.50),xyz.y) || xyz.x > interpQuadratic(vec2f(556.50,25.00),vec2f(623.00,50.00),vec2f(670.00,94.00),xyz.y))) { return 0.0; }
  if (xyz.y >= 0.00 && xyz.y < 25.00 && (xyz.x < interpQuadratic(vec2f(419.00,0.00),vec2f(296.00,0.00),vec2f(230.00,68.50),xyz.y) || xyz.x > interpQuadratic(vec2f(419.00,0.00),vec2f(490.00,0.00),vec2f(556.50,25.00),xyz.y))) { return 0.0; }
  return 1.0;
}

fn glyph_t(height: f32, xyz_in: vec3f) -> f32 {
  if (any(xyz_in < vec3f(45.00,-16.00,0.0)) || any(xyz_in > vec3f(715.00,1374.00,height))) { return 0.0; }
  let xyz = xyz_in - vec3f(45.00,-16.00,0.0);
  return glyph_t_1(xyz);
}

fn textMessage(emSize: f32, height: f32, xyz_orig: vec3f) -> f32 {
  let xyz = xyz_orig * vec3f(1649.00/emSize, 1649.00/emSize, 1.0);
  var result = 0.0;
  result += glyph_I(height, xyz-vec3f(0.00,0.0,0.0));
  result += glyph_R(height, xyz-vec3f(628.00,0.0,0.0));
  result += glyph_M(height, xyz-vec3f(2146.00,0.0,0.0));
  result += glyph_F(height, xyz-vec3f(4030.00,0.0,0.0));
  result += glyph_f(height, xyz-vec3f(5584.00,0.0,0.0));
  result += glyph_o(height, xyz-vec3f(6274.00,0.0,0.0));
  result += glyph_n(height, xyz-vec3f(7412.00,0.0,0.0));
  result += glyph_t(height, xyz-vec3f(8550.00,0.0,0.0));
  result += glyph_s(height, xyz-vec3f(9313.00,0.0,0.0));
  return result;
}

fn mainModel4(xyz_in: vec3f) -> vec4f {
  var xyz = xyz_in;
  xyz += vec3f(3.0, 0.0, 1.5);
  var materials = vec4f(0.0);
  materials[0] = textMessage(1.0, 0.5, xyz);
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

// Translated from text-1.irmf. The generated glyph outlines are point-in-
// polygon tests; this preserves their expressions.
const interpLine = (A, B, y) => {
  const p = y.sub(A.y).div(B.y.sub(A.y));
  return p.mul(B.x.sub(A.x)).add(A.x);
};

const interpQuadratic = (p0, p1, p2, y) => {
  const a = p2.y.add(p0.y).sub(p1.y.mul(2.0));
  const b = p1.y.sub(p0.y).mul(2.0);
  const c = p0.y.sub(y);
  const disc = b.mul(b).sub(a.mul(c).mul(4.0));
  const bad = disc.lessThan(0.0);
  const det = sqrt(disc.max(0.0));
  const t1 = det.sub(b).div(a.mul(2.0));
  const t2 = det.negate().sub(b).div(a.mul(2.0));
  const t2in = t2.greaterThanEqual(0.0).and(t2.lessThanEqual(1.0));
  const t = select(t2in, t2, t1);
  const omt = float(1.0).sub(t);
  const x = omt.mul(omt).mul(p0.x)
    .add(omt.mul(t).mul(2.0).mul(p1.x))
    .add(t.mul(t).mul(p2.x));
  return select(bad, float(0.0), x);
};


const glyph_F_1 = (p) => {
  const blocked = (((((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(904, 1467))))).or(((((p).y).greaterThanEqual(1306)).and(((p).y).lessThanEqual(1467))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpLine(vec2(904, 1306), vec2(904, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(793)).and(((p).y).lessThan(1306))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpLine(vec2(200, 793), vec2(200, 1306), (p).y)))))).or(((((p).y).greaterThanEqual(631)).and(((p).y).lessThan(793))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpLine(vec2(802, 631), vec2(802, 793), (p).y)))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThan(631))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpLine(vec2(200, 0), vec2(200, 631), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_F = (height, p) => {
  const blocked = (any((p).lessThan(vec3(178, 0, 0)))).or(any((p).greaterThan(vec3(1082, 1467, height))));
  const q = p.sub(vec3(178.00,0.00,0.0));
  return select(blocked, float(0.0), glyph_F_1(q));
};

const glyph_I_1 = (p) => {
  const blocked = ((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(199, 1467))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThanEqual(1467))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpLine(vec2(199, 0), vec2(199, 1467), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_I = (height, p) => {
  const blocked = (any((p).lessThan(vec3(215, 0, 0)))).or(any((p).greaterThan(vec3(414, 1467, height))));
  const q = p.sub(vec3(215.00,0.00,0.0));
  return select(blocked, float(0.0), glyph_I_1(q));
};

const glyph_M_1 = (p) => {
  const blocked = (((((((((((((((((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(1527, 1467))))).or(((((p).y).greaterThanEqual(1462.5)).and(((p).y).lessThanEqual(1467))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(188, 1462), vec2(174, 1467), vec2(147, 1467), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(1339.5, 1462.5), vec2(1354, 1467), vec2(1381, 1467), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(1462)).and(((p).y).lessThan(1462.5))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(188, 1462), vec2(174, 1467), vec2(147, 1467), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(1312, 1434), vec2(1325, 1458), vec2(1339.5, 1462.5), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(1434)).and(((p).y).lessThan(1462))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(215, 1434), vec2(202, 1457), vec2(188, 1462), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(1312, 1434), vec2(1325, 1458), vec2(1339.5, 1462.5), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(1178)).and(((p).y).lessThan(1434))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(722, 530), vec2(215, 1434), (p).y))).and(((p).x).lessThan(interpLine(vec2(815, 531), vec2(1312, 1434), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(1175)).and(((p).y).lessThan(1178))).and((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(172.5, 1125.5), vec2(171, 1151), vec2(168, 1178), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(722, 530), vec2(215, 1434), (p).y))).and(((p).x).lessThan(interpLine(vec2(815, 531), vec2(1312, 1434), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(1125.5)).and(((p).y).lessThan(1175))).and(((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(172.5, 1125.5), vec2(171, 1151), vec2(168, 1178), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(722, 530), vec2(215, 1434), (p).y))).and(((p).x).lessThan(interpLine(vec2(815, 531), vec2(1312, 1434), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(1354, 1124.5), vec2(1355, 1149), vec2(1357, 1175), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(1124.5)).and(((p).y).lessThan(1125.5))).and(((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(174, 1078), vec2(174, 1100), vec2(172.5, 1125.5), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(722, 530), vec2(215, 1434), (p).y))).and(((p).x).lessThan(interpLine(vec2(815, 531), vec2(1312, 1434), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(1354, 1124.5), vec2(1355, 1149), vec2(1357, 1175), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(1078)).and(((p).y).lessThan(1124.5))).and(((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(174, 1078), vec2(174, 1100), vec2(172.5, 1125.5), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(722, 530), vec2(215, 1434), (p).y))).and(((p).x).lessThan(interpLine(vec2(815, 531), vec2(1312, 1434), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(1353, 1078), vec2(1353, 1100), vec2(1354, 1124.5), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(531)).and(((p).y).lessThan(1078))).and(((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(174, 0), vec2(174, 1078), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(722, 530), vec2(215, 1434), (p).y))).and(((p).x).lessThan(interpLine(vec2(815, 531), vec2(1312, 1434), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpLine(vec2(1353, 0), vec2(1353, 1078), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(530)).and(((p).y).lessThan(531))).and(((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(174, 0), vec2(174, 1078), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(722, 530), vec2(215, 1434), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(789.5, 479), vec2(801, 505), vec2(815, 531), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpLine(vec2(1353, 0), vec2(1353, 1078), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(479)).and(((p).y).lessThan(530))).and(((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(174, 0), vec2(174, 1078), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpQuadratic(vec2(747, 478.5), vec2(736, 505), vec2(722, 530), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(789.5, 479), vec2(801, 505), vec2(815, 531), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpLine(vec2(1353, 0), vec2(1353, 1078), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(478.5)).and(((p).y).lessThan(479))).and(((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(174, 0), vec2(174, 1078), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpQuadratic(vec2(747, 478.5), vec2(736, 505), vec2(722, 530), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(768, 424), vec2(778, 453), vec2(789.5, 479), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpLine(vec2(1353, 0), vec2(1353, 1078), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(424)).and(((p).y).lessThan(478.5))).and(((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(174, 0), vec2(174, 1078), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpQuadratic(vec2(768, 424), vec2(758, 452), vec2(747, 478.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(768, 424), vec2(778, 453), vec2(789.5, 479), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpLine(vec2(1353, 0), vec2(1353, 1078), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(257)).and(((p).y).lessThan(424))).and((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(174, 0), vec2(174, 1078), (p).y))).and(((p).x).lessThan(interpLine(vec2(682, 257), vec2(168, 1178), (p).y))))).or((((p).x).greaterThan(interpLine(vec2(854, 257), vec2(1357, 1175), (p).y))).and(((p).x).lessThan(interpLine(vec2(1353, 0), vec2(1353, 1078), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(211)).and(((p).y).lessThan(257))).and((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(174, 0), vec2(174, 1078), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(754, 211), vec2(708, 211), vec2(682, 257), (p).y))))).or((((p).x).greaterThan(interpQuadratic(vec2(783, 211), vec2(828, 211), vec2(854, 257), (p).y))).and(((p).x).lessThan(interpLine(vec2(1353, 0), vec2(1353, 1078), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThan(211))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(174, 0), vec2(174, 1078), (p).y))).and(((p).x).lessThan(interpLine(vec2(1353, 0), vec2(1353, 1078), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1527, 0), vec2(1527, 1467), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_M = (height, p) => {
  const blocked = (any((p).lessThan(vec3(178, 0, 0)))).or(any((p).greaterThan(vec3(1705, 1467, height))));
  const q = p.sub(vec3(178.00,0.00,0.0));
  return select(blocked, float(0.0), glyph_M_1(q));
};

const glyph_R_1 = (p) => {
  const blocked = (((((((((((((((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(1084, 1467))))).or(((((p).y).greaterThanEqual(1439)).and(((p).y).lessThanEqual(1467))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(654.5, 1439), vec2(553, 1467), vec2(414, 1467), (p).y)))))).or(((((p).y).greaterThanEqual(1358)).and(((p).y).lessThan(1439))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(822, 1358), vec2(756, 1411), vec2(654.5, 1439), (p).y)))))).or(((((p).y).greaterThanEqual(1229.5)).and(((p).y).lessThan(1358))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(920, 1229.5), vec2(888, 1305), vec2(822, 1358), (p).y)))))).or(((((p).y).greaterThanEqual(1061)).and(((p).y).lessThan(1229.5))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(952, 1061), vec2(952, 1154), vec2(920, 1229.5), (p).y)))))).or(((((p).y).greaterThanEqual(915.5)).and(((p).y).lessThan(1061))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(927.5, 915.5), vec2(952, 983), vec2(952, 1061), (p).y)))))).or(((((p).y).greaterThanEqual(794)).and(((p).y).lessThan(915.5))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(856.5, 794), vec2(903, 848), vec2(927.5, 915.5), (p).y)))))).or(((((p).y).greaterThanEqual(702.5)).and(((p).y).lessThan(794))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(742.5, 702.5), vec2(810, 740), vec2(856.5, 794), (p).y)))))).or(((((p).y).greaterThanEqual(645)).and(((p).y).lessThan(702.5))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(591, 645), vec2(675, 665), vec2(742.5, 702.5), (p).y)))))).or(((((p).y).greaterThanEqual(612)).and(((p).y).lessThan(645))).and((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(656, 583), vec2(628, 623), vec2(591, 645), (p).y)))))).or(((((p).y).greaterThanEqual(601.5)).and(((p).y).lessThan(612))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(197, 0), vec2(197, 612), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(409.5, 601.5), vec2(389, 612), vec2(348, 612), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(656, 583), vec2(628, 623), vec2(591, 645), (p).y)))))).or(((((p).y).greaterThanEqual(583)).and(((p).y).lessThan(601.5))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(197, 0), vec2(197, 612), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(447, 566), vec2(430, 591), vec2(409.5, 601.5), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(656, 583), vec2(628, 623), vec2(591, 645), (p).y)))))).or(((((p).y).greaterThanEqual(566)).and(((p).y).lessThan(583))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(197, 0), vec2(197, 612), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(447, 566), vec2(430, 591), vec2(409.5, 601.5), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1084, 0), vec2(656, 583), (p).y)))))).or(((((p).y).greaterThanEqual(42)).and(((p).y).lessThan(566))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(197, 0), vec2(197, 612), (p).y))).and(((p).x).lessThan(interpLine(vec2(828, 42), vec2(447, 566), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1084, 0), vec2(656, 583), (p).y)))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThan(42))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1467), (p).y))).or((((p).x).greaterThan(interpLine(vec2(197, 0), vec2(197, 612), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(908, 0), vec2(854, 0), vec2(828, 42), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(1084, 0), vec2(656, 583), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_R_2 = (p) => {
  const blocked = (((((((any(((p).xy).lessThan(vec2(197, 757)))).or(any(((p).xy).greaterThan(vec2(758, 1311))))).or(((((p).y).greaterThanEqual(1244.5)).and(((p).y).lessThanEqual(1311))).and((((p).x).lessThan(interpLine(vec2(197, 757), vec2(197, 1311), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(671.5, 1244.5), vec2(585, 1311), vec2(414, 1311), (p).y)))))).or(((((p).y).greaterThanEqual(1047)).and(((p).y).lessThan(1244.5))).and((((p).x).lessThan(interpLine(vec2(197, 757), vec2(197, 1311), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(758, 1047), vec2(758, 1178), vec2(671.5, 1244.5), (p).y)))))).or(((((p).y).greaterThanEqual(929)).and(((p).y).lessThan(1047))).and((((p).x).lessThan(interpLine(vec2(197, 757), vec2(197, 1311), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(735.5, 929), vec2(758, 982), vec2(758, 1047), (p).y)))))).or(((((p).y).greaterThanEqual(837.5)).and(((p).y).lessThan(929))).and((((p).x).lessThan(interpLine(vec2(197, 757), vec2(197, 1311), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(668.5, 837.5), vec2(713, 876), vec2(735.5, 929), (p).y)))))).or(((((p).y).greaterThanEqual(778)).and(((p).y).lessThan(837.5))).and((((p).x).lessThan(interpLine(vec2(197, 757), vec2(197, 1311), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(558, 778), vec2(624, 799), vec2(668.5, 837.5), (p).y)))))).or(((((p).y).greaterThanEqual(757)).and(((p).y).lessThan(778))).and((((p).x).lessThan(interpLine(vec2(197, 757), vec2(197, 1311), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(405, 757), vec2(492, 757), vec2(558, 778), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_R = (height, p) => {
  const blocked = (any((p).lessThan(vec3(199, 0, 0)))).or(any((p).greaterThan(vec3(1283, 1467, height))));
  const q = p.sub(vec3(199.00,0.00,0.0));
  return select(blocked, float(0.0), glyph_R_1(q).sub(glyph_R_2(q)));
};

const glyph_f_1 = (p) => {
  const blocked = ((((((((((((((((((((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(649, 1489))))).or(((((p).y).greaterThanEqual(1468)).and(((p).y).lessThanEqual(1489))).and((((p).x).lessThan(interpQuadratic(vec2(372.5, 1464.5), vec2(438, 1489), vec2(520, 1489), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(649, 1468), vec2(590, 1489), vec2(520, 1489), (p).y)))))).or(((((p).y).greaterThanEqual(1464.5)).and(((p).y).lessThan(1468))).and((((p).x).lessThan(interpQuadratic(vec2(372.5, 1464.5), vec2(438, 1489), vec2(520, 1489), (p).y))).or(((p).x).greaterThan(interpLine(vec2(645, 1377), vec2(649, 1468), (p).y)))))).or(((((p).y).greaterThanEqual(1392.5)).and(((p).y).lessThan(1464.5))).and((((p).x).lessThan(interpQuadratic(vec2(260.5, 1392.5), vec2(307, 1440), vec2(372.5, 1464.5), (p).y))).or(((p).x).greaterThan(interpLine(vec2(645, 1377), vec2(649, 1468), (p).y)))))).or(((((p).y).greaterThanEqual(1377)).and(((p).y).lessThan(1392.5))).and((((p).x).lessThan(interpQuadratic(vec2(188.5, 1275.5), vec2(214, 1345), vec2(260.5, 1392.5), (p).y))).or(((p).x).greaterThan(interpLine(vec2(645, 1377), vec2(649, 1468), (p).y)))))).or(((((p).y).greaterThanEqual(1353)).and(((p).y).lessThan(1377))).and((((p).x).lessThan(interpQuadratic(vec2(188.5, 1275.5), vec2(214, 1345), vec2(260.5, 1392.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(627.5, 1353), vec2(644, 1357), vec2(645, 1377), (p).y)))))).or(((((p).y).greaterThanEqual(1349)).and(((p).y).lessThan(1353))).and((((p).x).lessThan(interpQuadratic(vec2(188.5, 1275.5), vec2(214, 1345), vec2(260.5, 1392.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(581, 1349), vec2(611, 1349), vec2(627.5, 1353), (p).y)))))).or(((((p).y).greaterThanEqual(1336.5)).and(((p).y).lessThan(1349))).and((((p).x).lessThan(interpQuadratic(vec2(188.5, 1275.5), vec2(214, 1345), vec2(260.5, 1392.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(464.5, 1336.5), vec2(503, 1349), vec2(550, 1349), (p).y)))))).or(((((p).y).greaterThanEqual(1296.5)).and(((p).y).lessThan(1336.5))).and((((p).x).lessThan(interpQuadratic(vec2(188.5, 1275.5), vec2(214, 1345), vec2(260.5, 1392.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(398, 1296.5), vec2(426, 1324), vec2(464.5, 1336.5), (p).y)))))).or(((((p).y).greaterThanEqual(1275.5)).and(((p).y).lessThan(1296.5))).and((((p).x).lessThan(interpQuadratic(vec2(188.5, 1275.5), vec2(214, 1345), vec2(260.5, 1392.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(355.5, 1224), vec2(370, 1269), vec2(398, 1296.5), (p).y)))))).or(((((p).y).greaterThanEqual(1224)).and(((p).y).lessThan(1275.5))).and((((p).x).lessThan(interpQuadratic(vec2(163, 1117), vec2(163, 1206), vec2(188.5, 1275.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(355.5, 1224), vec2(370, 1269), vec2(398, 1296.5), (p).y)))))).or(((((p).y).greaterThanEqual(1117)).and(((p).y).lessThan(1224))).and((((p).x).lessThan(interpQuadratic(vec2(163, 1117), vec2(163, 1206), vec2(188.5, 1275.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(341, 1112), vec2(341, 1179), vec2(355.5, 1224), (p).y)))))).or(((((p).y).greaterThanEqual(1112)).and(((p).y).lessThan(1117))).and((((p).x).lessThan(interpLine(vec2(163, 1017), vec2(163, 1117), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(341, 1112), vec2(341, 1179), vec2(355.5, 1224), (p).y)))))).or(((((p).y).greaterThanEqual(1017)).and(((p).y).lessThan(1112))).and((((p).x).lessThan(interpLine(vec2(163, 1017), vec2(163, 1117), (p).y))).or(((p).x).greaterThan(interpLine(vec2(341, 1017), vec2(341, 1112), (p).y)))))).or(((((p).y).greaterThanEqual(942)).and(((p).y).lessThan(1017))).and((((p).x).lessThan(interpLine(vec2(0, 942), vec2(0, 1017), (p).y))).or(((p).x).greaterThan(interpLine(vec2(641, 885), vec2(641, 1017), (p).y)))))).or(((((p).y).greaterThanEqual(911)).and(((p).y).lessThan(942))).and((((p).x).lessThan(interpQuadratic(vec2(13.5, 911), vec2(0, 922), vec2(0, 942), (p).y))).or(((p).x).greaterThan(interpLine(vec2(641, 885), vec2(641, 1017), (p).y)))))).or(((((p).y).greaterThanEqual(895)).and(((p).y).lessThan(911))).and((((p).x).lessThan(interpQuadratic(vec2(49, 895), vec2(27, 900), vec2(13.5, 911), (p).y))).or(((p).x).greaterThan(interpLine(vec2(641, 885), vec2(641, 1017), (p).y)))))).or(((((p).y).greaterThanEqual(885)).and(((p).y).lessThan(895))).and((((p).x).lessThan(interpLine(vec2(163, 882), vec2(49, 895), (p).y))).or(((p).x).greaterThan(interpLine(vec2(641, 885), vec2(641, 1017), (p).y)))))).or(((((p).y).greaterThanEqual(882)).and(((p).y).lessThan(885))).and((((p).x).lessThan(interpLine(vec2(163, 882), vec2(49, 895), (p).y))).or(((p).x).greaterThan(interpLine(vec2(347, 0), vec2(347, 885), (p).y)))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThan(882))).and((((p).x).lessThan(interpLine(vec2(163, 0), vec2(163, 882), (p).y))).or(((p).x).greaterThan(interpLine(vec2(347, 0), vec2(347, 885), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_f = (height, p) => {
  const blocked = (any((p).lessThan(vec3(27, 0, 0)))).or(any((p).greaterThan(vec3(676, 1489, height))));
  const q = p.sub(vec3(27.00,0.00,0.0));
  return select(blocked, float(0.0), glyph_f_1(q));
};

const glyph_n_1 = (p) => {
  const blocked = ((((((((((((((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(863, 1054))))).or(((((p).y).greaterThanEqual(1037)).and(((p).y).lessThanEqual(1054))).and((((p).x).lessThan(interpQuadratic(vec2(323, 1008), vec2(406, 1054), vec2(516, 1054), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(666, 1025.5), vec2(601, 1054), vec2(516, 1054), (p).y)))))).or(((((p).y).greaterThanEqual(1025.5)).and(((p).y).lessThan(1037))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(157, 999), vec2(147, 1037), vec2(108, 1037), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(323, 1008), vec2(406, 1054), vec2(516, 1054), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(666, 1025.5), vec2(601, 1054), vec2(516, 1054), (p).y)))))).or(((((p).y).greaterThanEqual(1008)).and(((p).y).lessThan(1025.5))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(157, 999), vec2(147, 1037), vec2(108, 1037), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(323, 1008), vec2(406, 1054), vec2(516, 1054), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(774.5, 945.5), vec2(731, 997), vec2(666, 1025.5), (p).y)))))).or(((((p).y).greaterThanEqual(999)).and(((p).y).lessThan(1008))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(157, 999), vec2(147, 1037), vec2(108, 1037), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(172, 887), vec2(240, 962), vec2(323, 1008), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(774.5, 945.5), vec2(731, 997), vec2(666, 1025.5), (p).y)))))).or(((((p).y).greaterThanEqual(945.5)).and(((p).y).lessThan(999))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpLine(vec2(172, 887), vec2(157, 999), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(172, 887), vec2(240, 962), vec2(323, 1008), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(774.5, 945.5), vec2(731, 997), vec2(666, 1025.5), (p).y)))))).or(((((p).y).greaterThanEqual(908)).and(((p).y).lessThan(945.5))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpLine(vec2(172, 887), vec2(157, 999), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(172, 887), vec2(240, 962), vec2(323, 1008), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(840.5, 821.5), vec2(818, 894), vec2(774.5, 945.5), (p).y)))))).or(((((p).y).greaterThanEqual(887)).and(((p).y).lessThan(908))).and((((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpLine(vec2(172, 887), vec2(157, 999), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(172, 887), vec2(240, 962), vec2(323, 1008), (p).y))))).or((((p).x).greaterThan(interpQuadratic(vec2(311, 869), vec2(381, 908), vec2(462, 908), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(626.5, 843), vec2(573, 908), vec2(462, 908), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(840.5, 821.5), vec2(818, 894), vec2(774.5, 945.5), (p).y)))))).or(((((p).y).greaterThanEqual(869)).and(((p).y).lessThan(887))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(311, 869), vec2(381, 908), vec2(462, 908), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(626.5, 843), vec2(573, 908), vec2(462, 908), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(840.5, 821.5), vec2(818, 894), vec2(774.5, 945.5), (p).y)))))).or(((((p).y).greaterThanEqual(843)).and(((p).y).lessThan(869))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(182, 764), vec2(241, 830), vec2(311, 869), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(626.5, 843), vec2(573, 908), vec2(462, 908), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(840.5, 821.5), vec2(818, 894), vec2(774.5, 945.5), (p).y)))))).or(((((p).y).greaterThanEqual(821.5)).and(((p).y).lessThan(843))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(182, 764), vec2(241, 830), vec2(311, 869), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(680, 660), vec2(680, 778), vec2(626.5, 843), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(840.5, 821.5), vec2(818, 894), vec2(774.5, 945.5), (p).y)))))).or(((((p).y).greaterThanEqual(764)).and(((p).y).lessThan(821.5))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(182, 764), vec2(241, 830), vec2(311, 869), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(680, 660), vec2(680, 778), vec2(626.5, 843), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(863, 660), vec2(863, 749), vec2(840.5, 821.5), (p).y)))))).or(((((p).y).greaterThanEqual(660)).and(((p).y).lessThan(764))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpLine(vec2(182, 0), vec2(182, 764), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(680, 660), vec2(680, 778), vec2(626.5, 843), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(863, 660), vec2(863, 749), vec2(840.5, 821.5), (p).y)))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThan(660))).and(((((p).x).lessThan(interpLine(vec2(0, 0), vec2(0, 1037), (p).y))).or((((p).x).greaterThan(interpLine(vec2(182, 0), vec2(182, 764), (p).y))).and(((p).x).lessThan(interpLine(vec2(680, 0), vec2(680, 660), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(863, 0), vec2(863, 660), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_n = (height, p) => {
  const blocked = (any((p).lessThan(vec3(150, 0, 0)))).or(any((p).greaterThan(vec3(1013, 1054, height))));
  const q = p.sub(vec3(150.00,0.00,0.0));
  return select(blocked, float(0.0), glyph_n_1(q));
};

const glyph_o_1 = (p) => {
  const blocked = (((((((((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(989, 1068))))).or(((((p).y).greaterThanEqual(1030)).and(((p).y).lessThanEqual(1068))).and((((p).x).lessThan(interpQuadratic(vec2(290, 1030), vec2(382, 1068), vec2(495, 1068), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(700.5, 1030), vec2(609, 1068), vec2(495, 1068), (p).y)))))).or(((((p).y).greaterThanEqual(922.5)).and(((p).y).lessThan(1030))).and((((p).x).lessThan(interpQuadratic(vec2(133.5, 922.5), vec2(198, 992), vec2(290, 1030), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(856, 922.5), vec2(792, 992), vec2(700.5, 1030), (p).y)))))).or(((((p).y).greaterThanEqual(754)).and(((p).y).lessThan(922.5))).and((((p).x).lessThan(interpQuadratic(vec2(34.5, 754), vec2(69, 853), vec2(133.5, 922.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(954.5, 754), vec2(920, 853), vec2(856, 922.5), (p).y)))))).or(((((p).y).greaterThanEqual(533)).and(((p).y).lessThan(754))).and((((p).x).lessThan(interpQuadratic(vec2(0, 533), vec2(0, 655), vec2(34.5, 754), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(989, 533), vec2(989, 655), vec2(954.5, 754), (p).y)))))).or(((((p).y).greaterThanEqual(312)).and(((p).y).lessThan(533))).and((((p).x).lessThan(interpQuadratic(vec2(34.5, 312), vec2(0, 410), vec2(0, 533), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(954.5, 312), vec2(989, 410), vec2(989, 533), (p).y)))))).or(((((p).y).greaterThanEqual(144)).and(((p).y).lessThan(312))).and((((p).x).lessThan(interpQuadratic(vec2(133.5, 144), vec2(69, 214), vec2(34.5, 312), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(856, 144), vec2(920, 214), vec2(954.5, 312), (p).y)))))).or(((((p).y).greaterThanEqual(37)).and(((p).y).lessThan(144))).and((((p).x).lessThan(interpQuadratic(vec2(289.5, 37), vec2(198, 74), vec2(133.5, 144), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(700.5, 37), vec2(792, 74), vec2(856, 144), (p).y)))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThan(37))).and((((p).x).lessThan(interpQuadratic(vec2(495, 0), vec2(381, 0), vec2(289.5, 37), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(495, 0), vec2(609, 0), vec2(700.5, 37), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_o_2 = (p) => {
  const blocked = ((((((((((any(((p).xy).lessThan(vec2(188, 142)))).or(any(((p).xy).greaterThan(vec2(800, 924))))).or(((((p).y).greaterThanEqual(897.5)).and(((p).y).lessThanEqual(924))).and((((p).x).lessThan(interpQuadratic(vec2(360, 897.5), vec2(417, 924), vec2(495, 924), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(724.5, 820.5), vec2(649, 924), vec2(495, 924), (p).y)))))).or(((((p).y).greaterThanEqual(821)).and(((p).y).lessThan(897.5))).and((((p).x).lessThan(interpQuadratic(vec2(264.5, 821), vec2(303, 871), vec2(360, 897.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(724.5, 820.5), vec2(649, 924), vec2(495, 924), (p).y)))))).or(((((p).y).greaterThanEqual(820.5)).and(((p).y).lessThan(821))).and((((p).x).lessThan(interpQuadratic(vec2(207, 697.5), vec2(226, 771), vec2(264.5, 821), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(724.5, 820.5), vec2(649, 924), vec2(495, 924), (p).y)))))).or(((((p).y).greaterThanEqual(697.5)).and(((p).y).lessThan(820.5))).and((((p).x).lessThan(interpQuadratic(vec2(207, 697.5), vec2(226, 771), vec2(264.5, 821), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(800, 532), vec2(800, 717), vec2(724.5, 820.5), (p).y)))))).or(((((p).y).greaterThanEqual(532)).and(((p).y).lessThan(697.5))).and((((p).x).lessThan(interpQuadratic(vec2(188, 532), vec2(188, 624), vec2(207, 697.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(800, 532), vec2(800, 717), vec2(724.5, 820.5), (p).y)))))).or(((((p).y).greaterThanEqual(367.5)).and(((p).y).lessThan(532))).and((((p).x).lessThan(interpQuadratic(vec2(207, 367.5), vec2(188, 440), vec2(188, 532), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(724.5, 245), vec2(800, 348), vec2(800, 532), (p).y)))))).or(((((p).y).greaterThanEqual(245)).and(((p).y).lessThan(367.5))).and((((p).x).lessThan(interpQuadratic(vec2(264.5, 245), vec2(226, 295), vec2(207, 367.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(724.5, 245), vec2(800, 348), vec2(800, 532), (p).y)))))).or(((((p).y).greaterThanEqual(168.5)).and(((p).y).lessThan(245))).and((((p).x).lessThan(interpQuadratic(vec2(360.5, 168.5), vec2(303, 195), vec2(264.5, 245), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(495, 142), vec2(649, 142), vec2(724.5, 245), (p).y)))))).or(((((p).y).greaterThanEqual(142)).and(((p).y).lessThan(168.5))).and((((p).x).lessThan(interpQuadratic(vec2(495, 142), vec2(418, 142), vec2(360.5, 168.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(495, 142), vec2(649, 142), vec2(724.5, 245), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_o = (height, p) => {
  const blocked = (any((p).lessThan(vec3(74, -14, 0)))).or(any((p).greaterThan(vec3(1063, 1054, height))));
  const q = p.sub(vec3(74.00,-14.00,0.0));
  return select(blocked, float(0.0), glyph_o_1(q).sub(glyph_o_2(q)));
};

const glyph_s_1 = (p) => {
  const blocked = (((((((((((((((((((((((((((((((((((((((((((((((((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(739, 1070))))).or(((((p).y).greaterThanEqual(1045.5)).and(((p).y).lessThanEqual(1070))).and((((p).x).lessThan(interpQuadratic(vec2(240, 1045.5), vec2(309, 1070), vec2(397, 1070), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(580.5, 1037.5), vec2(499, 1070), vec2(397, 1070), (p).y)))))).or(((((p).y).greaterThanEqual(1037.5)).and(((p).y).lessThan(1045.5))).and((((p).x).lessThan(interpQuadratic(vec2(124, 979.5), vec2(171, 1021), vec2(240, 1045.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(580.5, 1037.5), vec2(499, 1070), vec2(397, 1070), (p).y)))))).or(((((p).y).greaterThanEqual(979.5)).and(((p).y).lessThan(1037.5))).and((((p).x).lessThan(interpQuadratic(vec2(124, 979.5), vec2(171, 1021), vec2(240, 1045.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(721, 949), vec2(662, 1005), vec2(580.5, 1037.5), (p).y)))))).or(((((p).y).greaterThanEqual(949)).and(((p).y).lessThan(979.5))).and((((p).x).lessThan(interpQuadratic(vec2(52.5, 883), vec2(77, 938), vec2(124, 979.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(721, 949), vec2(662, 1005), vec2(580.5, 1037.5), (p).y)))))).or(((((p).y).greaterThanEqual(934)).and(((p).y).lessThan(949))).and((((p).x).lessThan(interpQuadratic(vec2(52.5, 883), vec2(77, 938), vec2(124, 979.5), (p).y))).or(((p).x).greaterThan(interpLine(vec2(680, 882), vec2(721, 949), (p).y)))))).or(((((p).y).greaterThanEqual(922)).and(((p).y).lessThan(934))).and(((((p).x).lessThan(interpQuadratic(vec2(52.5, 883), vec2(77, 938), vec2(124, 979.5), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(318, 922), vec2(355, 934), vec2(401, 934), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(493, 922), vec2(454, 934), vec2(401, 934), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(680, 882), vec2(721, 949), (p).y)))))).or(((((p).y).greaterThanEqual(896)).and(((p).y).lessThan(922))).and(((((p).x).lessThan(interpQuadratic(vec2(52.5, 883), vec2(77, 938), vec2(124, 979.5), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(255, 889.5), vec2(281, 910), vec2(318, 922), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(560, 896), vec2(532, 910), vec2(493, 922), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(680, 882), vec2(721, 949), (p).y)))))).or(((((p).y).greaterThanEqual(889.5)).and(((p).y).lessThan(896))).and(((((p).x).lessThan(interpQuadratic(vec2(52.5, 883), vec2(77, 938), vec2(124, 979.5), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(255, 889.5), vec2(281, 910), vec2(318, 922), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(608, 871), vec2(588, 882), vec2(560, 896), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(680, 882), vec2(721, 949), (p).y)))))).or(((((p).y).greaterThanEqual(883)).and(((p).y).lessThan(889.5))).and(((((p).x).lessThan(interpQuadratic(vec2(52.5, 883), vec2(77, 938), vec2(124, 979.5), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(215, 842), vec2(229, 869), vec2(255, 889.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(608, 871), vec2(588, 882), vec2(560, 896), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(680, 882), vec2(721, 949), (p).y)))))).or(((((p).y).greaterThanEqual(882)).and(((p).y).lessThan(883))).and(((((p).x).lessThan(interpQuadratic(vec2(28, 769), vec2(28, 828), vec2(52.5, 883), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(215, 842), vec2(229, 869), vec2(255, 889.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(608, 871), vec2(588, 882), vec2(560, 896), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(680, 882), vec2(721, 949), (p).y)))))).or(((((p).y).greaterThanEqual(871)).and(((p).y).lessThan(882))).and(((((p).x).lessThan(interpQuadratic(vec2(28, 769), vec2(28, 828), vec2(52.5, 883), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(215, 842), vec2(229, 869), vec2(255, 889.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(608, 871), vec2(588, 882), vec2(560, 896), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(643, 860), vec2(668, 859), vec2(680, 882), (p).y)))))).or(((((p).y).greaterThanEqual(860)).and(((p).y).lessThan(871))).and(((((p).x).lessThan(interpQuadratic(vec2(28, 769), vec2(28, 828), vec2(52.5, 883), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(215, 842), vec2(229, 869), vec2(255, 889.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(643, 860), vec2(628, 860), vec2(608, 871), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(643, 860), vec2(668, 859), vec2(680, 882), (p).y)))))).or(((((p).y).greaterThanEqual(859)).and(((p).y).lessThan(860))).and(((((p).x).lessThan(interpQuadratic(vec2(28, 769), vec2(28, 828), vec2(52.5, 883), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(215, 842), vec2(229, 869), vec2(255, 889.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(643, 860), vec2(668, 859), vec2(680, 882), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(643, 860), vec2(668, 859), vec2(680, 882), (p).y)))))).or(((((p).y).greaterThanEqual(842)).and(((p).y).lessThan(859))).and((((p).x).lessThan(interpQuadratic(vec2(28, 769), vec2(28, 828), vec2(52.5, 883), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(215, 842), vec2(229, 869), vec2(255, 889.5), (p).y)))))).or(((((p).y).greaterThanEqual(783)).and(((p).y).lessThan(842))).and((((p).x).lessThan(interpQuadratic(vec2(28, 769), vec2(28, 828), vec2(52.5, 883), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(201, 783), vec2(201, 815), vec2(215, 842), (p).y)))))).or(((((p).y).greaterThanEqual(769)).and(((p).y).lessThan(783))).and((((p).x).lessThan(interpQuadratic(vec2(28, 769), vec2(28, 828), vec2(52.5, 883), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(224, 716.5), vec2(201, 743), vec2(201, 783), (p).y)))))).or(((((p).y).greaterThanEqual(716.5)).and(((p).y).lessThan(769))).and((((p).x).lessThan(interpQuadratic(vec2(51, 655.5), vec2(28, 702), vec2(28, 769), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(224, 716.5), vec2(201, 743), vec2(201, 783), (p).y)))))).or(((((p).y).greaterThanEqual(670.5)).and(((p).y).lessThan(716.5))).and((((p).x).lessThan(interpQuadratic(vec2(51, 655.5), vec2(28, 702), vec2(28, 769), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(285, 670.5), vec2(247, 690), vec2(224, 716.5), (p).y)))))).or(((((p).y).greaterThanEqual(655.5)).and(((p).y).lessThan(670.5))).and((((p).x).lessThan(interpQuadratic(vec2(51, 655.5), vec2(28, 702), vec2(28, 769), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(371, 636), vec2(323, 651), vec2(285, 670.5), (p).y)))))).or(((((p).y).greaterThanEqual(636)).and(((p).y).lessThan(655.5))).and((((p).x).lessThan(interpQuadratic(vec2(112, 577), vec2(74, 609), vec2(51, 655.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(371, 636), vec2(323, 651), vec2(285, 670.5), (p).y)))))).or(((((p).y).greaterThanEqual(604)).and(((p).y).lessThan(636))).and((((p).x).lessThan(interpQuadratic(vec2(112, 577), vec2(74, 609), vec2(51, 655.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(470, 604), vec2(419, 621), vec2(371, 636), (p).y)))))).or(((((p).y).greaterThanEqual(577)).and(((p).y).lessThan(604))).and((((p).x).lessThan(interpQuadratic(vec2(112, 577), vec2(74, 609), vec2(51, 655.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(569, 566.5), vec2(521, 587), vec2(470, 604), (p).y)))))).or(((((p).y).greaterThanEqual(566.5)).and(((p).y).lessThan(577))).and((((p).x).lessThan(interpQuadratic(vec2(198.5, 524), vec2(150, 545), vec2(112, 577), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(569, 566.5), vec2(521, 587), vec2(470, 604), (p).y)))))).or(((((p).y).greaterThanEqual(524)).and(((p).y).lessThan(566.5))).and((((p).x).lessThan(interpQuadratic(vec2(198.5, 524), vec2(150, 545), vec2(112, 577), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(655, 515.5), vec2(617, 546), vec2(569, 566.5), (p).y)))))).or(((((p).y).greaterThanEqual(515.5)).and(((p).y).lessThan(524))).and((((p).x).lessThan(interpQuadratic(vec2(298, 487), vec2(247, 503), vec2(198.5, 524), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(655, 515.5), vec2(617, 546), vec2(569, 566.5), (p).y)))))).or(((((p).y).greaterThanEqual(487)).and(((p).y).lessThan(515.5))).and((((p).x).lessThan(interpQuadratic(vec2(298, 487), vec2(247, 503), vec2(198.5, 524), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(716, 440.5), vec2(693, 485), vec2(655, 515.5), (p).y)))))).or(((((p).y).greaterThanEqual(456)).and(((p).y).lessThan(487))).and((((p).x).lessThan(interpQuadratic(vec2(397.5, 456), vec2(349, 471), vec2(298, 487), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(716, 440.5), vec2(693, 485), vec2(655, 515.5), (p).y)))))).or(((((p).y).greaterThanEqual(440.5)).and(((p).y).lessThan(456))).and((((p).x).lessThan(interpQuadratic(vec2(484, 421), vec2(446, 441), vec2(397.5, 456), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(716, 440.5), vec2(693, 485), vec2(655, 515.5), (p).y)))))).or(((((p).y).greaterThanEqual(421)).and(((p).y).lessThan(440.5))).and((((p).x).lessThan(interpQuadratic(vec2(484, 421), vec2(446, 441), vec2(397.5, 456), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(739, 333), vec2(739, 396), vec2(716, 440.5), (p).y)))))).or(((((p).y).greaterThanEqual(373)).and(((p).y).lessThan(421))).and((((p).x).lessThan(interpQuadratic(vec2(545, 373), vec2(522, 401), vec2(484, 421), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(739, 333), vec2(739, 396), vec2(716, 440.5), (p).y)))))).or(((((p).y).greaterThanEqual(333)).and(((p).y).lessThan(373))).and((((p).x).lessThan(interpQuadratic(vec2(568, 302), vec2(568, 345), vec2(545, 373), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(739, 333), vec2(739, 396), vec2(716, 440.5), (p).y)))))).or(((((p).y).greaterThanEqual(302)).and(((p).y).lessThan(333))).and((((p).x).lessThan(interpQuadratic(vec2(568, 302), vec2(568, 345), vec2(545, 373), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(713.5, 200.5), vec2(739, 261), vec2(739, 333), (p).y)))))).or(((((p).y).greaterThanEqual(236)).and(((p).y).lessThan(302))).and((((p).x).lessThan(interpQuadratic(vec2(555, 236), vec2(568, 267), vec2(568, 302), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(713.5, 200.5), vec2(739, 261), vec2(739, 333), (p).y)))))).or(((((p).y).greaterThanEqual(223)).and(((p).y).lessThan(236))).and((((p).x).lessThan(interpQuadratic(vec2(515.5, 181.5), vec2(542, 205), vec2(555, 236), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(713.5, 200.5), vec2(739, 261), vec2(739, 333), (p).y)))))).or(((((p).y).greaterThanEqual(215.5)).and(((p).y).lessThan(223))).and(((((p).x).lessThan(interpQuadratic(vec2(62.5, 215.5), vec2(74, 223), vec2(93, 223), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(131.5, 208.5), vec2(111, 223), vec2(93, 223), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(515.5, 181.5), vec2(542, 205), vec2(555, 236), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(713.5, 200.5), vec2(739, 261), vec2(739, 333), (p).y)))))).or(((((p).y).greaterThanEqual(208.5)).and(((p).y).lessThan(215.5))).and(((((p).x).lessThan(interpQuadratic(vec2(43, 195), vec2(51, 208), vec2(62.5, 215.5), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(131.5, 208.5), vec2(111, 223), vec2(93, 223), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(515.5, 181.5), vec2(542, 205), vec2(555, 236), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(713.5, 200.5), vec2(739, 261), vec2(739, 333), (p).y)))))).or(((((p).y).greaterThanEqual(200.5)).and(((p).y).lessThan(208.5))).and(((((p).x).lessThan(interpQuadratic(vec2(43, 195), vec2(51, 208), vec2(62.5, 215.5), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(181, 176.5), vec2(152, 194), vec2(131.5, 208.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(515.5, 181.5), vec2(542, 205), vec2(555, 236), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(713.5, 200.5), vec2(739, 261), vec2(739, 333), (p).y)))))).or(((((p).y).greaterThanEqual(195)).and(((p).y).lessThan(200.5))).and(((((p).x).lessThan(interpQuadratic(vec2(43, 195), vec2(51, 208), vec2(62.5, 215.5), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(181, 176.5), vec2(152, 194), vec2(131.5, 208.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(515.5, 181.5), vec2(542, 205), vec2(555, 236), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(637.5, 95.5), vec2(688, 140), vec2(713.5, 200.5), (p).y)))))).or(((((p).y).greaterThanEqual(181.5)).and(((p).y).lessThan(195))).and(((((p).x).lessThan(interpLine(vec2(0, 126), vec2(43, 195), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(181, 176.5), vec2(152, 194), vec2(131.5, 208.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(515.5, 181.5), vec2(542, 205), vec2(555, 236), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(637.5, 95.5), vec2(688, 140), vec2(713.5, 200.5), (p).y)))))).or(((((p).y).greaterThanEqual(176.5)).and(((p).y).lessThan(181.5))).and(((((p).x).lessThan(interpLine(vec2(0, 126), vec2(43, 195), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(181, 176.5), vec2(152, 194), vec2(131.5, 208.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(449, 144.5), vec2(489, 158), vec2(515.5, 181.5), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(637.5, 95.5), vec2(688, 140), vec2(713.5, 200.5), (p).y)))))).or(((((p).y).greaterThanEqual(145)).and(((p).y).lessThan(176.5))).and(((((p).x).lessThan(interpLine(vec2(0, 126), vec2(43, 195), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(251.5, 145), vec2(210, 159), vec2(181, 176.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(449, 144.5), vec2(489, 158), vec2(515.5, 181.5), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(637.5, 95.5), vec2(688, 140), vec2(713.5, 200.5), (p).y)))))).or(((((p).y).greaterThanEqual(144.5)).and(((p).y).lessThan(145))).and(((((p).x).lessThan(interpLine(vec2(0, 126), vec2(43, 195), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(356, 131), vec2(293, 131), vec2(251.5, 145), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(449, 144.5), vec2(489, 158), vec2(515.5, 181.5), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(637.5, 95.5), vec2(688, 140), vec2(713.5, 200.5), (p).y)))))).or(((((p).y).greaterThanEqual(131)).and(((p).y).lessThan(144.5))).and(((((p).x).lessThan(interpLine(vec2(0, 126), vec2(43, 195), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(356, 131), vec2(293, 131), vec2(251.5, 145), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(356, 131), vec2(409, 131), vec2(449, 144.5), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(637.5, 95.5), vec2(688, 140), vec2(713.5, 200.5), (p).y)))))).or(((((p).y).greaterThanEqual(126)).and(((p).y).lessThan(131))).and((((p).x).lessThan(interpLine(vec2(0, 126), vec2(43, 195), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(637.5, 95.5), vec2(688, 140), vec2(713.5, 200.5), (p).y)))))).or(((((p).y).greaterThanEqual(95.5)).and(((p).y).lessThan(126))).and((((p).x).lessThan(interpQuadratic(vec2(149.5, 35.5), vec2(61, 71), vec2(0, 126), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(637.5, 95.5), vec2(688, 140), vec2(713.5, 200.5), (p).y)))))).or(((((p).y).greaterThanEqual(35.5)).and(((p).y).lessThan(95.5))).and((((p).x).lessThan(interpQuadratic(vec2(149.5, 35.5), vec2(61, 71), vec2(0, 126), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(514.5, 25.5), vec2(587, 51), vec2(637.5, 95.5), (p).y)))))).or(((((p).y).greaterThanEqual(25.5)).and(((p).y).lessThan(35.5))).and((((p).x).lessThan(interpQuadratic(vec2(347, 0), vec2(238, 0), vec2(149.5, 35.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(514.5, 25.5), vec2(587, 51), vec2(637.5, 95.5), (p).y)))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThan(25.5))).and((((p).x).lessThan(interpQuadratic(vec2(347, 0), vec2(238, 0), vec2(149.5, 35.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(347, 0), vec2(442, 0), vec2(514.5, 25.5), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_s = (height, p) => {
  const blocked = (any((p).lessThan(vec3(63, -16, 0)))).or(any((p).greaterThan(vec3(802, 1054, height))));
  const q = p.sub(vec3(63.00,-16.00,0.0));
  return select(blocked, float(0.0), glyph_s_1(q));
};

const glyph_t_1 = (p) => {
  const blocked = ((((((((((((((((((((((any(((p).xy).lessThan(vec2(0, 0)))).or(any(((p).xy).greaterThan(vec2(670, 1390))))).or(((((p).y).greaterThanEqual(1380)).and(((p).y).lessThanEqual(1390))).and((((p).x).lessThan(interpQuadratic(vec2(225.5, 1380), vec2(237, 1390), vec2(254, 1390), (p).y))).or(((p).x).greaterThan(interpLine(vec2(346, 1033), vec2(346, 1390), (p).y)))))).or(((((p).y).greaterThanEqual(1355)).and(((p).y).lessThan(1380))).and((((p).x).lessThan(interpQuadratic(vec2(212, 1355), vec2(214, 1370), vec2(225.5, 1380), (p).y))).or(((p).x).greaterThan(interpLine(vec2(346, 1033), vec2(346, 1390), (p).y)))))).or(((((p).y).greaterThanEqual(1035)).and(((p).y).lessThan(1355))).and((((p).x).lessThan(interpLine(vec2(170, 1035), vec2(212, 1355), (p).y))).or(((p).x).greaterThan(interpLine(vec2(346, 1033), vec2(346, 1390), (p).y)))))).or(((((p).y).greaterThanEqual(1033)).and(((p).y).lessThan(1035))).and((((p).x).lessThan(interpLine(vec2(0, 1013), vec2(170, 1035), (p).y))).or(((p).x).greaterThan(interpLine(vec2(346, 1033), vec2(346, 1390), (p).y)))))).or(((((p).y).greaterThanEqual(1013)).and(((p).y).lessThan(1033))).and((((p).x).lessThan(interpLine(vec2(0, 1013), vec2(170, 1035), (p).y))).or(((p).x).greaterThan(interpLine(vec2(643, 901), vec2(643, 1033), (p).y)))))).or(((((p).y).greaterThanEqual(941)).and(((p).y).lessThan(1013))).and((((p).x).lessThan(interpLine(vec2(0, 941), vec2(0, 1013), (p).y))).or(((p).x).greaterThan(interpLine(vec2(643, 901), vec2(643, 1033), (p).y)))))).or(((((p).y).greaterThanEqual(911)).and(((p).y).lessThan(941))).and((((p).x).lessThan(interpQuadratic(vec2(11.5, 911), vec2(0, 921), vec2(0, 941), (p).y))).or(((p).x).greaterThan(interpLine(vec2(643, 901), vec2(643, 1033), (p).y)))))).or(((((p).y).greaterThanEqual(901)).and(((p).y).lessThan(911))).and((((p).x).lessThan(interpQuadratic(vec2(39, 901), vec2(23, 901), vec2(11.5, 911), (p).y))).or(((p).x).greaterThan(interpLine(vec2(643, 901), vec2(643, 1033), (p).y)))))).or(((((p).y).greaterThanEqual(278)).and(((p).y).lessThan(901))).and((((p).x).lessThan(interpLine(vec2(164, 266), vec2(164, 901), (p).y))).or(((p).x).greaterThan(interpLine(vec2(346, 278), vec2(346, 901), (p).y)))))).or(((((p).y).greaterThanEqual(266)).and(((p).y).lessThan(278))).and((((p).x).lessThan(interpLine(vec2(164, 266), vec2(164, 901), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(378, 180.5), vec2(346, 212), vec2(346, 278), (p).y)))))).or(((((p).y).greaterThanEqual(198)).and(((p).y).lessThan(266))).and((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(378, 180.5), vec2(346, 212), vec2(346, 278), (p).y)))))).or(((((p).y).greaterThanEqual(190.5)).and(((p).y).lessThan(198))).and(((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(378, 180.5), vec2(346, 212), vec2(346, 278), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(572, 190.5), vec2(583, 198), vec2(591, 198), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(617, 181), vec2(605, 198), vec2(591, 198), (p).y)))))).or(((((p).y).greaterThanEqual(181)).and(((p).y).lessThan(190.5))).and(((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(378, 180.5), vec2(346, 212), vec2(346, 278), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(545.5, 173.5), vec2(561, 183), vec2(572, 190.5), (p).y))))).or(((p).x).greaterThan(interpQuadratic(vec2(617, 181), vec2(605, 198), vec2(591, 198), (p).y)))))).or(((((p).y).greaterThanEqual(180.5)).and(((p).y).lessThan(181))).and(((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(378, 180.5), vec2(346, 212), vec2(346, 278), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(545.5, 173.5), vec2(561, 183), vec2(572, 190.5), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(670, 94), vec2(617, 181), (p).y)))))).or(((((p).y).greaterThanEqual(173.5)).and(((p).y).lessThan(180.5))).and(((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(460, 149), vec2(410, 149), vec2(378, 180.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(545.5, 173.5), vec2(561, 183), vec2(572, 190.5), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(670, 94), vec2(617, 181), (p).y)))))).or(((((p).y).greaterThanEqual(156.5)).and(((p).y).lessThan(173.5))).and(((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(460, 149), vec2(410, 149), vec2(378, 180.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(509.5, 156.5), vec2(530, 164), vec2(545.5, 173.5), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(670, 94), vec2(617, 181), (p).y)))))).or(((((p).y).greaterThanEqual(149)).and(((p).y).lessThan(156.5))).and(((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or((((p).x).greaterThan(interpQuadratic(vec2(460, 149), vec2(410, 149), vec2(378, 180.5), (p).y))).and(((p).x).lessThan(interpQuadratic(vec2(460, 149), vec2(489, 149), vec2(509.5, 156.5), (p).y))))).or(((p).x).greaterThan(interpLine(vec2(670, 94), vec2(617, 181), (p).y)))))).or(((((p).y).greaterThanEqual(94)).and(((p).y).lessThan(149))).and((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or(((p).x).greaterThan(interpLine(vec2(670, 94), vec2(617, 181), (p).y)))))).or(((((p).y).greaterThanEqual(68.5)).and(((p).y).lessThan(94))).and((((p).x).lessThan(interpQuadratic(vec2(230, 68.5), vec2(164, 137), vec2(164, 266), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(556.5, 25), vec2(623, 50), vec2(670, 94), (p).y)))))).or(((((p).y).greaterThanEqual(25)).and(((p).y).lessThan(68.5))).and((((p).x).lessThan(interpQuadratic(vec2(419, 0), vec2(296, 0), vec2(230, 68.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(556.5, 25), vec2(623, 50), vec2(670, 94), (p).y)))))).or(((((p).y).greaterThanEqual(0)).and(((p).y).lessThan(25))).and((((p).x).lessThan(interpQuadratic(vec2(419, 0), vec2(296, 0), vec2(230, 68.5), (p).y))).or(((p).x).greaterThan(interpQuadratic(vec2(419, 0), vec2(490, 0), vec2(556.5, 25), (p).y)))));
  return select(blocked, float(0.0), float(1.0));
};

const glyph_t = (height, p) => {
  const blocked = (any((p).lessThan(vec3(45, -16, 0)))).or(any((p).greaterThan(vec3(715, 1374, height))));
  const q = p.sub(vec3(45.00,-16.00,0.0));
  return select(blocked, float(0.0), glyph_t_1(q));
};

const textMessage = (emSize, height, p) => {
  const q = p.mul(vec3(1649.00/emSize,1649.00/emSize,1.0));
  let result = float(0.0);
  result = result.add(glyph_I(height, q.sub(vec3(0.00,0,0))));
  result = result.add(glyph_R(height, q.sub(vec3(628.00,0,0))));
  result = result.add(glyph_M(height, q.sub(vec3(2146.00,0,0))));
  result = result.add(glyph_F(height, q.sub(vec3(4030.00,0,0))));
  result = result.add(glyph_f(height, q.sub(vec3(5584.00,0,0))));
  result = result.add(glyph_o(height, q.sub(vec3(6274.00,0,0))));
  result = result.add(glyph_n(height, q.sub(vec3(7412.00,0,0))));
  result = result.add(glyph_t(height, q.sub(vec3(8550.00,0,0))));
  result = result.add(glyph_s(height, q.sub(vec3(9313.00,0,0))));
  return result;
};

const mainModel4 = (p) => {
  const q = p.add(vec3(3.0, 0.0, 1.5));
  return [textMessage(1, 0.5, q)];
};
```

* Try loading [text-1.irmf (GLSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/016-text/text-1.irmf) now in the IRMF editor!
* Try loading [text-1-wgsl.irmf (WGSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/016-text/text-1-wgsl.irmf) now in the IRMF editor!
* Try loading [text-1-tsl.irmf (TSL)](https://gmlewis.github.io/irmf-editor/?s=github.com/gmlewis/irmf-examples/blob/master/examples/016-text/text-1-tsl.irmf) now in the IRMF editor!

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
