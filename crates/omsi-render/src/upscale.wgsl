// Render scale: the 3D picture, drawn smaller than the window, scaled up to it. A
// Catmull-Rom filter (nine bilinear taps) keeps edges and texture detail far better than a
// plain bilinear stretch; its overshoot is clamped to the neighbouring texels so that
// edges do not ring, and a contrast-adaptive sharpening pass (after AMD's CAS) brings back
// what the smaller picture lost, most where the picture is flat and least on hard edges.
struct Params {
    // xy: size of the source picture, z: sharpening 0..1, w: 1 = the picture at the window's
    // own size smoothed by FXAA instead (the plain graphics without multisampling)
    src: vec4<f32>,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var t_src: texture_2d<f32>;
@group(0) @binding(2) var s_src: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    var out: VsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(t_src, s_src, uv, 0.0).rgb;
}

fn catmull_rom(uv: vec2<f32>, size: vec2<f32>) -> vec3<f32> {
    let pos = uv * size;
    let t1 = floor(pos - 0.5) + 0.5;
    let f = pos - t1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    // the two middle taps become one bilinear tap between them
    let w12 = w1 + w2;
    let p0 = (t1 - 1.0) / size;
    let p3 = (t1 + 2.0) / size;
    let p12 = (t1 + w2 / w12) / size;
    var c = tap(vec2<f32>(p0.x, p0.y)) * w0.x * w0.y;
    c = c + tap(vec2<f32>(p12.x, p0.y)) * w12.x * w0.y;
    c = c + tap(vec2<f32>(p3.x, p0.y)) * w3.x * w0.y;
    c = c + tap(vec2<f32>(p0.x, p12.y)) * w0.x * w12.y;
    c = c + tap(vec2<f32>(p12.x, p12.y)) * w12.x * w12.y;
    c = c + tap(vec2<f32>(p3.x, p12.y)) * w3.x * w12.y;
    c = c + tap(vec2<f32>(p0.x, p3.y)) * w0.x * w3.y;
    c = c + tap(vec2<f32>(p12.x, p3.y)) * w12.x * w3.y;
    c = c + tap(vec2<f32>(p3.x, p3.y)) * w3.x * w3.y;
    return c;
}

// FXAA as the Enhanced post pass does it (after Lottes' FXAA 3.11), on a picture with
// linear values: the luma is taken of their square roots, near enough to the gamma curve.
fn luma(c: vec3<f32>) -> f32 {
    return dot(sqrt(max(c, vec3<f32>(0.0))), vec3<f32>(0.299, 0.587, 0.114));
}

fn lum_at(uv: vec2<f32>) -> f32 {
    return luma(tap(uv));
}

fn fxaa(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    let rgbm = tap(uv);
    let m = luma(rgbm);
    let n = lum_at(uv + vec2<f32>(0.0, -texel.y));
    let s = lum_at(uv + vec2<f32>(0.0, texel.y));
    let e = lum_at(uv + vec2<f32>(texel.x, 0.0));
    let w = lum_at(uv + vec2<f32>(-texel.x, 0.0));
    let hi = max(m, max(max(n, s), max(e, w)));
    let lo = min(m, min(min(n, s), min(e, w)));
    let range = hi - lo;
    if (range < max(0.0312, hi * 0.125)) {
        return rgbm;
    }
    let nw = lum_at(uv + vec2<f32>(-texel.x, -texel.y));
    let ne = lum_at(uv + vec2<f32>(texel.x, -texel.y));
    let sw = lum_at(uv + vec2<f32>(-texel.x, texel.y));
    let se = lum_at(uv + vec2<f32>(texel.x, texel.y));
    let avg = (2.0 * (n + s + e + w) + nw + ne + sw + se) / 12.0;
    let sub = smoothstep(0.0, 1.0, clamp(abs(avg - m) / range, 0.0, 1.0));
    let blend_sub = sub * sub * 0.75;
    let horz = abs(nw + sw - 2.0 * w) + abs(n + s - 2.0 * m) * 2.0 + abs(ne + se - 2.0 * e);
    let vert = abs(nw + ne - 2.0 * n) + abs(w + e - 2.0 * m) * 2.0 + abs(sw + se - 2.0 * s);
    let is_h = horz >= vert;
    var stp = select(texel.x, texel.y, is_h);
    let l1 = select(w, n, is_h);
    let l2 = select(e, s, is_h);
    let g1 = abs(l1 - m);
    let g2 = abs(l2 - m);
    let neg = g1 >= g2;
    let grad = max(g1, g2) * 0.25;
    let edge_l = 0.5 * (m + select(l2, l1, neg));
    if (neg) {
        stp = -stp;
    }
    var cuv = uv;
    if (is_h) {
        cuv.y = cuv.y + stp * 0.5;
    } else {
        cuv.x = cuv.x + stp * 0.5;
    }
    let dir = select(vec2<f32>(0.0, texel.y), vec2<f32>(texel.x, 0.0), is_h);
    let steps = array<f32, 12>(1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);
    var p_uv = cuv + dir;
    var n_uv = cuv - dir;
    var p_end = lum_at(p_uv) - edge_l;
    var n_end = lum_at(n_uv) - edge_l;
    var p_done = abs(p_end) >= grad;
    var n_done = abs(n_end) >= grad;
    for (var i = 1; i < 12; i = i + 1) {
        if (p_done && n_done) {
            break;
        }
        if (!p_done) {
            p_uv = p_uv + dir * steps[i];
            p_end = lum_at(p_uv) - edge_l;
            p_done = abs(p_end) >= grad;
        }
        if (!n_done) {
            n_uv = n_uv - dir * steps[i];
            n_end = lum_at(n_uv) - edge_l;
            n_done = abs(n_end) >= grad;
        }
    }
    var dp: f32;
    var dn: f32;
    if (is_h) {
        dp = p_uv.x - uv.x;
        dn = uv.x - n_uv.x;
    } else {
        dp = p_uv.y - uv.y;
        dn = uv.y - n_uv.y;
    }
    let near_neg = dn <= dp;
    let end_l = select(p_end, n_end, near_neg);
    let span = dp + dn;
    var blend_edge = 0.0;
    if ((m - edge_l < 0.0) != (end_l < 0.0)) {
        blend_edge = 0.5 - min(dp, dn) / span;
    }
    let blend = max(blend_edge, blend_sub);
    var fuv = uv;
    if (is_h) {
        fuv.y = fuv.y + blend * stp;
    } else {
        fuv.x = fuv.x + blend * stp;
    }
    return tap(fuv);
}

@fragment
fn fs_copy(in: VsOut) -> @location(0) vec4<f32> {
    // Present classic lighting without metering, grading, or a second AA/upscale pass.
    return textureLoad(t_src, vec2<i32>(in.clip.xy), 0);
}

@fragment
fn fs_panel(in: VsOut) -> @location(0) vec4<f32> {
    // Triple screen: one panel's picture 1:1 at its place in the window (src.z: its left
    // edge in window pixels).
    return textureLoad(t_src, vec2<i32>(in.clip.xy) - vec2<i32>(i32(p.src.z), 0), 0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let size = p.src.xy;
    if (p.src.w > 0.5) {
        return vec4<f32>(fxaa(in.uv, 1.0 / size), 1.0);
    }
    let px = 1.0 / size;
    // the four texels around the point bound the filtered value (no ringing)
    let base = (floor(in.uv * size - 0.5) + 0.5) * px;
    let a = tap(base);
    let b = tap(base + vec2<f32>(px.x, 0.0));
    let c = tap(base + vec2<f32>(0.0, px.y));
    let d = tap(base + px);
    let lo = min(min(a, b), min(c, d));
    let hi = max(max(a, b), max(c, d));
    var col = clamp(catmull_rom(in.uv, size), lo, hi);
    // contrast-adaptive sharpening on the source's texel grid
    if (p.src.z > 0.0) {
        let n = tap(in.uv - vec2<f32>(0.0, px.y));
        let s = tap(in.uv + vec2<f32>(0.0, px.y));
        let w = tap(in.uv - vec2<f32>(px.x, 0.0));
        let e = tap(in.uv + vec2<f32>(px.x, 0.0));
        let mn = min(col, min(min(n, s), min(w, e)));
        let mx = max(col, max(max(n, s), max(w, e)));
        let amp = sqrt(clamp(min(mn, vec3<f32>(1.0) - mx) / max(mx, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0)));
        let peak = -1.0 / mix(8.0, 5.0, p.src.z);
        let wgt = amp * peak;
        col = clamp((col + (n + s + w + e) * wgt) / (vec3<f32>(1.0) + 4.0 * wgt), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec4<f32>(col, 1.0);
}

// ---------------------------------------------------------------------------------------
// FSR 1 (AMD FidelityFX Super Resolution 1.0, MIT licence): EASU upscales the render-scale
// picture to the window's size into an intermediate target, RCAS sharpens that on its way
// to the window (`fs_easu`, `fs_rcas`; used instead of `fs_main` when the Upscaler setting
// says FSR 1). Both expect perceptual values, so the picture's linear ones are taken
// through a gamma 2.0 curve (a square root) and back. Ported to WGSL from ffx_fsr1.h
// (FsrEasuF, FsrRcasF); the `p.src.z` sharpening (0..0.8) sets RCAS's strength.
// ---------------------------------------------------------------------------------------
fn gather_r(uv: vec2<f32>) -> vec4<f32> {
    return sqrt(max(textureGather(0, t_src, s_src, uv), vec4<f32>(0.0)));
}

fn gather_g(uv: vec2<f32>) -> vec4<f32> {
    return sqrt(max(textureGather(1, t_src, s_src, uv), vec4<f32>(0.0)));
}

fn gather_b(uv: vec2<f32>) -> vec4<f32> {
    return sqrt(max(textureGather(2, t_src, s_src, uv), vec4<f32>(0.0)));
}

// accumulate the gradient direction and length of one of the four bilinear quadrants
fn easu_set(dir: ptr<function, vec2<f32>>, len: ptr<function, f32>, w: f32, la: f32, lb: f32, lc: f32, ld: f32, le: f32) {
    let len_x0 = max(abs(ld - lc), abs(lc - lb));
    let rcp_x = 1.0 / max(len_x0, 1e-8);
    let dir_x = ld - lb;
    (*dir).x = (*dir).x + dir_x * w;
    var len_x = clamp(abs(dir_x) * rcp_x, 0.0, 1.0);
    len_x = len_x * len_x;
    *len = *len + len_x * w;
    let len_y0 = max(abs(le - lc), abs(lc - la));
    let rcp_y = 1.0 / max(len_y0, 1e-8);
    let dir_y = le - la;
    (*dir).y = (*dir).y + dir_y * w;
    var len_y = clamp(abs(dir_y) * rcp_y, 0.0, 1.0);
    len_y = len_y * len_y;
    *len = *len + len_y * w;
}

// one tap of the (lanczos-2-like) kernel, rotated to the edge direction
fn easu_tap(ac: ptr<function, vec3<f32>>, aw: ptr<function, f32>, off: vec2<f32>, dir: vec2<f32>, len: vec2<f32>, lob: f32, clp: f32, c: vec3<f32>) {
    var v = vec2<f32>(off.x * dir.x + off.y * dir.y, off.x * (-dir.y) + off.y * dir.x);
    v = v * len;
    let d2 = min(v.x * v.x + v.y * v.y, clp);
    var wb = 0.4 * d2 - 1.0;
    var wa = lob * d2 - 1.0;
    wb = wb * wb;
    wa = wa * wa;
    wb = 1.5625 * wb - 0.5625;
    let w = wb * wa;
    *ac = *ac + c * w;
    *aw = *aw + w;
}

@fragment
fn fs_easu(in: VsOut) -> @location(0) vec4<f32> {
    let size = p.src.xy;
    let inv = 1.0 / size;
    // the position of 'f' in the picture, and where the output pixel sits inside its texel
    var pp = in.uv * size - 0.5;
    let fp = floor(pp);
    pp = pp - fp;
    // 12-tap kernel, read as four gathers:   b c / e f g h / i j k l / n o
    let p0 = fp * inv + vec2<f32>(inv.x, -inv.y);
    let p1 = p0 + vec2<f32>(-inv.x, 2.0 * inv.y);
    let p2 = p0 + vec2<f32>(inv.x, 2.0 * inv.y);
    let p3 = p0 + vec2<f32>(0.0, 4.0 * inv.y);
    let bczz_r = gather_r(p0);
    let bczz_g = gather_g(p0);
    let bczz_b = gather_b(p0);
    let ijfe_r = gather_r(p1);
    let ijfe_g = gather_g(p1);
    let ijfe_b = gather_b(p1);
    let klhg_r = gather_r(p2);
    let klhg_g = gather_g(p2);
    let klhg_b = gather_b(p2);
    let zzon_r = gather_r(p3);
    let zzon_g = gather_g(p3);
    let zzon_b = gather_b(p3);
    // luma times two
    let bczz_l = bczz_b * 0.5 + (bczz_r * 0.5 + bczz_g);
    let ijfe_l = ijfe_b * 0.5 + (ijfe_r * 0.5 + ijfe_g);
    let klhg_l = klhg_b * 0.5 + (klhg_r * 0.5 + klhg_g);
    let zzon_l = zzon_b * 0.5 + (zzon_r * 0.5 + zzon_g);
    let b_l = bczz_l.x;
    let c_l = bczz_l.y;
    let i_l = ijfe_l.x;
    let j_l = ijfe_l.y;
    let f_l = ijfe_l.z;
    let e_l = ijfe_l.w;
    let k_l = klhg_l.x;
    let l_l = klhg_l.y;
    let h_l = klhg_l.z;
    let g_l = klhg_l.w;
    let o_l = zzon_l.z;
    let n_l = zzon_l.w;
    // edge direction and strength, accumulated over the four bilinear quadrants
    var dir = vec2<f32>(0.0);
    var len = 0.0;
    easu_set(&dir, &len, (1.0 - pp.x) * (1.0 - pp.y), b_l, e_l, f_l, g_l, j_l);
    easu_set(&dir, &len, pp.x * (1.0 - pp.y), c_l, f_l, g_l, h_l, k_l);
    easu_set(&dir, &len, (1.0 - pp.x) * pp.y, f_l, i_l, j_l, k_l, n_l);
    easu_set(&dir, &len, pp.x * pp.y, g_l, j_l, k_l, l_l, o_l);
    // normalise the direction (a flat area gets 1, 0)
    let dir2 = dir * dir;
    var dir_r = dir2.x + dir2.y;
    let zro = dir_r < (1.0 / 32768.0);
    dir_r = inverseSqrt(max(dir_r, 1.0 / 32768.0));
    dir_r = select(dir_r, 1.0, zro);
    dir.x = select(dir.x, 1.0, zro);
    dir = dir * dir_r;
    len = len * 0.5;
    len = len * len;
    // stretch the kernel along an edge, shift the lobe with the amount of edge
    let stretch = (dir.x * dir.x + dir.y * dir.y) / max(max(abs(dir.x), abs(dir.y)), 1e-8);
    let len2 = vec2<f32>(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    let lob = 0.5 + ((0.25 - 0.04) - 0.5) * len;
    let clp = 1.0 / lob;
    // the four nearest texels bound the result (no ringing)
    let f_c = vec3<f32>(ijfe_r.z, ijfe_g.z, ijfe_b.z);
    let g_c = vec3<f32>(klhg_r.w, klhg_g.w, klhg_b.w);
    let j_c = vec3<f32>(ijfe_r.y, ijfe_g.y, ijfe_b.y);
    let k_c = vec3<f32>(klhg_r.x, klhg_g.x, klhg_b.x);
    let min4 = min(min(min(f_c, g_c), j_c), k_c);
    let max4 = max(max(max(f_c, g_c), j_c), k_c);
    var ac = vec3<f32>(0.0);
    var aw = 0.0;
    easu_tap(&ac, &aw, vec2<f32>(0.0, -1.0) - pp, dir, len2, lob, clp, vec3<f32>(bczz_r.x, bczz_g.x, bczz_b.x)); // b
    easu_tap(&ac, &aw, vec2<f32>(1.0, -1.0) - pp, dir, len2, lob, clp, vec3<f32>(bczz_r.y, bczz_g.y, bczz_b.y)); // c
    easu_tap(&ac, &aw, vec2<f32>(-1.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfe_r.x, ijfe_g.x, ijfe_b.x)); // i
    easu_tap(&ac, &aw, vec2<f32>(0.0, 1.0) - pp, dir, len2, lob, clp, j_c); // j
    easu_tap(&ac, &aw, vec2<f32>(0.0, 0.0) - pp, dir, len2, lob, clp, f_c); // f
    easu_tap(&ac, &aw, vec2<f32>(-1.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(ijfe_r.w, ijfe_g.w, ijfe_b.w)); // e
    easu_tap(&ac, &aw, vec2<f32>(1.0, 1.0) - pp, dir, len2, lob, clp, k_c); // k
    easu_tap(&ac, &aw, vec2<f32>(2.0, 1.0) - pp, dir, len2, lob, clp, vec3<f32>(klhg_r.y, klhg_g.y, klhg_b.y)); // l
    easu_tap(&ac, &aw, vec2<f32>(2.0, 0.0) - pp, dir, len2, lob, clp, vec3<f32>(klhg_r.z, klhg_g.z, klhg_b.z)); // h
    easu_tap(&ac, &aw, vec2<f32>(1.0, 0.0) - pp, dir, len2, lob, clp, g_c); // g
    easu_tap(&ac, &aw, vec2<f32>(1.0, 2.0) - pp, dir, len2, lob, clp, vec3<f32>(zzon_r.z, zzon_g.z, zzon_b.z)); // o
    easu_tap(&ac, &aw, vec2<f32>(0.0, 2.0) - pp, dir, len2, lob, clp, vec3<f32>(zzon_r.w, zzon_g.w, zzon_b.w)); // n
    let pix = min(max4, max(min4, ac / max(aw, 1e-8)));
    return vec4<f32>(pix * pix, 1.0);
}

fn rcas_px(sp: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    return sqrt(max(textureLoad(t_src, clamp(sp, vec2<i32>(0), dims - vec2<i32>(1)), 0).rgb, vec3<f32>(0.0)));
}

@fragment
fn fs_rcas(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(t_src));
    let sp = vec2<i32>(in.clip.xy);
    //   b
    // d e f
    //   h
    let b = rcas_px(sp + vec2<i32>(0, -1), dims);
    let d = rcas_px(sp + vec2<i32>(-1, 0), dims);
    let e = rcas_px(sp, dims);
    let f = rcas_px(sp + vec2<i32>(1, 0), dims);
    let h = rcas_px(sp + vec2<i32>(0, 1), dims);
    let mn4 = min(min(min(b, d), f), h);
    let mx4 = max(max(max(b, d), f), h);
    let hit_min = min(mn4, e) / max(4.0 * mx4, vec3<f32>(1e-5));
    let hit_max = (vec3<f32>(1.0) - max(mx4, e)) / min(4.0 * mn4 - vec3<f32>(4.0), vec3<f32>(-1e-5));
    let lobe3 = max(-hit_min, hit_max);
    // sharpness in stops of halving: the smaller the picture, the nearer to full strength
    let stops = mix(1.5, 0.2, clamp(p.src.z / 0.8, 0.0, 1.0));
    let lobe = max(-0.1875, min(max(max(lobe3.x, lobe3.y), lobe3.z), 0.0)) * exp2(-stops);
    let pix = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
    return vec4<f32>(pix * pix, 1.0);
}
