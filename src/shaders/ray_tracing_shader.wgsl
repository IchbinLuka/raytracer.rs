// ---------------------- Constants ---------------------- //

const SAMPLE_COUNT: u32 = 10u;
const BOUNCE_COUNT: u32 = 5u;
const PI: f32 = 3.1415926535897932385;

// ---------------------- Bindings ------------------------ //

@group(0)
@binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0)
@binding(5)
var input_texture: texture_2d<f32>;

@group(0)
@binding(1)
var<storage, read> grounds: array<Ground>;

@group(0)
@binding(2)
var<storage, read> spheres: array<Sphere>;

@group(0)
@binding(3)
var<storage, read> triangles: array<Triangle>;

@group(0)
@binding(4)
var<storage, read> materials: array<Material>;

@group(1)
@binding(0)
var<uniform> camera: Camera;

@group(1)
@binding(1)
var<uniform> iter_info: IterationInfo;

alias MatPtr = u32;

struct IterationInfo {
    iter_count: u32,
    counter: u32,
}


struct InputBuffer {
    spheres: array<Sphere>,
}


struct Camera {
    pos: vec3<f32>,
    look_at: vec3<f32>,
    up: vec3<f32>,
    size: vec2<u32>,
    fov: f32,
}

struct IntersectionResult {
    hit: bool,
    record: HitRecord,
}

struct HitRecord {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    front_face: bool,
    material: MatPtr,
}

// ---------------------- Utility ------------------------ //

fn get_material(mat_ptr: MatPtr) -> Material {
    return materials[mat_ptr];
}

fn hit_record_set_face_normal(record: ptr<function, HitRecord>, ray: Ray, outward_normal: vec3<f32>) {
    let front_face = dot(ray.dir, outward_normal) < 0.0;
    let normal = select(-outward_normal, outward_normal, front_face);
    (*record).front_face = front_face;
    (*record).normal = normal;
}

fn unit_vector(v: vec3<f32>) -> vec3<f32> {
    return v / length(v);
}


fn world_hit(ray: Ray, interval: Interval) -> IntersectionResult {
    var result: IntersectionResult;
    result.hit = false;
    result.record.t = interval.max;
    var in: Interval = interval;
    var t_max: f32 = interval.max;

    for (var i: u32 = 0u; i < arrayLength(&spheres); i++) {
        let sphere = spheres[i];
        let intersection = sphere_intersection(sphere, ray, Interval(in.min, result.record.t));
        if intersection.hit { result = intersection; }
    }


    for (var j: u32 = 0u; j < arrayLength(&grounds); j++) {
        let ground = grounds[j];
        let intersection = ground_intersection(ground, ray, Interval(in.min, result.record.t));
      
        if intersection.hit { result = intersection; }
    }

    for (var k: u32 = 0u; k < arrayLength(&triangles); k++) {
        let triangle = triangles[k];

        let intersection = triangle_intersection(triangle, ray, Interval(in.min, result.record.t));
        if intersection.hit { result = intersection; }
    }

    return result;
}

fn vec3_near_zero(v: vec3<f32>) -> bool {
    let s = 1e-8;
    return (abs(v.x) < s) && (abs(v.y) < s) && (abs(v.z) < s);
}

// ---------------------- Ray ----------------------------- //

struct Ray {
    pos: vec3<f32>,
    dir: vec3<f32>,  
    color: vec3<f32>,
}

fn ray_pos_at(t: f32, ray: Ray) -> vec3<f32> {
    return ray.pos + ray.dir * t;
}

fn ray_color(ray: Ray) -> vec3<f32> {

    var color: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    var current_ray = ray;

    for (var i: u32 = 0u; i < BOUNCE_COUNT; i++) {
        let intersection = world_hit(current_ray, Interval(0.0001, 1000000.0));
        if !intersection.hit { break; }
        
        let direction = intersection.record.normal + random_unit_vec3();
        current_ray = material_scatter(current_ray, intersection.record);
        if all(current_ray.dir == vec3<f32>(0.0, 0.0, 0.0)) {
            // Light did not scatter
            return current_ray.color;
        }
    }

    let unit_direction = unit_vector(current_ray.dir);
    let a = 0.5 * (unit_direction.y + 1.0);
    // return vec3<f32>(0.01, 0.01, 0.01);
    return ((1.0 - a) * vec3<f32>(1.0, 1.0, 1.0) + a * vec3<f32>(0.5, 0.7, 1.0)) * current_ray.color;
}

fn reflect(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn refract(uv: vec3<f32>, n: vec3<f32>, etai_over_etat: f32) -> vec3<f32> {
    let cos_theta = min(dot(-uv, n), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return r_out_parallel + r_out_perp;
}

fn schlick_reflectance(cosine: f32, ref_idx: f32) -> f32 {
    let r0_ = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0 = r0_ * r0_;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// ------------------------------------- Material ------------------------------------------- //

const LAMBERTIAN = 0u;
const METAL = 1u;
const DIELECTRIC = 2u;

struct Material {
    mat_type: u32,
    intensity: f32,
    color: vec3<f32>,
}

fn material_scatter(ray: Ray, hit_record: HitRecord) -> Ray {
    let reflected_direction = reflect(unit_vector(ray.dir), hit_record.normal);

    var scatter_direction: vec3<f32>;

    let material = get_material(hit_record.material);

    switch material.mat_type {
        case 0u: { // Lambertian
            scatter_direction = hit_record.normal + random_unit_vec3();

            // Avoid zero vector
            if vec3_near_zero(scatter_direction) {
                scatter_direction = hit_record.normal;
            }
            break;
        }
        case 1u: { // Metal
            scatter_direction = reflected_direction + material.intensity * random_unit_vec3();
            break;
        }
        case 2u: { // Dielectric
            let refraction_ratio = select(material.intensity, 1.0 / material.intensity, hit_record.front_face);

            let unit_direction = unit_vector(ray.dir);
            
            let cos_theta = min(dot(-unit_direction, hit_record.normal), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            let cannot_refract = refraction_ratio * sin_theta > 1.0;

            if cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > hybrid_taus() {
                // Ray cannot be refracted, since Snells law cannot be satisfied
                scatter_direction = reflect(unit_direction, hit_record.normal);
                break;
            }
            
            scatter_direction = refract(unit_direction, hit_record.normal, refraction_ratio);
            break;
        }
        case 3u: { // Light
            scatter_direction = vec3<f32>(0.0, 0.0, 0.0);
            return Ray(
                hit_record.point, 
                scatter_direction, 
                ray.color * material.color * material.intensity
            );
            // break;
        }
        default: {
            break; // TODO: Error
        }
    }

    let scattered = Ray(
        hit_record.point, 
        scatter_direction, 
        ray.color * material.color
    );
    return scattered;
}

// ------------------------------------- Shapes ------------------------------------------- //

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material: MatPtr,
}

fn sphere_intersection(sphere: Sphere, ray: Ray, interval: Interval) -> IntersectionResult {
    var result: IntersectionResult;

    let oc = ray.pos - sphere.center;

    let a = dot(ray.dir, ray.dir);
    let h = dot(oc, ray.dir);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = h * h - a * c;

    if discriminant < 0.0 {
        result.hit = false;
        return result;
    }

    let sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    var root = (-h - sqrtd) / a;
    if root <= interval.min || interval.max <= root {
        root = (-h + sqrtd) / a;
        if root <= interval.min || interval.max <= root {
            result.hit = false;
            return result;
        }
    }

    var record: HitRecord;

    result.hit = true;
    record.t = root;
    record.point = ray_pos_at(root, ray);
    let outward_normal = (record.point - sphere.center) / sphere.radius;
    hit_record_set_face_normal(&record, ray, outward_normal);
    record.material = sphere.material;

    result.record = record;

    return result;
}



struct Ground {
    center: vec3<f32>,
    width: f32,
    height: f32,
    material: MatPtr,
}


fn ground_intersection(ground: Ground, ray: Ray, interval: Interval) -> IntersectionResult {
    var result: IntersectionResult;

    let t = (ground.center.y - ray.pos.y) / ray.dir.y;

    let pos = ray_pos_at(t, ray);

    let x = pos.x - ground.center.x;
    let z = pos.z - ground.center.z;

    let intersected = interval_contains(Interval(-ground.width / 2.0, ground.width / 2.0), x) &&
                      interval_contains(Interval(-ground.height / 2.0, ground.height / 2.0), z);

    if t < interval.min || interval.max < t || !intersected {
        result.hit = false;
        return result;
    }

    var record: HitRecord;
    result.record = record;

    result.hit = true;
    record.t = t;
    record.point = ray_pos_at(t, ray);
    let outward_normal = vec3<f32>(0.0, 1.0, 0.0);
    hit_record_set_face_normal(&record, ray, outward_normal);
    record.material = ground.material;


    result.record = record;
    return result;
}

struct Triangle {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    material: MatPtr,
}

const EPSILON = 0.0000001;

// Calculates the intersection between a triangle and a ray using the Möller–Trumbore algorithm:
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm#C++_implementation
// This is a translation of the C++ code found on the Wikipedia page.
fn triangle_intersection(tr: Triangle, ray: Ray, interval: Interval) -> IntersectionResult {
    var result: IntersectionResult;

    if vec3_near_zero(tr.a - tr.b) || vec3_near_zero(tr.a - tr.c) || vec3_near_zero(tr.b - tr.c) {
        result.hit = false;
        return result;
    }

    // Möller–Trumbore intersection algorithm
    let edge1 = tr.b - tr.a;
    let edge2 = tr.c - tr.a;
    let h = cross(ray.dir, edge2);
    let a = dot(edge1, h);

    if a > -EPSILON && a < EPSILON {
        // This ray is parallel to this triangle.
        result.hit = false;
        return result;
    }

    let f = 1.0 / a;
    let s = ray.pos - tr.a;
    let u = f * dot(s, h);

    if u < 0.0 || u > 1.0 {
        result.hit = false;
        return result;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray.dir, q);

    if v < 0.0 || u + v > 1.0 {
        result.hit = false;
        return result;
    }

    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = f * dot(edge2, q);

    if t > EPSILON && interval_contains(interval, t) {
        // Ray intersection
        var record: HitRecord;
        result.hit = true;
        record.t = t;
        record.point = ray_pos_at(t, ray);
        let outward_normal = cross(edge1, edge2);
        hit_record_set_face_normal(&record, ray, outward_normal);
        record.material = tr.material;
        result.record = record;
        return result;
    } else {
        // This means that there is a line intersection but not a ray intersection.
        result.hit = false;
        return result;
    }
}

// ------------------------------- Interval ------------------------------ //

struct Interval {
    min: f32,
    max: f32,
}

fn interval_contains(interval: Interval, x: f32) -> bool {
    return interval.min <= x && x <= interval.max;
}

fn interval_clamp(interval: Interval, x: f32) -> f32 {
    return clamp(x, interval.min, interval.max);
}

// --------------------------------------------------------- //



//--------------------------------- Random number generator ----------------------------------------------------//
// https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf //
//--------------------------------------------------------------------------------------------------------------//

var<private> z1: u32 = 123456789u;
var<private> z2: u32 = 362436069u;
var<private> z3: u32 = 521288629u;
var<private> z4: u32 = 88675123u;

fn hybrid_taus() -> f32 {
    return 2.3283064365387e-10 * f32(
        taus_step(&z1, 13u, 19u, 12u, 4294967294u) ^
        taus_step(&z2, 2u, 25u, 4u, 4294967288u) ^
        taus_step(&z3, 3u, 11u, 17u, 4294967280u) ^
        lcg_step(&z4, 1664525u, 1013904223u)
    );
}

fn init_hybrid_taus(id: vec3<u32>) {
    z1 = seed(id.x);
    z2 = seed(id.y);
    z3 = seed(id.z);
    z4 = seed(id.x + id.y + id.z);
}


fn taus_step(z: ptr<private, u32>, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    let b = (((*z << s1) ^ *z) >> s2);
    *z = (((*z & m) << s3) ^ b);
    return *z;
}

fn lcg_step(z: ptr<private, u32>, a: u32, c: u32) -> u32 {
    *z = (a * *z + c);
    return *z;
}


fn seed(id: u32) -> u32 {
    return id * 1099087573u;
}

fn random_vec3() -> vec3<f32> {
    return vec3<f32>(hybrid_taus(), hybrid_taus(), hybrid_taus());
}

fn random_vec3_in_unit_sphere() -> vec3<f32> {
    loop {
        let p = random_vec3() * 2.0 - vec3<f32>(1.0, 1.0, 1.0);
        if dot(p, p) >= 1.0 {
            continue;
        }
        return p;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn random_unit_vec3() -> vec3<f32> {
    return unit_vector(random_vec3_in_unit_sphere());
}

fn random_vec3_on_hemisphere(normal: vec3<f32>) -> vec3<f32> {
    let on_unit_sphere = random_unit_vec3();
    if dot(on_unit_sphere, normal) > 0.0 {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

// ------------------------------------------------------------------------------ //


@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    init_hybrid_taus(gid + iter_info.counter);

    let fov: f32 = camera.fov;

    let image_width: u32 = camera.size.x;
    let image_height: u32 = camera.size.y;
    let aspect_ratio: f32 = f32(image_width) / f32(image_height);

    let camera_center = camera.pos;
    
    let focal_length: f32 = length(camera_center - camera.look_at);
    let theta = fov / 360.0 * PI * 2.0;
    let viewport_height = 2.0 * tan(theta / 2.0) * focal_length;
    
    let viewport_width: f32 = aspect_ratio * viewport_height;

    let w = unit_vector(camera_center - camera.look_at);
    let u = unit_vector(cross(camera.up, w));
    let v = cross(w, u);

    let viewport_u = viewport_width * u;
    let viewport_v = viewport_height * -v;

    let pixel_delta_u = viewport_u / f32(image_width);
    let pixel_delta_v = viewport_v / f32(image_height);

    let viewport_upper_left = camera_center - (focal_length * w) - viewport_u / 2.0 - viewport_v / 2.0;
    let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    let pixel_center: vec3<f32> = pixel00_loc + (f32(gid.x) * pixel_delta_u) + (f32(gid.y) * pixel_delta_v);

    var pixel_color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < SAMPLE_COUNT; k++) {
        let px = -0.5 * hybrid_taus();
        let py = -0.5 * hybrid_taus();

        let pixel_sample = pixel_center + (px * pixel_delta_u) + (py * pixel_delta_v);

        let ray = Ray(camera_center, pixel_sample - camera_center, vec3<f32>(1.0, 1.0, 1.0));

        pixel_color += ray_color(ray) / f32(SAMPLE_COUNT);
    }

    pixel_color = clamp(pixel_color, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));



    let initial = textureLoad(input_texture, vec2<i32>(gid.xy), 0);

    textureStore(
        output_texture, 
        vec2<i32>(i32(gid.x), i32(gid.y)), 
        vec4<f32>(initial.xyz + (pixel_color / f32(iter_info.iter_count)), 1.0)
    );
}