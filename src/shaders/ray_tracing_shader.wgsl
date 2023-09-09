// ---------------------- Constants ---------------------- //

const PI: f32 = 3.1415926535897932385;

const SAMPLE_COUNT: u32 = 800u;
const BOUNCE_COUNT: u32 = 10u;

// ---------------------- Bindings ------------------------ //

@group(0)
@binding(0)
var<storage, read> input_buffer: InputBuffer;

@group(0)
@binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0)
@binding(2)
var<storage, read> grounds: array<Ground>;

@group(0)
@binding(3)
var<storage, read> spheres: array<Sphere>;

struct InputBuffer {
    spheres: array<Sphere>,
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
    material: Material,
}

// ---------------------- Utility ------------------------ //

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

    // let test = array<IntersectionResult, total_item_count>();

    for (var i: u32 = 0u; i < arrayLength(&spheres); i++) {
        let sphere = spheres[i];
        let intersection = sphere_intersection(sphere, ray, Interval(in.min, t_max));
        if intersection.hit {
            result.hit = true;
            result.record = intersection.record;
            t_max = intersection.record.t;
        }
    }


    for (var j: u32 = 0u; j < arrayLength(&grounds); j++) {
        let ground = grounds[j];
        let intersection = ground_intersection(ground, ray, Interval(in.min, t_max));
        
        if intersection.hit {
            result.hit = true;
            result.record = intersection.record;
            t_max = intersection.record.t;
        }
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
        current_ray = material_scatter(intersection.record.material, current_ray, intersection.record);
    }

    let unit_direction = unit_vector(current_ray.dir);
    let a = 0.5 * (unit_direction.y + 1.0);
    return ((1.0 - a) * vec3<f32>(1.0, 1.0, 1.0) + a * vec3<f32>(0.5, 0.7, 1.0)) * current_ray.color;
}

fn reflect(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

// ------------------------------------- Material ------------------------------------------- //

struct Material {
    glossiness: f32,
    color: vec3<f32>,
}

fn material_scatter(material: Material, ray: Ray, hit_record: HitRecord) -> Ray {
    let reflected_direction = reflect(unit_vector(ray.dir), hit_record.normal);
    var diffuse_direction = hit_record.normal + random_unit_vec3();
    
    // Avoid zero vector
    if (vec3_near_zero(diffuse_direction)) {
        diffuse_direction = hit_record.normal;
    }

    let scattered = Ray(
        hit_record.point, 
        material.glossiness * reflected_direction + (1.0 - material.glossiness) * diffuse_direction, 
        ray.color * material.color
    );
    return scattered;
}

// ------------------------------------- Shapes ------------------------------------------- //

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material: Material,
}

fn sphere_intersection(sphere: Sphere, ray: Ray, interval: Interval) -> IntersectionResult {
    var result: IntersectionResult;

    let oc = ray.pos - sphere.center;

    let a = dot(ray.dir, ray.dir);
    let h = dot(oc, ray.dir);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = h * h - a * c;

    if (discriminant < 0.0) {
        result.hit = false;
        return result;
    }

    let sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    var root = (-h - sqrtd) / a;
    if (root <= interval.min || interval.max <= root) {
        root = (-h + sqrtd) / a;
        if (root <= interval.min || interval.max <= root) {
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
    material: Material,
}


fn ground_intersection(ground: Ground, ray: Ray, interval: Interval) -> IntersectionResult {
    var result: IntersectionResult;

    let t = (ground.center.y - ray.pos.y) / ray.dir.y;

    if (t < interval.min || interval.max < t) {
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
        if (dot(p, p) >= 1.0) {
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
    if (dot(on_unit_sphere, normal) > 0.0) {
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
    init_hybrid_taus(gid);

    let aspect_ratio: f32 = 1.0;
    let image_width: u32 = 1200u;
    let image_height: u32 = u32(f32(image_width) / aspect_ratio);
    
    let camera_center = vec3<f32>(0.0, 0.0, 0.0);
    let viewport_height: f32 = 2.0;
    let viewport_width: f32 = aspect_ratio * viewport_height;
    let focal_length: f32 = 1.0;

    let viewport_u = vec3<f32>(viewport_width, 0.0, 0.0);
    let viewport_v = vec3<f32>(0.0, -viewport_height, 0.0);

    let pixel_delta_u = viewport_u / f32(image_width);
    let pixel_delta_v = viewport_v / f32(image_height);

    let viewport_upper_left = camera_center - vec3<f32>(0.0, 0.0, focal_length) - viewport_u / 2.0 - viewport_v / 2.0;
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

    textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(pixel_color, 1.0));
}