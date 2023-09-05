struct ComputeInput {
    test: u8
}

struct Ray {
    pos: vec3<f32>,
    dir: vec3<f32>,  
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
}

const test_sphere: Sphere = Sphere(vec3<f32>(0.0, 0.0, -1.0), 0.5);

fn ray_pos_at(t: f32, ray: Ray) -> vec3<f32> {
    ray.pos + ray.dir * t
}

fn unit_vector(v: vec3<f32>) -> vec3<f32> {
    return v / length(v);
}

fn ray_color(ray: Ray) -> vec3<f32> {
    if (hit_sphere(test_sphere, ray)) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }

    let unit_direction = unit_vector(ray.dir);
    let t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * vec3<f32>(1.0, 1.0, 1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
}

struct IntersectionResult {
    hit: bool,
    point: vec3<f32>,
    normal: vec3<f32>,
}

fn sphere_intersection(sphere: Sphere, ray: Ray) -> IntersectionResult {
    let result: IntersectionResult;
    

    
    return result;
}

fn hit_sphere(sphere: Sphere, ray: Ray) -> bool {
    let oc = ray.pos - sphere.center;
    let a = dot(ray.dir, ray.dir);
    let b = 2.0 * dot(oc, ray.dir);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - 4.0 * a * c;
    return discriminant > 0.0;
}


@compute
fn main(input: ComputeInput) {
    let ray: Ray;


}