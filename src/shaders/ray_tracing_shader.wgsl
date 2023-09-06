
@group(0)
@binding(0)
var<storage, read> input_buffer: InputBuffer;

@group(0)
@binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

struct InputBuffer {
    spheres: array<Sphere>,
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
    return ray.pos + ray.dir * t;
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
    let result: IntersectionResult = IntersectionResult(false, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0));
    

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
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let aspect_ratio: f32 = 1.0;
    let image_width: u32 = 400u;
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

    let ray_direction = pixel_center - camera_center;
    let ray = Ray(camera_center, ray_direction);

    let pixel_color = ray_color(ray);

    textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(pixel_color, 1.0));
}