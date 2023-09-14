#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Ground {
    pub center: [f32; 3],
    pub width: f32,
    pub height: f32,
    _padding: [u32; 3],
    pub material: u32,

}

impl Ground {
    pub fn new(center: [f32; 3], width: f32, height: f32, material: u32) -> Self {
        Self {
            center, 
            width, 
            height, 
            _padding: [0; 3], 
            material,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub material: u32,
    _padding: [u32; 3],
}

impl Sphere {
    pub fn new(center: [f32; 3], radius: f32, material: u32) -> Self {
        Self {
            center, 
            radius, 
            material, 
            _padding: [0; 3],
        }
    }
}

pub const LAMBERTIAN: u32 = 0;
pub const METAL: u32 = 1;
pub const DIELECTRIC: u32 = 2;
pub const EMISSIVE: u32 = 3;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub mat_type: u32,
    pub intensity: f32,
    _padding: [u32; 2], 
    pub color: [f32; 3],
    _padding_2: u32,
}

impl Material {
    pub fn new(mat_type: u32, intensity: f32, color: [f32; 3]) -> Self {
        Self {
            mat_type, 
            intensity, 
            _padding: [0; 2], 
            color, 
            _padding_2: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    pub pos: [f32; 3],
    _padding: u32,
    pub look_at: [f32; 3],
    _padding_2: u32,
    pub up: [f32; 3],
    _padding_3: u32,
    pub size: [u32; 2],
    pub fov: f32,
    _padding_4: u32,
}

impl Camera {
    pub fn new(pos: [f32; 3], look_at: [f32; 3], up: [f32; 3], size: [u32; 2], fov: f32) -> Self {
        Self {
            pos, 
            look_at, 
            up, 
            size, 
            fov, 
            _padding: 0, 
            _padding_2: 0, 
            _padding_3: 0,
            _padding_4: 0, 
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle {
    pub a: [f32; 3],
    _padding: u32,
    pub b: [f32; 3],
    _padding_2: u32,
    pub c: [f32; 3],
    // _padding_4: [u32; 2],
    pub material: u32,
    
}

impl Triangle {
    pub fn new(a: [f32; 3], b: [f32; 3], c: [f32; 3], material: u32) -> Self {
        Self {
            a, 
            _padding: 0, 
            b, 
            _padding_2: 0, 
            c, 
            material, 
        }
    }
}