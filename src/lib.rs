#![feature(slice_pattern)]

use core::slice::SlicePattern;

use collada::{Triangles, PrimitiveElement};
use wgpu::{Device, util::DeviceExt};
use winit::dpi::PhysicalSize;
mod types;

use types::*;

fn padded_bytes_per_row(width: u32) -> usize {
    let bytes_per_row = width as usize * 4;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}

fn compute_work_group_count(
    (width, height): (u32, u32),
    (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;

    (x, y)
}


const IMAGE_SIZE: PhysicalSize<u32> = PhysicalSize::new(1200, 1200);


struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    camera_bind_group: wgpu::BindGroup,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    output_texture: wgpu::Texture,
}


impl State {
    async fn new(
        spheres: &[Sphere], 
        grounds: &[Ground], 
        triangles: &[Triangle], 
        materials: &[Material], 
        camera: Camera
    ) -> Self {
        let size = IMAGE_SIZE;

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        
        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        // let surface = unsafe { instance.create_surface(&window) }.unwrap();


        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ).await.unwrap_or(
            instance
                .enumerate_adapters(wgpu::Backends::all())
                .next(
                    // Check if this adapter supports our surface
                    // adapter.is_surface_supported(&surface)
                )
                .unwrap()
        );

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None, // Trace path
        ).await.unwrap();


        // Create a buffer to hold the results
        let buffer_size = 200usize;
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_bytes_per_row(size.width) as u64 * size.height as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_only_buffer = wgpu::BindGroupLayoutEntry {
            binding: 1, 
            visibility: wgpu::ShaderStages::COMPUTE, 
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage { read_only: true }
            },
            count: None
        };

        // Create a bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, 
                    visibility: wgpu::ShaderStages::COMPUTE, 
                    ty: wgpu::BindingType::StorageTexture { 
                        access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::Rgba8Unorm, 
                        view_dimension: wgpu::TextureViewDimension::D2
                    }, 
                    count: None
                }, 
                wgpu::BindGroupLayoutEntry {
                    binding: 1, 
                    ..read_only_buffer
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, 
                    ..read_only_buffer
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, 
                    ..read_only_buffer
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, 
                    ..read_only_buffer
                },
            ],
        });

        

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: 
                wgpu::TextureUsages::COPY_SRC | 
                wgpu::TextureUsages::STORAGE_BINDING | 
                wgpu::TextureUsages::TEXTURE_BINDING | 
                wgpu::TextureUsages::COPY_DST | 
                wgpu::TextureUsages::RENDER_ATTACHMENT
        });

        // Create a compute shader module
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/ray_tracing_shader.wgsl"));

        let camera_bindgroup_layout = Self::create_image_size_bind_group_layout(&device);


        // Create compute pipeline layout
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout, &camera_bindgroup_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let buffer_init_descriptor = wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0; 1]),
            usage: wgpu::BufferUsages::STORAGE,
        };

        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            contents: bytemuck::cast_slice(materials),
            ..buffer_init_descriptor
        });

        let ground_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(grounds),
            ..buffer_init_descriptor
        });

        let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(spheres),
            ..buffer_init_descriptor
        });

        let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(triangles),
            ..buffer_init_descriptor
        });
        

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0, 
                    resource: wgpu::BindingResource::TextureView(&output_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                }, 
                wgpu::BindGroupEntry {
                    binding: 1, 
                    resource: ground_buffer.as_entire_binding(),
                }, 
                wgpu::BindGroupEntry {
                    binding: 2, 
                    resource: sphere_buffer.as_entire_binding(),
                }, 
                wgpu::BindGroupEntry {
                    binding: 3, 
                    resource: triangle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4, 
                    resource: material_buffer.as_entire_binding(),
                }
            ],
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                camera
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &camera_bindgroup_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0, 
                    resource: camera_buffer.as_entire_binding()
                }
            ],
        });

        Self {
            // surface,
            device,
            queue,
            compute_pipeline,
            bind_group,
            input_buffer,
            output_buffer,
            // window,
            // render_pipeline,
            output_texture,
            camera_bind_group,
        }
    }

    fn create_image_size_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
    

        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, 
                    visibility: wgpu::ShaderStages::COMPUTE, 
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform
                    },
                    count: None
                }
            ],
        });
    }


    fn render(&mut self) {

        let size = IMAGE_SIZE;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            // Create a pipeline
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_bind_group(1, &self.camera_bind_group, &[]);


            let (dispatch_with, dispatch_height) =
                compute_work_group_count((size.width, size.height), (16, 16));

            cpass.dispatch_workgroups(
                dispatch_with, dispatch_height, 1 
            );
        }

        let padded_bytes_per_row = padded_bytes_per_row(size.width);
        let unpadded_bytes_per_row = size.width as usize * 4;
        

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row as u32),
                    rows_per_image: Some(size.height),
                },
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );

        // Submit the command encoder
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read the results from the buffer
        let buffer_slice = self.output_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |e| {
            if let Err(error) = e {
                println!("Error mapping buffer: {:?}", error);
            }
        });
        
        self.device.poll(wgpu::Maintain::Wait);

        let binding = buffer_slice.get_mapped_range();
        let data = binding.as_slice();
        

        let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * size.height as usize];
        for (padded, pixels) in data
            .chunks_exact(padded_bytes_per_row)
            .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row))
        {
            pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
        }

        if let Some(output_image) = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(size.width, size.height, &pixels[..]) {
            output_image.save("output.png").unwrap();
        }

    }

}

fn vertex_to_slice(vertex: &collada::Vertex) -> [f32; 3] {
    [vertex.x as f32, vertex.y as f32, vertex.z as f32]
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");;
        } else {
            env_logger::init();
        }
    }

    let suzanne = collada::document::ColladaDocument::from_str(include_str!("assets/suzanne.dae")).unwrap();

    let objects = suzanne.get_obj_set().unwrap().objects;

    let mut triangle_vec: Vec<Triangle> = Vec::new();

    for object in objects {
        for e in object.geometry.iter().flat_map(|e| &e.mesh) {
            if let PrimitiveElement::Triangles(triangles) = e {
                for (a, b, c) in &triangles.vertices {
                    triangle_vec.push(
                        Triangle::new(
                            vertex_to_slice(&object.vertices[*a]), 
                            vertex_to_slice(&object.vertices[*b]), 
                            vertex_to_slice(&object.vertices[*c]), 
                            1u32
                        )
                    );
                }
            }
        }
    }

    let materials = &[
        Material::new(METAL, 0.7, [0.8, 0.5, 0.25]), 
        Material::new(METAL, 0.2, [0.7, 0.7, 0.7]), 
        Material::new(DIELECTRIC, 1.5, [1.0, 0.9, 1.0]), 
        Material::new(LAMBERTIAN, 0.1, [0.1, 0.8, 0.1]), 
        Material::new(EMISSIVE, 3.0, [1.0, 1.0, 1.0]), 
    ];

    let spheres = &[
        // Sphere::new([0.5, 0.0, -1.7], 0.4, 1), 
        Sphere::new([0.0, -1.8, -1.0], 0.2, 2),
        // Sphere::new([-0.5, 0.0, -1.7], 0.4, 3),
        Sphere::new([0.0, -0.8, -1.0], 0.2, 4),
    ];

    let grounds = &[
        Ground::new([0.0, -2.0, 0.0], 100.0, 100.0, 0),
    ];

    let triangles = triangle_vec.as_slice();

    let camera = Camera::new(
        [0.0, 0.5, -5.0], 
        [0.0, 0.0, -1.5], 
        [0.0, 1.0, 0.0], 
        [IMAGE_SIZE.width, IMAGE_SIZE.height], 
        90.0
    );

    let mut state = State::new(spheres, grounds, triangles, materials, camera).await;
    state.render();
}