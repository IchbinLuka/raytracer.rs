#![feature(slice_pattern)]
use core::slice::SlicePattern;

use wgpu::{util::DeviceExt, TextureDescriptor, ImageDataLayout, ImageCopyTexture, Extent3d};
use winit::{window::{Window, WindowBuilder}, event_loop::EventLoop, event::Event, dpi::PhysicalSize};


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


const image_size: PhysicalSize<u32> = PhysicalSize::new(400, 400);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Ground {
    center: [f32; 3],
    width: f32,
    height: f32,
    // TODO: Find better way to pad to 32 bytes
    a: i32, 
    b: i32, 
    c: i32
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Sphere {
    center: [f32; 3],
    radius: f32,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    window: Window,
    output_texture: wgpu::Texture,
}


impl State {
    async fn new(window: Window) -> Self {
        let size = image_size;

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
        let surface = unsafe { instance.create_surface(&window) }.unwrap();


        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: true,
            },
        ).await.unwrap_or(
            instance
                .enumerate_adapters(wgpu::Backends::all())
                .find(|adapter| {
                    // Check if this adapter supports our surface
                    adapter.is_surface_supported(&surface)
                })
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


        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())            
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);


        let render_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/render_shader.wgsl"));
        
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let targets = [Some(wgpu::ColorTargetState { // 4.
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "vertex_main", // 1.
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &render_shader,
                entry_point: "fragment_main",
                targets: &targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1, // 2.
                mask: !0, // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        };

        let render_pipeline = device.create_render_pipeline(&pipeline_descriptor);

        // let swap_chain = device.create_(&surface, &swap_chain_desc);


        // Create a buffer to hold the results
        let buffer_size = 200 as usize;
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded_bytes_per_row(size.width) as u64 * size.height as u64 * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create a bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    count: None,
                }, 
                wgpu::BindGroupLayoutEntry {
                    binding: 1, 
                    visibility: wgpu::ShaderStages::COMPUTE, 
                    ty: wgpu::BindingType::StorageTexture { 
                        access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::Rgba8Unorm, 
                        view_dimension: wgpu::TextureViewDimension::D2
                    }, 
                    count: None
                }, 
                wgpu::BindGroupLayoutEntry {
                    binding: 2, 
                    visibility: wgpu::ShaderStages::COMPUTE, 
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    count: None
                }, 
                wgpu::BindGroupLayoutEntry {
                    binding: 3, 
                    visibility: wgpu::ShaderStages::COMPUTE, 
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    count: None
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

        // Create compute pipeline layout
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let ground_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[Ground {
                center: [0.0, -1.0, 0.0],
                width: 100.0,
                height: 100.0,
                a: 0, b: 0, c: 0
            }]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[Sphere {
                center: [0.0, 0.0, -1.0],
                radius: 0.5,
            }]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(input_buffer.as_entire_buffer_binding()),
                }, 
                wgpu::BindGroupEntry {
                    binding: 1, 
                    resource: wgpu::BindingResource::TextureView(&output_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                }, 
                wgpu::BindGroupEntry {
                    binding: 2, 
                    resource: ground_buffer.as_entire_binding(),
                }, 
                wgpu::BindGroupEntry {
                    binding: 3, 
                    resource: sphere_buffer.as_entire_binding(),
                }, 
            ],
        });

        Self {
            surface,
            device,
            queue,
            compute_pipeline,
            bind_group,
            input_buffer,
            output_buffer,
            window,
            render_pipeline,
            output_texture,
        }
    }

    fn render(&mut self) {

        let size = image_size;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            // Create a pipeline
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);


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


        // println!("data: {:?}", pixels);

        /*self.queue.write_texture(
            ImageCopyTexture {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            }, 
            data, 
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * size.width),
                rows_per_image: Some(size.height),
            }, 
            Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            }
        );*/

        return;

        // TODO: Render the output texture to the screen
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            // let view = self.output_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let view = self.surface.get_current_texture().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), // Clear the screen to white
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
        
            // Bind the texture to a shader
            render_pass.set_bind_group(0, &self.bind_group, &[]);
        
            // Draw a quad or a full-screen triangle using the texture
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw(0..3, 0..1); // Adjust the vertices and indices as needed
        }
        
        // Submit the render commands
        self.queue.submit(Some(encoder.finish()));

    }

}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    println!("Hello, world!");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(window).await;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                state.render();
            }
            _ => {}
        }
    })
}