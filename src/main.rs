use glam::{IVec3, Vec2, Vec3};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use std::cmp::{max, min};
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::iter::FromIterator;
use std::path::Path;

struct TriangleMesh {
    vertices: Vec<Vec2>,
    indices: Vec<IVec3>,
    colors: Vec<Vec3>,
}

fn normal(v: &Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}

struct DTriangleMesh {
    vertices: Vec<Vec2>,
    colors: Vec<Vec3>,
}

impl DTriangleMesh {
    fn new(num_vertices: i32, num_colors: i32) -> Self {
        let mut res = DTriangleMesh {
            vertices: Vec::<Vec2>::new(),
            colors: Vec::<Vec3>::new(),
        };
        res.vertices
            .resize(num_vertices as usize, Vec2::new(0f32, 0f32));
        res.colors
            .resize(num_colors as usize, Vec3::new(0f32, 0f32, 0f32));
        res
    }
}

#[derive(Eq, Hash, PartialEq, Ord, PartialOrd)]
struct Edge {
    v0: i32,
    v1: i32,
}

impl Edge {
    fn new(v0_: i32, v1_: i32) -> Self {
        let v0 = min(v0_, v1_);
        let v1 = max(v0_, v1_);
        Self { v0: v0, v1: v1 }
    }
}

struct Sampler {
    pmf: Vec<f32>,
    cdf: Vec<f32>,
}

fn build_edge_sampler(mesh: &TriangleMesh, edges: &Vec<Edge>) -> Sampler {
    let mut pmf = Vec::<f32>::new();
    let mut cdf = Vec::<f32>::new();
    pmf.reserve(edges.len());
    cdf.reserve(edges.len() + 1);
    cdf.push(0f32);
    for edge in edges {
        let v0 = mesh.vertices[edge.v0 as usize];
        let v1 = mesh.vertices[edge.v1 as usize];
        pmf.push((v1 - v0).length());
        cdf.push(cdf[cdf.len() - 1] + pmf[pmf.len() - 1]);
    }
    let length_sum = cdf[cdf.len() - 1];
    pmf = pmf.into_iter().map(|x| x / length_sum).collect();
    cdf = cdf.into_iter().map(|x| x / length_sum).collect();
    Sampler { pmf: pmf, cdf: cdf }
}

fn sample(sampler: &Sampler, u: f32) -> i32 {
    let cdf = &sampler.cdf;
    let mut i = 0;
    while cdf[i] < u {
        i += 1
    }
    (i as i32 - 1).clamp(0, cdf.len() as i32 - 2)
}

fn collect_edges(mesh: &TriangleMesh) -> Vec<Edge> {
    let mut set = HashSet::new();
    for index in &mesh.indices {
        set.insert(Edge::new(index.x, index.y));
        set.insert(Edge::new(index.y, index.z));
        set.insert(Edge::new(index.z, index.x));
    }
    let mut res = Vec::from_iter(set);
    res.sort();
    res
}
struct Img {
    color: Vec<Vec3>,
    width: i32,
    height: i32,
}

impl Img {
    fn new(width: i32, height: i32, val: Vec3) -> Self {
        let pixel_num = width * height;
        let buffer = vec![val; pixel_num as usize];
        Img {
            color: buffer,
            width: width,
            height: height,
        }
    }
}

fn save_img(img: &Img, filename: &str, flip: bool) -> io::Result<()> {
    let path = Path::new(filename);
    let mut f = File::create(path)?;
    let mut content = format!("P3\n{} {} 255\n", img.width, img.height);
    let pixel_count = (img.height * img.width) as usize;
    let tonemap = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as i32;
    for i in 0..pixel_count {
        let val = img.color[i];
        let val = if flip { -val } else { val };
        let val_s = format!("{} {} {} ", tonemap(val.x), tonemap(val.y), tonemap(val.z));
        content.push_str(&val_s);
    }
    f.write_all(content.as_bytes())
}

type MyRng = StdRng;

fn sample_1d(rng: &mut MyRng) -> f32 {
    let val: f32 = rng.gen();
    val
}

fn raytrace(mesh: &TriangleMesh, screen_pos: &Vec2) -> (Vec3, i32) {
    for (i, index) in mesh.indices.iter().enumerate() {
        let v0 = mesh.vertices[index.x as usize];
        let v1 = mesh.vertices[index.y as usize];
        let v2 = mesh.vertices[index.z as usize];
        let n01 = normal(&(v1 - v0));
        let n12 = normal(&(v2 - v1));
        let n20 = normal(&(v0 - v2));
        let side01 = (*screen_pos - v0).dot(n01) > 0.;
        let side12 = (*screen_pos - v1).dot(n12) > 0.;
        let side20 = (*screen_pos - v2).dot(n20) > 0.;
        if (side01 && side12 && side20) || (!side01 && !side12 && !side20) {
            //println!("{}  ",mesh.colors[i]);
            return (mesh.colors[i], i as i32);
        }
    }
    (Vec3::new(0., 0., 0.), -1)
}

fn render(mesh: &TriangleMesh, spp: i32, rng: &mut MyRng, img: &mut Img) {
    let sqrt_spp = ((spp as f32).sqrt()) as i32;
    let spp = sqrt_spp * sqrt_spp;
    for y in 0..img.height {
        for x in 0..img.width {
            for dy in 0..sqrt_spp {
                for dx in 0..sqrt_spp {
                    let xoff = (dx as f32 + sample_1d(rng)) / (sqrt_spp as f32);
                    let yoff = (dy as f32 + sample_1d(rng)) / (sqrt_spp as f32);
                    let screen_pos = Vec2::new(x as f32 + xoff, y as f32 + yoff);
                    let (color, _) = raytrace(mesh, &screen_pos);
                    img.color[(y * img.width + x) as usize] += color / (spp as f32);
                }
            }
        }
    }
}
fn compute_interior_derivatives(
    mesh: &TriangleMesh,
    adj: &Img,
    ispp: i32,
    rng: &mut MyRng,
    d_colors: &mut Vec<Vec3>,
) {
    let sqrt_spp = ((ispp as f32).sqrt()) as i32;
    let spp = sqrt_spp * sqrt_spp;
    for y in 0..adj.height {
        for x in 0..adj.width {
            for dy in 0..sqrt_spp {
                for dx in 0..sqrt_spp {
                    let xoff = (dx as f32 + sample_1d(rng)) / (sqrt_spp as f32);
                    let yoff = (dy as f32 + sample_1d(rng)) / (sqrt_spp as f32);
                    let screen_pos = Vec2::new(x as f32 + xoff, y as f32 + yoff);
                    let (_, idx) = raytrace(mesh, &screen_pos);
                    if idx >= 0 {
                        d_colors[idx as usize] +=
                            adj.color[(y * adj.width + x) as usize] / (spp as f32);
                    }
                }
            }
        }
    }
}

fn compute_edge_derivatives(
    mesh: &TriangleMesh,
    edges: &Vec<Edge>,
    edge_sampler: &Sampler,
    adj: &Img,
    espp: i32,
    rng: &mut MyRng,
    screen_dx: &mut Img,
    screen_dy: &mut Img,
    d_vertices: &mut Vec<Vec2>,
) {
    for _ in 0..espp {
        let edge_id = sample(&edge_sampler, sample_1d(rng));
        let edge = &edges[edge_id as usize];
        let pmf = &edge_sampler.pmf[edge_id as usize];
        let v0 = mesh.vertices[edge.v0 as usize];
        let v1 = mesh.vertices[edge.v1 as usize];
        let t = sample_1d(rng);
        let p = v0 + t * (v1 - v0);
        let xi = p.x as i32;
        let yi = p.y as i32;
        if xi < 0 || yi < 0 || xi >= adj.width || yi >= adj.height {
            continue;
        }
        let n = normal(&(v1 - v0).normalize());
        let (color_in, _) = raytrace(mesh, &(p - 1e-3f32 * n));
        let (color_out, _) = raytrace(mesh, &(p + 1e-3f32 * n));
        let pdf = pmf / ((v1 - v0).length());
        let weight = 1.0 / (pdf * espp as f32);
        let adj_v = (color_in - color_out).dot(adj.color[(yi * adj.width + xi) as usize]);
        let d_v0 = adj_v * weight * Vec2::new((1.0 - t) * n.x, (1.0 - t) * n.y);
        let d_v1 = adj_v * weight * Vec2::new(t * n.x, t * n.y);
        let dx = -n.x * (color_in - color_out) * weight;
        let dy = -n.y * (color_in - color_out) * weight;
        let idx = (yi * adj.width + xi) as usize;
        screen_dx.color[idx] += dx;
        screen_dy.color[idx] += dy;
        d_vertices[edge.v0 as usize] += d_v0;
        d_vertices[edge.v1 as usize] += d_v1;
    }
}
fn d_render(
    mesh: &TriangleMesh,
    adj: &Img,
    ispp: i32,
    espp: i32,
    rng: &mut MyRng,
    screen_dx: &mut Img,
    screen_dy: &mut Img,
    d_mesh: &mut DTriangleMesh,
) {
    compute_interior_derivatives(mesh, adj, ispp, rng, &mut d_mesh.colors);
    let edges = collect_edges(mesh);
    let edge_sampler = build_edge_sampler(mesh, &edges);
    compute_edge_derivatives(
        mesh,
        &edges,
        &edge_sampler,
        adj,
        espp,
        rng,
        screen_dx,
        screen_dy,
        &mut d_mesh.vertices,
    );
}

fn main() {
    let mesh = TriangleMesh {
        vertices: vec![
            Vec2::new(50.0, 25.0),
            Vec2::new(200.0, 200.0),
            Vec2::new(15.0, 150.0),
            Vec2::new(200.0, 15.0),
            Vec2::new(150.0, 250.0),
            Vec2::new(50.0, 100.0),
        ],
        indices: vec![IVec3::new(0, 1, 2), IVec3::new(3, 4, 5)],
        colors: vec![Vec3::new(0.3, 0.5, 0.3), Vec3::new(0.3, 0.3, 0.5)],
    };

    let mut rng = StdRng::seed_from_u64(123u64);
    let mut img = Img::new(256, 256, Vec3::new(0., 0., 0.));
    render(&mesh, 64, &mut rng, &mut img);
    let filename = "color.ppm";
    match save_img(&img, filename, false) {
        Err(why) => println!("{}", why),
        Ok(_) => println!("write success to {}", filename),
    };
    let adj_img = Img::new(256, 256, Vec3::new(1., 1., 1.));
    let mut dx = Img::new(256, 256, Vec3::new(0., 0., 0.));
    let mut dy = Img::new(256, 256, Vec3::new(0., 0., 0.));
    let mut d_mesh = DTriangleMesh::new(mesh.vertices.len() as i32, mesh.colors.len() as i32);
    d_render(
        &mesh,
        &adj_img,
        4,                      /* interior_samples_per_pixel */
        img.width * img.height, /* edge_samples_in_total */
        &mut rng,
        &mut dx,
        &mut dy,
        &mut d_mesh,
    );
    let filename = "dx_pos.ppm";
    match save_img(&dx, filename, false) {
        Err(why) => println!("{}", why),
        Ok(_) => println!("write success to {}", filename),
    };
    let filename = "dx_neg.ppm";
    match save_img(&dx, filename, true) {
        Err(why) => println!("{}", why),
        Ok(_) => println!("write success to {}", filename),
    };
    let filename = "dy_pos.ppm";
    match save_img(&dy, filename, false) {
        Err(why) => println!("{}", why),
        Ok(_) => println!("write success to {}", filename),
    };
    let filename = "dy_neg.ppm";
    match save_img(&dy, filename, true) {
        Err(why) => println!("{}", why),
        Ok(_) => println!("write success to {}", filename),
    };
}
