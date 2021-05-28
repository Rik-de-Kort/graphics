use bmp::{Image, Pixel};
use std::f64::consts;

mod basics;
use basics::*;
#[derive(Debug)]
struct Ray {
    origin: Point3,
    direction: Vector3,
}

fn pixel_value(luminance: Radiance3, k: f64, gamma: f64) -> Color3 {
    let mut result = luminance / k;

    let m = luminance.r.max(luminance.g.max(luminance.b)).max(1.0);
    result = result / m;

    let to_clamp = (m - 1.0) * 0.2;
    let m = if to_clamp < 0.0 {
        0.0
    } else if to_clamp > 1.0 {
        1.0
    } else {
        to_clamp
    };

    result = result * (1.0 - m) + Radiance3 { r: m, g: m, b: m };
    result = result.powf(1.0 / gamma);
    result.into()
}

// Surfaces
#[derive(PartialEq)]
struct Triangle([Point3; 3]);

impl Triangle {
    fn new(p0: Point3, p1: Point3, p2: Point3) -> Triangle {
        Triangle([p0, p1, p2])
    }

    fn zero() -> Triangle {
        Triangle::new(
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        )
    }
}

struct IndexedTriangleList {
    vertex_array: Vec<Point3>,
    index_array: Vec<usize>,
}

impl IndexedTriangleList {
    fn peek_faces(&self) {
        println!(
            "{:?}",
            self.index_array
                .iter()
                .take(9)
                .map(|i| self.vertex_array[*i])
                .collect::<Vec<_>>()
        );
    }

    fn linear_transform(&mut self, scale: f64, translate: Vector3) {
        for p in self.vertex_array.iter_mut() {
            *p = *p * scale + translate.into();
        }
    }

    fn from_file(path: &str) -> IndexedTriangleList {
        let contents = fs::read_to_string(path).expect("Oopsie doopsie");
        let lines = contents.split("\r\n");

        fn parse_vertex(s: &str) -> Point3 {
            let coords = s
                .split(' ')
                .filter(|p| (*p != "v") & (*p != ""))
                .map(|p| {
                    p.parse::<f64>()
                        .unwrap_or_else(|_| panic!("{} is not a float", p))
                })
                .collect::<Vec<f64>>();
            Point3::new(coords[0], coords[1], coords[2])
        }

        fn parse_face(s: &str) -> Vec<usize> {
            let vertices: Vec<usize> = s
                .split(' ')
                .filter(|p| (*p != "f") & (*p != ""))
                .map(|p| p.split('/').next().expect(&"empty line starting with 'f '"))
                .map(|p| {
                    p.parse::<usize>()
                        .unwrap_or_else(|_| panic!("{} is not a usize", p))
                })
                .map(|p| p - 1) // .obj is 1-indexed
                .collect();

            assert!((vertices.len() == 3) || (vertices.len() == 4));

            if vertices.len() == 4 {
                vec![
                    vertices[0],
                    vertices[1],
                    vertices[2],
                    vertices[0],
                    vertices[2],
                    vertices[3],
                ]
            } else {
                vertices
            }
        }

        let (vertices, other): (Vec<&str>, Vec<&str>) = lines.partition(|s| s.starts_with("v "));
        let (faces, _): (Vec<&str>, Vec<&str>) = other.iter().partition(|s| s.starts_with("f "));

        IndexedTriangleList {
            vertex_array: vertices.into_iter().map(parse_vertex).collect(),
            index_array: faces.into_iter().flat_map(parse_face).collect(),
        }
    }

    fn get_triangle(&self, index: usize) -> Triangle {
        assert!(index < self.index_array.len() - 2);
        Triangle::new(
            self.vertex_array[self.index_array[index]],
            self.vertex_array[self.index_array[index + 1]],
            self.vertex_array[self.index_array[index + 2]],
        )
    }

    fn find_first_intersection(&self, ray: &Ray) -> Option<IntersectionInfo> {
        let mut result = None;
        for i in (0..self.index_array.len() - 2).step_by(3) {
            // TODO: build iterator for this horrible shit
            // This iterator is below, but it performs like shit
            let triangle = self.get_triangle(i);
            if let Some(intersection) = ray_triangle_intersect(ray, &triangle) {
                result = match result {
                    None => Some(intersection),
                    Some(old) => {
                        if old.t < intersection.t {
                            Some(old)
                        } else {
                            Some(intersection)
                        }
                    }
                }
            }
        }
        result
    }
}

use std::iter::{IntoIterator, Iterator};

struct IndexedTriangleListIterator<'a> {
    underlying: &'a IndexedTriangleList,
    index: usize,
}

impl Iterator for IndexedTriangleListIterator<'_> {
    type Item = Triangle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.underlying.index_array.len() - 2 {
            None
        } else {
            Some(Triangle::new(
                self.underlying.vertex_array[self.underlying.index_array[self.index]],
                self.underlying.vertex_array[self.underlying.index_array[self.index + 1]],
                self.underlying.vertex_array[self.underlying.index_array[self.index + 2]],
            ))
        }
    }
}

impl<'a> IntoIterator for &'a IndexedTriangleList {
    type Item = Triangle;
    type IntoIter = IndexedTriangleListIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IndexedTriangleListIterator {
            underlying: &self,
            index: 0,
        }
    }
}

#[derive(Debug)]
struct IntersectionInfo {
    b: [f64; 3],
    t: f64,
}

fn ray_triangle_intersect(ray: &Ray, triangle: &Triangle) -> Option<IntersectionInfo> {
    let eps = 1e-6;

    // TODO: this syntax is horrible, better to impl Index?
    let edge_1 = triangle.0[1] - triangle.0[0];
    let edge_2 = triangle.0[2] - triangle.0[0];

    let normal = edge_1.cross(edge_2).normalize();
    // dbg!((normal * edge_1).abs());
    assert!((normal * edge_1).abs() < eps);
    // dbg!((normal * edge_2).abs());
    assert!((normal * edge_2).abs() < eps);
    // dbg!((normal * (triangle.0[2]-triangle.0[1])).abs());
    assert!((normal * (triangle.0[2] - triangle.0[1])).abs() < eps);
    // println!("Normal {:?}", normal);
    // println!("Direction {:?}", ray.direction);
    let q: Vector3 = ray.direction.cross(edge_2);
    let a: f64 = edge_1 * q;

    // Backfacing / nearly parallel, or close to limit of precision
    if (a.abs() <= eps) || (normal * ray.direction >= 0.0) {
        return None;
    }

    let s = (ray.origin - triangle.0[0]) / a;
    let r = s.cross(edge_1);

    let b_0 = s * q;
    let b_1 = r * ray.direction;
    let b_2 = 1.0 - b_0 - b_1;

    let t = edge_2 * r;

    // Intersected outside triangle
    if (b_0 < 0.0) || (b_1 < 0.0) || (b_2 < 0.0) || (t < 0.0) {
        return None;
    }

    Some(IntersectionInfo {
        b: [b_0, b_1, b_2],
        t,
    })
}

fn calculate_luminance(ray: &Ray, scene: &IndexedTriangleList) -> Radiance3 {
    let val = if scene.find_first_intersection(ray).is_some() {
        1.0
    } else {
        0.0
    };
    Radiance3 {
        r: val,
        g: val,
        b: val,
    }
}

// Camera model
struct PinholeCamera {
    z_near: f64,
    vertical_field_of_view: f64,
}

impl PinholeCamera {
    fn get_primary_ray(&self, x: f64, y: f64, width: u64, height: u64) -> Ray {
        let side = -2.0 * (self.vertical_field_of_view * consts::PI / 180.0 / 2.0).tan();

        let fwidth = width as f64;
        let fheight = height as f64;

        let p = Point3 {
            x: self.z_near * (x / fwidth - 0.5) * side * fwidth / fheight,
            y: self.z_near * -(y / fheight - 0.5) * side,
            z: self.z_near,
        };

        let w: Vector3 = p.into();
        Ray {
            origin: p,
            direction: w.normalize(),
        }
    }

    fn render(&self, mut img: Image, scene: &IndexedTriangleList) -> Image {
        let width: u64 = img.get_width().into();
        let height: u64 = img.get_height().into();
        for (x, y) in img.coordinates() {
            // TODO: work out this x and y coordinate type thing.
            let ray = self.get_primary_ray(x as f64 + 0.5, y as f64 + 0.5, width, height);
            let luminance = calculate_luminance(&ray, scene);

            img.set_pixel(x, y, luminance.into());
        }
        img
    }
}

impl From<Radiance3> for Pixel {
    fn from(col: Radiance3) -> Self {
        fn float_to_value(x: f64) -> u8 {
            if x < 0.0 {
                0
            } else if x > 1.0 {
                255
            } else {
                (x * 255.0).round() as u8
            }
        }
        Self {
            r: float_to_value(col.r),
            g: float_to_value(col.g),
            b: float_to_value(col.b),
        }
    }
}

use std::fs;

fn main() {
    let mut teapot = IndexedTriangleList::from_file("teapot.obj");
    teapot.linear_transform(
        0.042,
        Vector3 {
            x: 0.0,
            y: -1.5,
            z: -5.0,
        },
    );

    // teapot.peek_faces();

    let camera = PinholeCamera {
        z_near: -0.03,
        vertical_field_of_view: 45.0,
    };
    let _ = camera
        .render(Image::new(160, 100), &teapot)
        .save("result.bmp");
}
