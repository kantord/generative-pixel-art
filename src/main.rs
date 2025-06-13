// pixel_art_ga/src/main.rs
// Genetic‑algorithm pixel‑art converter — edition 2024–ready
// -----------------------------------------------------------------------------
// BUILD
//   cargo run --release -- \
//     --input original.jpg --output out.png \
//     --width 64 --height 64 --palette-size 16 \
//     --population 200 --generations 1000
// -----------------------------------------------------------------------------
// Crates
//   image   – load & save images
//   rand    – RNG (0.9 API: `random`, `random_range`, …)
//   rayon   – parallel fitness evaluation
//   clap    – CLI arg parsing
//   anyhow  – ergonomic errors
// -----------------------------------------------------------------------------
// Cargo.toml excerpt:
// [package]
// edition = "2024"
//
// [dependencies]
// image = "0.25"
// rand  = "0.9"
// rayon = "1.10"
// clap  = { version = "4.5", features = ["derive"] }
// anyhow = "1.0"

use anyhow::Result;
use clap::Parser;
use image::{GenericImageView, ImageBuffer, Rgb, RgbImage, imageops::{FilterType, blur}};
use rand::{Rng, rng, seq::SliceRandom, prelude::IndexedRandom};
use rayon::prelude::*;
use std::path::PathBuf;

// ---------------- CLI ---------------------------------------------------------
#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Genetic‑algorithm pixel‑art converter", long_about = None)]
struct Args {
    /// Path to the input photograph/illustration
    #[arg(short, long)]
    input: PathBuf,

    /// Path where the best solution PNG will be written
    #[arg(short, long)]
    output: PathBuf,

    /// Target pixel‑art width (px)
    #[arg(short = 'w', long, default_value_t = 64)]
    width: u32,

    /// Number of colours in the palette
    #[arg(short = 'p', long, default_value_t = 16)]
    palette_size: usize,

    /// Number of individuals in the population
    #[arg(short = 'n', long, default_value_t = 100)]
    population: usize,

    /// Number of generations to run
    #[arg(short = 'g', long, default_value_t = 50000)]
    generations: usize,

    /// Probability that any gene will mutate during mutation phase
    #[arg(short = 'm', long, default_value_t = 0.03)]
    mutation_rate: f32,
}

// ---------------- Genome & GA primitives -------------------------------------
#[derive(Clone)]
struct Individual {
    palette: Vec<[u8; 3]>, // length = palette_size
    image: Vec<[u8; 3]>,   // length = width * height
    fitness: f32,          // lower == better (mean absolute error)
    width: u32,
    height: u32,
}

impl Individual {
    fn random<R: Rng>(rng: &mut R, palette_size: usize, width: u32, height: u32) -> Self {
        let palette: Vec<[u8; 3]> = (0..palette_size)
            .map(|_| [rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()])
            .collect();
        
        let n_pixels = (width * height) as usize;
        let image: Vec<[u8; 3]> = (0..n_pixels)
            .map(|_| palette[rng.random_range(0..palette_size)])
            .collect();

        Self {
            palette,
            image,
            fitness: f32::MAX,
            width,
            height,
        }
    }

    /// Compute mean absolute error against target pixels
    fn evaluate(&mut self, target: &[[u8; 3]]) {
        let mut err: f32 = 0.0;
        for (p, t) in self.image.iter().zip(target.iter()) {
            let dr = (p[0] as f32 - t[0] as f32).abs();
            let dg = (p[1] as f32 - t[1] as f32).abs();
            let db = (p[2] as f32 - t[2] as f32).abs();
            err += (dr + dg + db) / 3.0;
        }
        self.fitness = err / target.len() as f32;
    }

    /// Mutate image pixels
    fn mutate_image<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        let n_pixels = (self.width * self.height) as usize;
        let pixels_to_mutate = (n_pixels as f32 * rate).round() as usize;
        
        for _ in 0..pixels_to_mutate {
            let idx = rng.random_range(0..n_pixels);
            self.image[idx] = self.palette[rng.random_range(0..self.palette.len())];
        }
    }

    /// Mutate palette colors
    fn mutate_palette<R: Rng>(&mut self, rng: &mut R) {
        if rng.random::<f32>() < 0.2 { // 1/5 chance to mutate palette
            let idx = rng.random_range(0..self.palette.len());
            let old_color = self.palette[idx];
            let new_color = [
                rng.random::<u8>(),
                rng.random::<u8>(),
                rng.random::<u8>(),
            ];
            self.palette[idx] = new_color;

            // Update all pixels using the old color
            for pixel in &mut self.image {
                if pixel == &old_color {
                    *pixel = new_color;
                }
            }
        }
    }

    /// Mutate both image and palette
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.mutate_image(rng, rate);
        self.mutate_palette(rng);
    }

    /// Render individual into RgbImage
    fn render(&self) -> RgbImage {
        let mut im = ImageBuffer::new(self.width, self.height);
        for (i, pixel) in im.pixels_mut().enumerate() {
            *pixel = Rgb(self.image[i]);
        }
        im
    }
}

// ---------------- Genetic algorithm driver -----------------------------------
fn evolve(args: &Args, target: &[[u8; 3]], width: u32, height: u32) -> Individual {
    let mut rng = rng();
    let mut population: Vec<Individual> = (0..args.population)
        .map(|_| Individual::random(&mut rng, args.palette_size, width, height))
        .collect();

    // Evaluate initial pop
    population
        .par_iter_mut()
        .for_each(|ind| ind.evaluate(target));

    for generation in 0..args.generations {
        // Sort ascending fitness
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        if generation % 100 == 0 {
            println!("gen {generation}: best mae {:.2}", population[0].fitness);
        }

        // Early stop if perfect match
        if population[0].fitness == 0.0 {
            break;
        }

        // Keep best individual and mutate copies
        let best = population[0].clone();
        population = (0..args.population)
            .map(|_| {
                let mut child = best.clone();
                child.mutate(&mut rng, args.mutation_rate);
                child
            })
            .collect();

        // Evaluate new population
        population
            .par_iter_mut()
            .for_each(|ind| ind.evaluate(target));
    }

    population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    population.remove(0)
}

// ---------------- Utility -----------------------------------------------------
fn load_and_resize(path: &PathBuf, target_width: u32) -> Result<(Vec<[u8; 3]>, u32, u32)> {
    let img = image::open(path)?;
    let (orig_width, orig_height) = img.dimensions();
    let aspect_ratio = orig_height as f32 / orig_width as f32;
    let target_height = (target_width as f32 * aspect_ratio).round() as u32;
    
    // Apply Gaussian blur and resize
    let img = img.resize_exact(target_width * 2, target_height * 2, FilterType::Gaussian)
        .resize_exact(target_width, target_height, FilterType::Gaussian);
    
    let mut v = Vec::with_capacity((target_width * target_height) as usize);
    for (_, _, p) in img.pixels() {
        v.push([p[0], p[1], p[2]]);
    }
    Ok((v, target_width, target_height))
}

// ---------------- MAIN --------------------------------------------------------
fn main() -> Result<()> {
    let args = Args::parse();
    let (target_pixels, width, height) = load_and_resize(&args.input, args.width)?;
    println!("Resizing to {}x{} pixels", width, height);

    let best = evolve(&args, &target_pixels, width, height);

    // Save rendered result
    let rendered = best.render();
    rendered.save(&args.output)?;

    // Also write the palette as a horizontal strip
    let palette_img = ImageBuffer::from_fn(args.palette_size as u32 * 8, 8, |x, _y| {
        let idx = (x / 8) as usize;
        let [r, g, b] = best.palette[idx];
        Rgb([r, g, b])
    });
    let palette_path = args.output.with_file_name("palette.png");
    palette_img.save(&palette_path)?;

    println!(
        "Done! Best MAE: {:.2}. Output: {}   Palette: {}",
        best.fitness,
        args.output.display(),
        palette_path.display()
    );
    Ok(())
}
