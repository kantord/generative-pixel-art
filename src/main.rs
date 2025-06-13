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
use image::{GenericImageView, ImageBuffer, Rgb, RgbImage, imageops::FilterType};
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
    #[arg(short = 'n', long, default_value_t = 200)]
    population: usize,

    /// Number of generations to run
    #[arg(short = 'g', long, default_value_t = 1000)]
    generations: usize,

    /// Probability that any gene will mutate during mutation phase
    #[arg(short = 'm', long, default_value_t = 0.05)]
    mutation_rate: f32,

    /// Fraction of top individuals preserved each generation (elitism)
    #[arg(short = 'e', long, default_value_t = 0.05)]
    elite_rate: f32,
}

// ---------------- Genome & GA primitives -------------------------------------
#[derive(Clone)]
struct Individual {
    palette: Vec<[u8; 3]>, // length = palette_size
    pixels: Vec<u8>,       // length = width * height, each value 0..palette_size‑1
    fitness: f32,          // lower == better (mean‑squared error)
    width: u32,
    height: u32,
}

impl Individual {
    fn random<R: Rng>(rng: &mut R, palette_size: usize, width: u32, height: u32) -> Self {
        let palette = (0..palette_size)
            .map(|_| [rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()])
            .collect();
        let pixels = (0..(width * height) as usize)
            .map(|_| rng.random_range(0..palette_size as u8))
            .collect();
        Self {
            palette,
            pixels,
            fitness: f32::MAX,
            width,
            height,
        }
    }

    /// Compute mean‑squared error against target pixels (pre‑resized to same W×H).
    fn evaluate(&mut self, target: &[[u8; 3]]) {
        let mut err: f32 = 0.0;
        for (i, &idx) in self.pixels.iter().enumerate() {
            let p = self.palette[idx as usize];
            let t = target[i];
            let dr = p[0] as f32 - t[0] as f32;
            let dg = p[1] as f32 - t[1] as f32;
            let db = p[2] as f32 - t[2] as f32;
            err += (dr * dr + dg * dg + db * db) / 3.0;
        }
        self.fitness = err / target.len() as f32;
    }

    /// Uniform crossover on palette & pixels.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let palette: Vec<[u8; 3]> = self
            .palette
            .iter()
            .zip(&other.palette)
            .map(|(a, b)| if rng.random_bool(0.5) { *a } else { *b })
            .collect();
        let pixels: Vec<u8> = self
            .pixels
            .iter()
            .zip(&other.pixels)
            .map(|(a, b)| if rng.random_bool(0.5) { *a } else { *b })
            .collect();
        Self {
            palette,
            pixels,
            fitness: f32::MAX,
            width: self.width,
            height: self.height,
        }
    }

    /// Mutate palette channels and/or pixel indices.
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32, palette_size: usize) {
        // Mutate palette colors with smaller changes
        for chan in self.palette.iter_mut().flat_map(|rgb| rgb.iter_mut()) {
            if rng.random::<f32>() < rate {
                let delta: i16 = rng.random_range(-8..=8); // smaller tweak
                *chan = (*chan as i16 + delta).clamp(0, 255) as u8;
            }
        }
        // Mutate pixel indices less frequently
        for idx in &mut self.pixels {
            if rng.random::<f32>() < rate * 0.5 {
                *idx = rng.random_range(0..palette_size as u8);
            }
        }
    }

    /// Render individual into RgbImage.
    fn render(&self) -> RgbImage {
        let mut im = ImageBuffer::new(self.width, self.height);
        for (i, pixel) in im.pixels_mut().enumerate() {
            let [r, g, b] = self.palette[self.pixels[i] as usize];
            *pixel = Rgb([r, g, b]);
        }
        im
    }
}

// ---------------- Genetic algorithm driver -----------------------------------
fn evolve(args: &Args, target: &[[u8; 3]], width: u32, height: u32) -> Individual {
    let n_pixels = (width * height) as usize;
    let mut rng = rng();
    let mut population: Vec<Individual> = (0..args.population)
        .map(|_| Individual::random(&mut rng, args.palette_size, width, height))
        .collect();

    // Evaluate initial pop
    population
        .par_iter_mut()
        .for_each(|ind| ind.evaluate(target));

    let elite_n = ((args.elite_rate * args.population as f32).ceil() as usize).max(1);

    for generation in 0..args.generations {
        // Sort ascending fitness
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        if generation % 50 == 0 {
            println!("gen {generation}: best mse {:.2}", population[0].fitness);
        }

        // Early stop if perfect match
        if population[0].fitness == 0.0 {
            break;
        }

        // Preserve elites
        let mut next = population[..elite_n].to_vec();

        // Create rest via tournament selection + crossover
        while next.len() < args.population {
            // Tournament selection - pick 3 random individuals and take the best
            let tournament_size = 3;
            let mut tournament = Vec::with_capacity(tournament_size);
            for _ in 0..tournament_size {
                tournament.push(population.as_slice().choose(&mut rng).unwrap());
            }
            tournament.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
            let p1 = tournament[0];
            
            // Second parent from another tournament
            tournament.clear();
            for _ in 0..tournament_size {
                tournament.push(population.as_slice().choose(&mut rng).unwrap());
            }
            tournament.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
            let p2 = tournament[0];

            let mut child = p1.crossover(p2, &mut rng);
            child.mutate(&mut rng, args.mutation_rate, args.palette_size);
            next.push(child);
        }
        population = next;
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
    
    let img = img.resize_exact(target_width, target_height, FilterType::Nearest);
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
        "Done! Best MSE: {:.2}. Output: {}   Palette: {}",
        best.fitness,
        args.output.display(),
        palette_path.display()
    );
    Ok(())
}
