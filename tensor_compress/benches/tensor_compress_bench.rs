// SPDX-License-Identifier: MIT OR Apache-2.0
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use peak_alloc::PeakAlloc;
use tensor_compress::{
    compress_ids, decompress_ids, rle_decode, rle_encode, tt_cosine_similarity, tt_decompose,
    tt_dot_product, tt_reconstruct, TTConfig,
};

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn bench_delta_compress(c: &mut Criterion) {
    let ids: Vec<u64> = (0..10_000).collect();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("compress_ids_10k_sequential", |b| {
        b.iter(|| compress_ids(black_box(&ids)))
    });
    println!(
        "  compress_ids_10k peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_delta_decompress(c: &mut Criterion) {
    let ids: Vec<u64> = (0..10_000).collect();
    let compressed = compress_ids(&ids);

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("decompress_ids_10k_sequential", |b| {
        b.iter(|| decompress_ids(black_box(&compressed)))
    });
    println!(
        "  decompress_ids_10k peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_rle_encode(c: &mut Criterion) {
    let data: Vec<i32> = (0..1000).flat_map(|i| vec![i % 10; 100]).collect();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("rle_encode_100k_values", |b| {
        b.iter(|| rle_encode(black_box(&data)))
    });
    println!(
        "  rle_encode_100k peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_rle_decode(c: &mut Criterion) {
    let data: Vec<i32> = (0..1000).flat_map(|i| vec![i % 10; 100]).collect();
    let encoded = rle_encode(&data);

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("rle_decode_100k_values", |b| {
        b.iter(|| rle_decode(black_box(&encoded)))
    });
    println!(
        "  rle_decode_100k peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

// Tensor Train (TT) decomposition benchmarks

fn bench_tt_decompose_256d(c: &mut Criterion) {
    let vector: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
    let config = TTConfig::for_dim(256).unwrap();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("tt_decompose_256d", |b| {
        b.iter(|| tt_decompose(black_box(&vector), black_box(&config)))
    });
    println!(
        "\n  tt_decompose_256d peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_tt_decompose_1024d(c: &mut Criterion) {
    let vector: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
    let config = TTConfig::for_dim(1024).unwrap();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("tt_decompose_1024d", |b| {
        b.iter(|| tt_decompose(black_box(&vector), black_box(&config)))
    });
    println!(
        "  tt_decompose_1024d peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_tt_decompose_4096d(c: &mut Criterion) {
    let vector: Vec<f32> = (0..4096).map(|i| (i as f32).sin()).collect();
    let config = TTConfig::for_dim(4096).unwrap();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("tt_decompose_4096d", |b| {
        b.iter(|| tt_decompose(black_box(&vector), black_box(&config)))
    });
    println!(
        "  tt_decompose_4096d peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_tt_reconstruct_4096d(c: &mut Criterion) {
    let vector: Vec<f32> = (0..4096).map(|i| (i as f32).sin()).collect();
    let config = TTConfig::for_dim(4096).unwrap();
    let tt = tt_decompose(&vector, &config).unwrap();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("tt_reconstruct_4096d", |b| {
        b.iter(|| tt_reconstruct(black_box(&tt)))
    });
    println!(
        "  tt_reconstruct_4096d peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_tt_dot_product_4096d(c: &mut Criterion) {
    let v1: Vec<f32> = (0..4096).map(|i| (i as f32).sin()).collect();
    let v2: Vec<f32> = (0..4096).map(|i| (i as f32).cos()).collect();
    let config = TTConfig::for_dim(4096).unwrap();
    let tt1 = tt_decompose(&v1, &config).unwrap();
    let tt2 = tt_decompose(&v2, &config).unwrap();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("tt_dot_product_4096d", |b| {
        b.iter(|| tt_dot_product(black_box(&tt1), black_box(&tt2)))
    });
    println!(
        "  tt_dot_product_4096d peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_tt_cosine_similarity_4096d(c: &mut Criterion) {
    let v1: Vec<f32> = (0..4096).map(|i| (i as f32).sin()).collect();
    let v2: Vec<f32> = (0..4096).map(|i| (i as f32).cos()).collect();
    let config = TTConfig::for_dim(4096).unwrap();
    let tt1 = tt_decompose(&v1, &config).unwrap();
    let tt2 = tt_decompose(&v2, &config).unwrap();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("tt_cosine_similarity_4096d", |b| {
        b.iter(|| tt_cosine_similarity(black_box(&tt1), black_box(&tt2)))
    });
    println!(
        "  tt_cosine_similarity_4096d peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );
}

fn bench_tt_compression_ratio(c: &mut Criterion) {
    let vector: Vec<f32> = (0..4096).map(|i| (i as f32).sin()).collect();
    let config = TTConfig::for_dim(4096).unwrap();

    c.bench_function("tt_compression_ratio_4096d", |b| {
        b.iter(|| {
            let tt = tt_decompose(black_box(&vector), black_box(&config)).unwrap();
            let ratio = tt.compression_ratio();
            black_box(ratio)
        })
    });
}

criterion_group!(
    benches,
    bench_delta_compress,
    bench_delta_decompress,
    bench_rle_encode,
    bench_rle_decode,
    bench_tt_decompose_256d,
    bench_tt_decompose_1024d,
    bench_tt_decompose_4096d,
    bench_tt_reconstruct_4096d,
    bench_tt_dot_product_4096d,
    bench_tt_cosine_similarity_4096d,
    bench_tt_compression_ratio,
);
criterion_main!(benches);
