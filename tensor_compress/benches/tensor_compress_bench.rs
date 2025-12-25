use criterion::{black_box, criterion_group, criterion_main, Criterion};
use peak_alloc::PeakAlloc;
use tensor_compress::{
    compress_ids, decompress_ids, quantize_binary, quantize_int8, rle_decode, rle_encode,
};

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn bench_quantize_int8(c: &mut Criterion) {
    let vector: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0) - 0.5).collect();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("quantize_int8_768d", |b| {
        b.iter(|| quantize_int8(black_box(&vector)))
    });
    println!("\n  quantize_int8_768d peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());
}

fn bench_quantize_binary(c: &mut Criterion) {
    let vector: Vec<f32> = (0..768)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("quantize_binary_768d", |b| {
        b.iter(|| quantize_binary(black_box(&vector)))
    });
    println!("  quantize_binary_768d peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());
}

fn bench_delta_compress(c: &mut Criterion) {
    let ids: Vec<u64> = (0..10_000).collect();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("compress_ids_10k_sequential", |b| {
        b.iter(|| compress_ids(black_box(&ids)))
    });
    println!("  compress_ids_10k peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());
}

fn bench_delta_decompress(c: &mut Criterion) {
    let ids: Vec<u64> = (0..10_000).collect();
    let compressed = compress_ids(&ids);

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("decompress_ids_10k_sequential", |b| {
        b.iter(|| decompress_ids(black_box(&compressed)))
    });
    println!("  decompress_ids_10k peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());
}

fn bench_rle_encode(c: &mut Criterion) {
    let data: Vec<i32> = (0..1000).flat_map(|i| vec![i % 10; 100]).collect();

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("rle_encode_100k_values", |b| {
        b.iter(|| rle_encode(black_box(&data)))
    });
    println!("  rle_encode_100k peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());
}

fn bench_rle_decode(c: &mut Criterion) {
    let data: Vec<i32> = (0..1000).flat_map(|i| vec![i % 10; 100]).collect();
    let encoded = rle_encode(&data);

    PEAK_ALLOC.reset_peak_usage();
    c.bench_function("rle_decode_100k_values", |b| {
        b.iter(|| rle_decode(black_box(&encoded)))
    });
    println!("  rle_decode_100k peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());
}

criterion_group!(
    benches,
    bench_quantize_int8,
    bench_quantize_binary,
    bench_delta_compress,
    bench_delta_decompress,
    bench_rle_encode,
    bench_rle_decode,
);
criterion_main!(benches);
