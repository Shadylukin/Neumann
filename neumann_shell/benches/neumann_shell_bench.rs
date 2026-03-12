// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(missing_docs)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neumann_shell::Shell;
use peak_alloc::PeakAlloc;

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn bench_execute_commands(c: &mut Criterion) {
    let mut group = c.benchmark_group("shell_execute");

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("empty_input", |b| {
        b.iter_batched(
            Shell::new,
            |mut shell| {
                let result = shell.execute(black_box(""));
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    println!(
        "\n  empty_input peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("help", |b| {
        b.iter_batched(
            Shell::new,
            |mut shell| {
                let result = shell.execute(black_box("help"));
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    println!("  help peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("select_all", |b| {
        b.iter_batched(
            || {
                let mut shell = Shell::new();
                let _ = shell.execute("CREATE TABLE users (id INT, name TEXT)");
                for i in 0..100 {
                    let _ = shell.execute(&format!("INSERT INTO users VALUES ({i}, 'user{i}')"));
                }
                shell
            },
            |mut shell| {
                let result = shell.execute(black_box("SELECT * FROM users"));
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    println!(
        "  select_all peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("select_where", |b| {
        b.iter_batched(
            || {
                let mut shell = Shell::new();
                let _ = shell.execute("CREATE TABLE users (id INT, name TEXT)");
                for i in 0..100 {
                    let _ = shell.execute(&format!("INSERT INTO users VALUES ({i}, 'user{i}')"));
                }
                shell
            },
            |mut shell| {
                let result = shell.execute(black_box("SELECT * FROM users WHERE id = 50"));
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    println!(
        "  select_where peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    group.finish();
}

fn bench_format_output(c: &mut Criterion) {
    let mut group = c.benchmark_group("shell_format");

    PEAK_ALLOC.reset_peak_usage();
    group.bench_with_input(
        BenchmarkId::new("format_rows", 1000),
        &1000,
        |b, &row_count| {
            b.iter_batched(
                || {
                    let mut shell = Shell::new();
                    let _ = shell.execute("CREATE TABLE bench (id INT, name TEXT, email TEXT)");
                    for i in 0..row_count {
                        let _ = shell.execute(&format!(
                            "INSERT INTO bench VALUES ({i}, 'user{i}', 'user{i}@example.com')"
                        ));
                    }
                    shell
                },
                |mut shell| {
                    let result = shell.execute(black_box("SELECT * FROM bench"));
                    black_box(result)
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );
    println!(
        "\n  format_1000_rows peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    group.finish();
}

fn bench_insert_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("shell_insert");

    for size in [100, 500, 1000] {
        PEAK_ALLOC.reset_peak_usage();
        group.bench_with_input(BenchmarkId::new("rows", size), &size, |b, &size| {
            b.iter_batched(
                Shell::new,
                |mut shell| {
                    let _ = shell.execute("CREATE TABLE test (id INT, data TEXT)");
                    for i in 0..size {
                        let _ = shell.execute(&format!("INSERT INTO test VALUES ({i}, 'data{i}')"));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
        println!(
            "\n  insert_{} peak RAM: {:.1} KB",
            size,
            PEAK_ALLOC.peak_usage_as_kb()
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_execute_commands,
    bench_format_output,
    bench_insert_scaling
);
criterion_main!(benches);
