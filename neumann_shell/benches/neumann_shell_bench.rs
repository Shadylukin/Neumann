use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neumann_shell::Shell;
use peak_alloc::PeakAlloc;

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn bench_execute_commands(c: &mut Criterion) {
    let mut group = c.benchmark_group("shell_execute");

    group.bench_function("empty_input", |b| {
        PEAK_ALLOC.reset_peak_usage();
        b.iter_batched(
            Shell::new,
            |mut shell| {
                let result = shell.execute(black_box(""));
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        );
        println!("  empty_input peak RAM: {} KB", PEAK_ALLOC.peak_usage_as_kb());
    });

    group.bench_function("help", |b| {
        PEAK_ALLOC.reset_peak_usage();
        b.iter_batched(
            Shell::new,
            |mut shell| {
                let result = shell.execute(black_box("help"));
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        );
        println!("  help peak RAM: {} KB", PEAK_ALLOC.peak_usage_as_kb());
    });

    group.bench_function("select_all", |b| {
        PEAK_ALLOC.reset_peak_usage();
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
        println!("  select_all peak RAM: {} KB", PEAK_ALLOC.peak_usage_as_kb());
    });

    group.bench_function("select_where", |b| {
        PEAK_ALLOC.reset_peak_usage();
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
        println!("  select_where peak RAM: {} KB", PEAK_ALLOC.peak_usage_as_kb());
    });

    group.finish();
}

fn bench_format_output(c: &mut Criterion) {
    let mut group = c.benchmark_group("shell_format");

    group.bench_with_input(
        BenchmarkId::new("format_rows", 1000),
        &1000,
        |b, &row_count| {
            PEAK_ALLOC.reset_peak_usage();
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
            println!("  format_{}_rows peak RAM: {} KB", row_count, PEAK_ALLOC.peak_usage_as_kb());
        },
    );

    group.finish();
}

fn bench_insert_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("shell_insert");

    for size in [100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("rows", size), &size, |b, &size| {
            PEAK_ALLOC.reset_peak_usage();
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
            println!("  insert_{} peak RAM: {} KB", size, PEAK_ALLOC.peak_usage_as_kb());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_execute_commands, bench_format_output, bench_insert_scaling);
criterion_main!(benches);
