use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neumann_shell::Shell;

fn bench_execute_commands(c: &mut Criterion) {
    let mut group = c.benchmark_group("shell_execute");

    let shell = Shell::new();

    // Setup
    shell.execute("CREATE TABLE users (id INT, name TEXT)");
    for i in 0..100 {
        shell.execute(&format!("INSERT INTO users VALUES ({i}, 'user{i}')"));
    }

    group.bench_function("empty_input", |b| {
        b.iter(|| {
            let result = shell.execute(black_box(""));
            black_box(result);
        });
    });

    group.bench_function("help", |b| {
        b.iter(|| {
            let result = shell.execute(black_box("help"));
            black_box(result);
        });
    });

    group.bench_function("select_all", |b| {
        b.iter(|| {
            let result = shell.execute(black_box("SELECT * FROM users"));
            black_box(result);
        });
    });

    group.bench_function("select_where", |b| {
        b.iter(|| {
            let result = shell.execute(black_box("SELECT * FROM users WHERE id = 50"));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_format_output(c: &mut Criterion) {
    let mut group = c.benchmark_group("shell_format");

    let shell = Shell::new();
    shell.execute("CREATE TABLE bench (id INT, name TEXT, email TEXT)");
    for i in 0..1000 {
        shell.execute(&format!(
            "INSERT INTO bench VALUES ({i}, 'user{i}', 'user{i}@example.com')"
        ));
    }

    group.bench_function("format_1000_rows", |b| {
        b.iter(|| {
            let result = shell.execute(black_box("SELECT * FROM bench"));
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_execute_commands, bench_format_output);
criterion_main!(benches);
