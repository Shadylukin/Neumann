//! Build script for compiling protobuf definitions.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "proto/neumann.proto";
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);

    // Recompile if proto changes
    println!("cargo:rerun-if-changed={proto_file}");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        // Generate file descriptor set for reflection service
        .file_descriptor_set_path(out_dir.join("neumann_descriptor.bin"))
        .compile_protos(&[proto_file], &["proto/"])?;

    Ok(())
}
