// SPDX-License-Identifier: MIT OR Apache-2.0
//! Build script for compiling protobuf definitions.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the proto file from neumann_server
    let proto_file = "../neumann_server/proto/neumann.proto";

    // Only compile if remote feature is enabled
    #[cfg(feature = "remote")]
    {
        // Recompile if proto changes
        println!("cargo:rerun-if-changed={proto_file}");

        tonic_build::configure()
            .build_server(false)
            .build_client(true)
            .compile_protos(&[proto_file], &["../neumann_server/proto/"])?;
    }

    // Always rerun if this build script changes
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}
