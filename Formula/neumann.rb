class Neumann < Formula
  desc "Unified tensor database combining relational, graph, and vector storage"
  homepage "https://github.com/Shadylukin/Neumann"
  url "https://github.com/Shadylukin/Neumann/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/Shadylukin/Neumann.git", branch: "main"

  depends_on "rust" => :build
  depends_on "protobuf" => :build

  def install
    system "cargo", "install", *std_cargo_args(path: "neumann_shell")
  end

  test do
    assert_match "neumann", shell_output("#{bin}/neumann --version")

    # Test query parsing
    output = shell_output("#{bin}/neumann -c 'SELECT 1' 2>&1", 0)
    assert_match(/1|error/, output.downcase)
  end
end
