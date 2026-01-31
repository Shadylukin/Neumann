class Neumann < Formula
  desc "Unified tensor database combining relational, graph, and vector storage"
  homepage "https://github.com/Shadylukin/Neumann"
  version "0.1.0"
  license any_of: ["MIT", "Apache-2.0"]

  on_macos do
    on_arm do
      url "https://github.com/Shadylukin/Neumann/releases/download/v0.1.0/neumann-v0.1.0-aarch64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER_ARM64_SHA256"
    end
    on_intel do
      url "https://github.com/Shadylukin/Neumann/releases/download/v0.1.0/neumann-v0.1.0-x86_64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER_X64_SHA256"
    end
  end

  on_linux do
    url "https://github.com/Shadylukin/Neumann/releases/download/v0.1.0/neumann-v0.1.0-x86_64-unknown-linux-gnu.tar.gz"
    sha256 "PLACEHOLDER_LINUX_SHA256"
  end

  def install
    bin.install "neumann"
    bash_completion.install "completions/neumann.bash" => "neumann"
    zsh_completion.install "completions/_neumann"
    fish_completion.install "completions/neumann.fish"
    man1.install "man/neumann.1"
  end

  test do
    assert_match "neumann #{version}", shell_output("#{bin}/neumann --version")
    # Test command execution
    output = shell_output("#{bin}/neumann -c 'SELECT 1' --no-boot 2>&1")
    assert_match(/ok|error/i, output.downcase)
  end
end
