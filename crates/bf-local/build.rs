fn main() {
    println!(
        "cargo:rustc-env=BFI_TARGET_TRIPLE={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".into())
    );
}
