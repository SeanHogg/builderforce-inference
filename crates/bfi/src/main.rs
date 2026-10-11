fn main() {
    if let Err(e) = bfi::run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
