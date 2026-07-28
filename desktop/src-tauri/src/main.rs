// ArchitectOS desktop shell.
//
// The shell is intentionally thin: it hosts the web view and nothing
// more. All engine communication happens over the WebSocket the Python
// process exposes, so the Rust layer carries no research logic and needs
// no rebuild when the engine changes.
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running ArchitectOS");
}
