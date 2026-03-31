// LiteLLM 2.0 - Rust Migration
// Fast. Safe. Rust.

use axum::{routing::post, Router};
use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    println!("🚀 LiteLLM 2.0 (Rust)");
    println!("📋 Status: Migration in progress");
    println!("📖 See: RUST_MIGRATION.md");
    
    let app = Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/health", axum::routing::get(health));
    
    let addr = SocketAddr::from(([0, 0, 0, 0], 4000));
    println!("🌐 Listening on {}", addr);
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn chat_completions() -> &'static str {
    "Migration in progress - See RUST_MIGRATION.md"
}

async fn health() -> &'static str {
    "OK"
}
