//! Crate root (module path `crate`).
//! This doc comment mentions `use crate::ignored::Thing;` but it is a COMMENT
//! and must NOT be detected as an edge.

mod a; // file-backed -> src/a/mod.rs  (module `crate::a`)
mod b; // file-backed -> src/b.rs       (module `crate::b`)

// Grouped/nested use with `self` in the group: licenses both `crate::a` (self)
// and `crate::a::c` (parent module of the `c::Helper` leaf, which is a real node).
use crate::a::{c::Helper, self};

// External crate root -> never resolves to an intra-crate node (NOT in edges.json).
use std::collections::HashMap;

// Re-export: still a real edge to `crate::b`.
pub use crate::b::ReExported;

pub fn run() {
    let _ = HashMap::<u32, u32>::new();
    let _ = Helper::default();
    ReExported::go();
}
